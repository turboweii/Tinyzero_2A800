# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exact structural difficulty and adaptive sampling utilities for Countdown.

The curriculum deliberately uses only properties that can be proven by an
exact arithmetic solver. It does not count syntactically different expression
trees as different "solutions".
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache

import numpy as np
from torch.utils.data import Sampler

LEVELS = ("L0", "L1", "L2", "L3")


def _reachable(numbers: tuple[int, ...], allow_multiply: bool, allow_divide: bool) -> set[Fraction]:
    """Return exact values reachable while using every input exactly once."""
    size = len(numbers)
    dp: dict[int, set[Fraction]] = {}
    for index, number in enumerate(numbers):
        dp[1 << index] = {Fraction(number)}

    for subset_size in range(2, size + 1):
        for mask in range(1, 1 << size):
            if bin(mask).count("1") != subset_size:
                continue
            values: set[Fraction] = set()
            left_mask = (mask - 1) & mask
            while left_mask:
                right_mask = mask ^ left_mask
                if right_mask and left_mask < right_mask:
                    for left in dp[left_mask]:
                        for right in dp[right_mask]:
                            values.add(left + right)
                            values.add(left - right)
                            values.add(right - left)
                            if allow_multiply:
                                values.add(left * right)
                            if allow_divide:
                                if right != 0:
                                    values.add(left / right)
                                if left != 0:
                                    values.add(right / left)
                left_mask = (left_mask - 1) & mask
            dp[mask] = values
    return dp[(1 << size) - 1]


def classify_countdown_difficulty(numbers: tuple[int, ...], target: int) -> str:
    """Classify a 3/4-number task using exact, verifiable operation requirements.

    L0: 3 numbers and an addition/subtraction-only expression exists.
    L1: 3 numbers need multiplication but not division, or 4 numbers have an
        addition/subtraction-only expression.
    L2: 3 numbers require division, or 4 numbers need multiplication but not
        division.
    L3: 4 numbers require division.
    """
    return _classify_countdown_difficulty(
        tuple(sorted(int(number) for number in numbers)),
        int(target),
    )


@lru_cache(maxsize=None)
def _classify_countdown_difficulty(numbers: tuple[int, ...], target: int) -> str:
    target_value = Fraction(int(target))
    if len(numbers) not in (3, 4):
        raise ValueError(f"Countdown curriculum expects 3 or 4 numbers, got {numbers}")

    additive = target_value in _reachable(numbers, allow_multiply=False, allow_divide=False)
    if len(numbers) == 3 and additive:
        return "L0"
    if len(numbers) == 4 and additive:
        return "L1"

    without_division = target_value in _reachable(numbers, allow_multiply=True, allow_divide=False)
    if len(numbers) == 3:
        return "L1" if without_division else "L2"
    return "L2" if without_division else "L3"


def _bounded_normalize(weights: dict[str, float], minimum: float, maximum: float) -> dict[str, float]:
    """Project positive weights to a bounded simplex."""
    result = {level: max(minimum, min(maximum, float(weights[level]))) for level in LEVELS}
    for _ in range(16):
        total = sum(result.values())
        if abs(total - 1.0) < 1e-9:
            break
        if total < 1.0:
            adjustable = [level for level in LEVELS if result[level] < maximum - 1e-12]
            if not adjustable:
                break
            room = sum(maximum - result[level] for level in adjustable)
            for level in adjustable:
                result[level] += (1.0 - total) * (maximum - result[level]) / room
        else:
            adjustable = [level for level in LEVELS if result[level] > minimum + 1e-12]
            if not adjustable:
                break
            room = sum(result[level] - minimum for level in adjustable)
            for level in adjustable:
                result[level] -= (total - 1.0) * (result[level] - minimum) / room
    total = sum(result.values())
    return {level: result[level] / total for level in LEVELS}


def quota_from_weights(weights: dict[str, float], total: int) -> dict[str, int]:
    """Use largest remainders so per-level quotas sum exactly to total."""
    raw = {level: weights[level] * total for level in LEVELS}
    quota = {level: int(math.floor(raw[level])) for level in LEVELS}
    remainder = total - sum(quota.values())
    order = sorted(LEVELS, key=lambda level: raw[level] - quota[level], reverse=True)
    for level in order[:remainder]:
        quota[level] += 1
    return quota


class AdaptiveCurriculumBatchSampler(Sampler):
    """Sample batches by structural level and adapt toward mixed-correct groups."""

    def __init__(self, dataset, batch_size: int, config, seed: int = 42, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        initial = config.get("initial_weights", {"L0": 0.45, "L1": 0.30, "L2": 0.20, "L3": 0.05})
        self.weights = _bounded_normalize(
            {level: float(initial[level]) for level in LEVELS},
            minimum=float(config.get("min_weight", 0.05)),
            maximum=float(config.get("max_weight", 0.60)),
        )
        self.minimum = float(config.get("min_weight", 0.05))
        self.maximum = float(config.get("max_weight", 0.60))
        self.smoothing = float(config.get("smoothing", 0.20))
        self.max_change = float(config.get("max_change_per_update", 0.05))
        self.update_every = int(config.get("update_every", 20))
        self.warmup_steps = int(config.get("warmup_steps", 40))
        self.min_mixed_groups = int(config.get("min_mixed_groups", 16))

        self.level_by_index = self._load_or_compute_levels()
        self.pools = {level: np.flatnonzero(self.level_by_index == level).astype(np.int64).tolist() for level in LEVELS}
        for level, pool in self.pools.items():
            if not pool:
                raise ValueError(f"Countdown curriculum level {level} is empty")

        self.interval = {level: Counter(total=0, all_wrong=0, mixed=0, all_correct=0) for level in LEVELS}
        self.last_metrics: dict[str, float] = {}

    def _load_or_compute_levels(self) -> np.ndarray:
        dataframe = self.dataset.dataframe
        has_precomputed = "difficulty" in dataframe.columns and dataframe["difficulty"].isin(LEVELS).all()
        if has_precomputed:
            return dataframe["difficulty"].to_numpy(dtype=object)

        print("Countdown difficulty labels not found; computing exact structural levels once...")
        levels = []
        for position, (_, row) in enumerate(dataframe.iterrows(), 1):
            existing = row.get("difficulty", None)
            if existing in LEVELS:
                levels.append(existing)
                continue
            ground_truth = row["reward_model"]["ground_truth"]
            levels.append(classify_countdown_difficulty(
                tuple(ground_truth["numbers"]),
                int(ground_truth["target"]),
            ))
            if position % 50000 == 0:
                print(f"Computed Countdown difficulty for {position}/{len(dataframe)} rows")
        dataframe["difficulty"] = levels
        return np.asarray(levels, dtype=object)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def sample_level_indices(self, counts: dict[str, int]) -> list[int]:
        indices = []
        for level in LEVELS:
            count = int(counts.get(level, 0))
            if count <= 0:
                continue
            pool = self.pools[level]
            if count <= len(pool):
                indices.extend(self.rng.sample(pool, count))
            else:
                indices.extend(self.rng.choices(pool, k=count))
        self.rng.shuffle(indices)
        return indices

    def sample_batch_indices(self) -> list[int]:
        return self.sample_level_indices(quota_from_weights(self.weights, self.batch_size))

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch_indices()

    def record_group(self, level: str, correct_count: int, group_size: int) -> None:
        stats = self.interval[level]
        stats["total"] += 1
        if correct_count == 0:
            stats["all_wrong"] += 1
        elif correct_count == group_size:
            stats["all_correct"] += 1
        else:
            stats["mixed"] += 1

    def maybe_update(self, global_step: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for level in LEVELS:
            stats = self.interval[level]
            total = max(1, stats["total"])
            metrics[f"curriculum/{level}_all_wrong_rate"] = stats["all_wrong"] / total
            metrics[f"curriculum/{level}_mixed_rate"] = stats["mixed"] / total
            metrics[f"curriculum/{level}_all_correct_rate"] = stats["all_correct"] / total
            metrics[f"curriculum/{level}_weight"] = self.weights[level]

        should_update = global_step >= self.warmup_steps and global_step % self.update_every == 0
        total_mixed = sum(self.interval[level]["mixed"] for level in LEVELS)
        if should_update and total_mixed >= self.min_mixed_groups:
            mixed_rates = {
                level: self.interval[level]["mixed"] / max(1, self.interval[level]["total"]) for level in LEVELS
            }
            rate_sum = sum(mixed_rates.values())
            if rate_sum > 0:
                target = _bounded_normalize(
                    {level: mixed_rates[level] / rate_sum for level in LEVELS},
                    self.minimum,
                    self.maximum,
                )
                proposed = {}
                for level in LEVELS:
                    smoothed = (1.0 - self.smoothing) * self.weights[level] + self.smoothing * target[level]
                    delta = max(-self.max_change, min(self.max_change, smoothed - self.weights[level]))
                    proposed[level] = self.weights[level] + delta
                self.weights = _bounded_normalize(proposed, self.minimum, self.maximum)
                metrics["curriculum/updated"] = 1.0
            else:
                metrics["curriculum/updated"] = 0.0
        else:
            metrics["curriculum/updated"] = 0.0

        if should_update:
            self.interval = {level: Counter(total=0, all_wrong=0, mixed=0, all_correct=0) for level in LEVELS}
        for level in LEVELS:
            metrics[f"curriculum/{level}_weight"] = self.weights[level]
        self.last_metrics = metrics
        return metrics
