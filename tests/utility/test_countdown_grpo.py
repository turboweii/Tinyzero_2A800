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

import math

import pandas as pd
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.dataset.countdown_curriculum import (
    AdaptiveCurriculumBatchSampler,
    classify_countdown_difficulty,
    quota_from_weights,
)
from verl.utils.reward_score.countdown import compute_overlong_penalty, compute_score


def _response(answer):
    return f"A conversation\nAssistant: <think>work</think>\n<answer>{answer}</answer>"


def test_countdown_dense_reward_is_strictly_gated():
    ground_truth = {"target": 6, "numbers": [1, 2, 3]}

    assert compute_score("Assistant: no answer", ground_truth) == 0.0
    assert compute_score(_response("hello"), ground_truth) == 0.05
    assert compute_score(_response("1 + 2"), ground_truth) == 0.10

    wrong = compute_score(_response("1 + 2 - 3"), ground_truth)
    expected = 0.05 + 0.05 + 0.10 + 0.10 * math.exp(-6 / 10)
    assert math.isclose(wrong, expected, rel_tol=1e-7)
    assert compute_score(_response("(1 + 2) + 3"), ground_truth) == 1.0


def test_structural_difficulty_uses_operation_requirements():
    assert classify_countdown_difficulty((1, 2, 3), 6) == "L0"
    assert classify_countdown_difficulty((2, 3, 4), 10) == "L1"
    assert classify_countdown_difficulty((2, 5, 6), 8) == "L2"
    assert classify_countdown_difficulty((1, 2, 3, 4), 10) == "L1"
    assert classify_countdown_difficulty((1, 2, 3, 4), 13) == "L2"
    assert classify_countdown_difficulty((7, 48, 75, 24), 15) == "L3"


def test_curriculum_quota_sums_to_batch_size():
    quota = quota_from_weights({"L0": 0.45, "L1": 0.30, "L2": 0.20, "L3": 0.05}, 256)
    assert quota == {"L0": 115, "L1": 77, "L2": 51, "L3": 13}
    assert sum(quota.values()) == 256


def test_overlong_penalty_is_linear_only_in_the_buffer():
    assert compute_overlong_penalty(768, 1024, 256, 0.1) == 0.0
    assert compute_overlong_penalty(896, 1024, 256, 0.1) == -0.05
    assert compute_overlong_penalty(1024, 1024, 256, 0.1) == -0.1


def test_clip_higher_uses_asymmetric_upper_bound():
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.full((1, 1), math.log(1.6))
    advantages = torch.ones(1, 1)
    mask = torch.ones(1, 1)

    loss, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.4,
    )
    assert torch.allclose(loss, torch.tensor(-1.4))


def test_adaptive_curriculum_moves_toward_mixed_groups():

    class Dataset:
        dataframe = pd.DataFrame({"difficulty": ["L0"] * 100 + ["L1"] * 100 + ["L2"] * 100 + ["L3"] * 100})

        def __len__(self):
            return len(self.dataframe)

    config = {
        "initial_weights": {
            "L0": 0.45,
            "L1": 0.30,
            "L2": 0.20,
            "L3": 0.05
        },
        "warmup_steps": 40,
        "update_every": 20,
        "min_mixed_groups": 4,
        "min_weight": 0.05,
        "max_weight": 0.60,
        "smoothing": 0.20,
        "max_change_per_update": 0.05,
    }
    sampler = AdaptiveCurriculumBatchSampler(Dataset(), 16, config)
    for level in ("L0", "L1", "L3"):
        for _ in range(20):
            sampler.record_group(level, correct_count=0, group_size=4)
    for _ in range(20):
        sampler.record_group("L2", correct_count=2, group_size=4)

    old_weight = sampler.weights["L2"]
    sampler.maybe_update(40)
    assert sampler.weights["L2"] > old_weight
