"""Print deterministic real examples for each Countdown curriculum level."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import pandas as pd

from verl.utils.dataset.countdown_curriculum import classify_countdown_difficulty
from verl.utils.reward_score.countdown import evaluate_equation


def solve(numbers: tuple[int, ...], target: int, allow_multiply: bool, allow_divide: bool):
    """Return one exact expression using every number once, or None."""
    count = len(numbers)
    dp: dict[int, dict[Fraction, str]] = {}
    for index, number in enumerate(numbers):
        dp[1 << index] = {Fraction(number): str(number)}

    for subset_size in range(2, count + 1):
        for mask in range(1, 1 << count):
            if bin(mask).count("1") != subset_size:
                continue
            values: dict[Fraction, str] = {}
            left_mask = (mask - 1) & mask
            while left_mask:
                right_mask = mask ^ left_mask
                if right_mask and left_mask < right_mask:
                    for left, left_expr in dp[left_mask].items():
                        for right, right_expr in dp[right_mask].items():
                            candidates = [
                                (left + right, f"({left_expr} + {right_expr})"),
                                (left - right, f"({left_expr} - {right_expr})"),
                                (right - left, f"({right_expr} - {left_expr})"),
                            ]
                            if allow_multiply:
                                candidates.append((left * right, f"({left_expr} * {right_expr})"))
                            if allow_divide:
                                if right != 0:
                                    candidates.append((left / right, f"({left_expr} / {right_expr})"))
                                if left != 0:
                                    candidates.append((right / left, f"({right_expr} / {left_expr})"))
                            for value, expression in candidates:
                                values.setdefault(value, expression)
                left_mask = (left_mask - 1) & mask
            dp[mask] = values

    return dp[(1 << count) - 1].get(Fraction(target))


def subtype(numbers: tuple[int, ...], target: int) -> tuple[str, str]:
    additive = solve(numbers, target, allow_multiply=False, allow_divide=False)
    no_division = solve(numbers, target, allow_multiply=True, allow_divide=False)
    unrestricted = solve(numbers, target, allow_multiply=True, allow_divide=True)
    if unrestricted is None:
        raise AssertionError(f"Dataset row is not solvable: {numbers} -> {target}")

    count = len(numbers)
    if count == 3 and additive is not None:
        return "L0", "3 numbers; an addition/subtraction-only expression exists"
    if count == 3 and no_division is not None:
        return "L1", "3 numbers; multiplication is needed, but division is not"
    if count == 4 and additive is not None:
        return "L1", "4 numbers; an addition/subtraction-only expression exists"
    if count == 3:
        return "L2", "3 numbers; no solution exists without division"
    if no_division is not None:
        return "L2", "4 numbers; multiplication is needed, but division is not"
    return "L3", "4 numbers; no solution exists without division"


def key_for(level: str, reason: str) -> str:
    if level == "L1":
        return "L1-3-multiply" if reason.startswith("3 numbers") else "L1-4-additive"
    if level == "L2":
        return "L2-3-division" if reason.startswith("3 numbers") else "L2-4-multiply"
    return level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--per-subtype", type=int, default=3)
    args = parser.parse_args()

    frame = pd.read_parquet(args.parquet, columns=["target", "nums"])
    wanted = {
        "L0": args.per_subtype,
        "L1-3-multiply": args.per_subtype,
        "L1-4-additive": args.per_subtype,
        "L2-3-division": args.per_subtype,
        "L2-4-multiply": args.per_subtype,
        "L3": args.per_subtype,
    }
    found = {key: [] for key in wanted}

    for index, row in frame.iterrows():
        numbers = tuple(int(number) for number in row["nums"])
        target = int(row["target"])
        level = classify_countdown_difficulty(numbers, target)
        checked_level, reason = subtype(numbers, target)
        assert level == checked_level
        key = key_for(level, reason)
        if len(found[key]) >= wanted[key]:
            continue

        expression = solve(numbers, target, allow_multiply=True, allow_divide=True)
        evaluated = evaluate_equation(expression)
        assert evaluated is not None
        value, used_numbers = evaluated
        assert value == Fraction(target)
        assert sorted(used_numbers) == sorted(numbers)

        found[key].append(
            {
                "index": int(index),
                "target": target,
                "numbers": list(numbers),
                "level": level,
                "reason": reason,
                "expression": expression,
            }
        )
        if all(len(found[name]) >= count for name, count in wanted.items()):
            break

    for name, examples in found.items():
        print(f"\n## {name}")
        for example in examples:
            print(
                f"index={example['index']} "
                f"numbers={example['numbers']} target={example['target']} "
                f"expression={example['expression']} reason={example['reason']}"
            )


if __name__ == "__main__":
    main()
