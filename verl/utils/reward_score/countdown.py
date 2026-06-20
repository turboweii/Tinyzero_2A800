"""Programmatically verifiable reward for the Countdown task."""

from __future__ import annotations

import ast
import math
import random
import re
from collections import Counter
from fractions import Fraction


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def compute_overlong_penalty(
    response_length: int,
    max_response_length: int,
    buffer_length: int,
    penalty_factor: float,
) -> float:
    expected_length = max_response_length - buffer_length
    exceed_length = max(0, int(response_length) - expected_length)
    return min(
        -exceed_length / max(1, int(buffer_length)) * float(penalty_factor),
        0.0,
    )


def _assistant_text(solution_str: str) -> str | None:
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    if "<|im_start|>assistant" in solution_str:
        return solution_str.split("<|im_start|>assistant", 1)[1]
    return None


def extract_solution(solution_str: str) -> str | None:
    """Extract the last complete answer expression from the assistant response."""
    assistant = _assistant_text(solution_str)
    if assistant is None:
        return None
    matches = list(ANSWER_PATTERN.finditer(assistant))
    return matches[-1].group(1).strip() if matches else None


def _eval_node(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        if node.value < 0:
            raise ValueError("Input literals must be non-negative")
        return Fraction(node.value), [node.value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value, numbers = _eval_node(node.operand)
        return (value if isinstance(node.op, ast.UAdd) else -value), numbers
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, left_numbers = _eval_node(node.left)
        right, right_numbers = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        else:
            if right == 0:
                raise ZeroDivisionError
            value = left / right
        return value, left_numbers + right_numbers
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def evaluate_equation(equation_str: str) -> tuple[Fraction, list[int]] | None:
    """Parse an arithmetic expression and evaluate it exactly with Fraction."""
    try:
        tree = ast.parse(equation_str, mode="eval")
        return _eval_node(tree)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def validate_equation(equation_str: str, available_numbers) -> bool:
    evaluated = evaluate_equation(equation_str)
    if evaluated is None:
        return False
    _, used_numbers = evaluated
    return Counter(used_numbers) == Counter(int(number) for number in available_numbers)


def compute_score(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.1,
    score=1.0,
    answer_bonus=0.05,
    syntax_bonus=0.05,
    number_usage_bonus=0.10,
    proximity_max=0.10,
    proximity_scale=10.0,
    return_details=False,
):
    """Compute dense but strictly gated Countdown reward.

    Incorrect answers receive:
      0.05 for a complete answer tag,
      +0.05 for a valid arithmetic AST,
      +0.10 for using every input exactly once,
      +up to 0.10 for numerical proximity (only after number validation).
    A fully correct expression receives exactly 1.0.
    """
    del method, format_score  # retained in the signature for backwards compatibility
    target = int(ground_truth["target"])
    numbers = [int(number) for number in ground_truth["numbers"]]
    equation = extract_solution(solution_str)

    details = {
        "answer_exists": equation is not None,
        "syntax_valid": False,
        "numbers_valid": False,
        "exact_correct": False,
        "proximity_reward": 0.0,
        "task_score": 0.0,
    }

    if equation is not None:
        details["task_score"] = float(answer_bonus)
        evaluated = evaluate_equation(equation)
        if evaluated is not None:
            result, used_numbers = evaluated
            details["syntax_valid"] = True
            details["task_score"] += float(syntax_bonus)
            if Counter(used_numbers) == Counter(numbers):
                details["numbers_valid"] = True
                details["task_score"] += float(number_usage_bonus)
                if result == Fraction(target):
                    details["exact_correct"] = True
                    details["task_score"] = float(score)
                else:
                    distance = abs(float(result - Fraction(target)))
                    details["proximity_reward"] = float(proximity_max) * math.exp(
                        -distance / float(proximity_scale)
                    )
                    details["task_score"] += details["proximity_reward"]

    if random.randint(1, 256) == 1:
        print(
            "[countdown reward]",
            {
                "target": target,
                "numbers": numbers,
                "equation": equation,
                **details,
            },
        )

    return details if return_details else details["task_score"]
