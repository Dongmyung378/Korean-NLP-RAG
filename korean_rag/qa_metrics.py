"""Dependency-free KorQuAD v1 EM and Korean character F1.

The normalization and character counting follow the official evaluator at
https://korquad.github.io/dataset/evaluate-v1.0.py. Scores are fractions unless
the function name explicitly includes ``percent``. Empty-answer policy is
reported separately: the original v1 evaluator assigns empty/empty F1 zero.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import string


_SPACE_SYMBOLS = frozenset("'\"《》<>〈〉()‘’")
_ASCII_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(value: str) -> str:
    """Apply the official Korean evaluator's punctuation and whitespace rules."""
    if not isinstance(value, str):
        raise TypeError("An answer must be a string.")
    spaced = "".join(" " if char in _SPACE_SYMBOLS else char for char in value)
    unpunctuated = "".join(char for char in spaced.lower() if char not in _ASCII_PUNCTUATION)
    return " ".join(unpunctuated.split())


def exact_match_score(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def character_f1_score(prediction: str, gold: str) -> float:
    """Compute multiset character overlap, ignoring normalized whitespace."""
    predicted = "".join(normalize_answer(prediction).split())
    expected = "".join(normalize_answer(gold).split())
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, gold_answers: Iterable[str]) -> dict[str, float]:
    """Use the best gold annotation separately for EM and character F1."""
    gold = tuple(gold_answers)
    if not gold or any(not isinstance(answer, str) or not answer for answer in gold):
        raise ValueError("Answerable examples require at least one nonempty gold answer.")
    return {
        "exact_match": max(exact_match_score(prediction, answer) for answer in gold),
        "character_f1": max(character_f1_score(prediction, answer) for answer in gold),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a binomial Wilson interval, without pretending dependent QAs are IID."""
    if isinstance(successes, bool) or isinstance(total, bool) or not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("Counts must be integers.")
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("Counts must satisfy 0 <= successes <= total.")
    if total == 0:
        return None
    fraction = successes / total
    denominator = 1 + z * z / total
    centre = (fraction + z * z / (2 * total)) / denominator
    radius = z * ((fraction * (1 - fraction) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]
