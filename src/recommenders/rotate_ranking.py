"""Validation-only selection and fixed-universe ranking for RotatE scores."""

from __future__ import annotations

from collections import Counter
from heapq import nsmallest
from math import isfinite, log2


WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GAIN = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}


def recipe_order(recipe: str) -> tuple[int, str]:
    suffix = recipe.removeprefix("hummus_recipe_")
    return (int(suffix), recipe) if suffix.isdecimal() else (10**20, recipe)


def percentile_scores(values: dict[str, float]) -> dict[str, float]:
    """Within-user, within-relation empirical percentile; ties share a value."""
    if not values:
        return {}
    if any(not isfinite(value) for value in values.values()):
        raise ValueError("Non-finite KGE score")
    counts = Counter(values.values())
    below = 0
    percentiles = {}
    for value in sorted(counts):
        percentiles[value] = (below + (counts[value] + 1) / 2) / len(values)
        below += counts[value]
    return {recipe: percentiles[value] for recipe, value in values.items()}


def rank_candidates(recipe_ids: list[str], excluded_training: set[str],
                    scores: dict[str, dict[str, float]], weight_five: float) -> list[str]:
    """Rank all eligible recipes; unembedded recipes follow scorable recipes."""
    if not 0 <= weight_five <= 1 or len(recipe_ids) != len(set(recipe_ids)):
        raise ValueError("Invalid weight or duplicate candidate")
    eligible = set(recipe_ids) - excluded_training
    four = scores["RATED_4"]
    five = scores["RATED_5"]
    if set(four) != set(five) or not set(four) <= set(recipe_ids):
        raise ValueError("Positive-relation scores must cover the same candidate subset")
    p4 = percentile_scores({r: v for r, v in four.items() if r in eligible})
    p5 = percentile_scores({r: v for r, v in five.items() if r in eligible})
    scorable = sorted(p4, key=lambda r: (-(1 - weight_five) * p4[r] - weight_five * p5[r], recipe_order(r)))
    unscorable = sorted(eligible - set(scorable), key=recipe_order)
    result = scorable + unscorable
    if len(result) != len(eligible):
        raise AssertionError("Candidate universe changed during ranking")
    return result


def ndcg_at_k(ranking: list[str], ratings: dict[str, int], k: int = 10) -> float | None:
    relevant = [GAIN[rating] for rating in ratings.values() if GAIN[rating] > 0]
    if not relevant:
        return None
    dcg = sum(GAIN[ratings.get(recipe, 1)] / log2(position + 2)
              for position, recipe in enumerate(ranking[:k]))
    ideal = sum(gain / log2(position + 2)
                for position, gain in enumerate(sorted(relevant, reverse=True)[:k]))
    return dcg / ideal


def select_weight(validation_users: list[dict], recipe_ids: list[str], k: int = 10) -> dict:
    """Select weight on validation NDCG only; test labels are not an input.

    Each user has keys `training_recipes`, `scores`, and `validation_ratings`.
    Users with no positive validation item are excluded from weight selection.
    Ties favor the weight closest to 0.5, then the smaller weight.
    """
    totals = {weight: 0.0 for weight in WEIGHT_GRID}
    usable_users = 0
    catalog = set(recipe_ids)
    for user in validation_users:
        ratings = user["validation_ratings"]
        if not any(GAIN[rating] > 0 for rating in ratings.values()):
            continue
        eligible = catalog - set(user["training_recipes"])
        four, five = user["scores"]["RATED_4"], user["scores"]["RATED_5"]
        if set(four) != set(five) or not set(four) <= catalog:
            raise ValueError("Positive-relation scores must cover the same candidate subset")
        p4 = percentile_scores({r: v for r, v in four.items() if r in eligible})
        p5 = percentile_scores({r: v for r, v in five.items() if r in eligible})
        unscorable = nsmallest(k, eligible - set(p4), key=recipe_order)
        for weight in WEIGHT_GRID:
            top = nsmallest(k, p4, key=lambda r: (-(1 - weight) * p4[r] - weight * p5[r], recipe_order(r)))
            if len(top) < k:
                top.extend(unscorable[:k - len(top)])
            totals[weight] += ndcg_at_k(top, ratings, k)
        usable_users += 1
    if not usable_users:
        raise ValueError("No user has a positive validation rating")
    means = {str(weight): totals[weight] / usable_users for weight in WEIGHT_GRID}
    selected = max(WEIGHT_GRID, key=lambda w: (means[str(w)], -abs(w - 0.5), -w))
    return {"weight_five": selected, "weight_four": 1 - selected,
            "validation_mean_ndcg_at_k": means, "users_with_positive_validation": usable_users,
            "k": k}
