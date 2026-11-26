"""Independent cohort-scale checks for fitted and validation-selected item CF."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf import GRID, OUTPUT, WEIGHTS


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def gain(rating: int) -> int:
    return 2 if rating == 5 else 1 if rating == 4 else 0


def verify(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    fit = json.loads((output / "fit_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "validation_selection.json").read_text(encoding="utf-8"))
    assert fit["status"] == "fitted_training_only" and selection["status"] == "validation_selected_no_test_access"
    assert fit["training_users"] == 4000 and fit["training_ratings"] == 112000
    assert fit["catalog_recipes"] == selection["candidate_catalog"] == 75972
    assert fit["contract_lock_sha256"] == selection["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json")
    assert selection["fit_manifest_sha256"] == sha(output / "fit_manifest.json")
    for key, filename in (("similarity_sha256", "positive_item_cosine.csv"),
                          ("popularity_sha256", "training_positive_popularity.csv")):
        assert fit[key] == sha(output / filename)
    for key, filename in (("validation_per_user_sha256", "validation_per_user.csv"),
                          ("validation_grid_recommendations_sha256", "validation_recommendations_by_grid.csv"),
                          ("validation_selected_recommendations_sha256", "validation_recommendations_selected.csv")):
        assert selection[key] == sha(output / filename)
    assert fit["reviews_sha256"] == sha(data / "reviews.csv") and fit["recipes_sha256"] == sha(data / "recipes.csv")

    train = defaultdict(dict)
    validation = defaultdict(dict)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train":
            train[row["user_id"]][row["recipe_id"]] = int(row["rating"])
        elif row["split"] == "validation":
            validation[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    assert len(train) == len(validation) == 4000
    assert all(len(history) == 28 for history in train.values())
    assert all(len(ratings) == 4 for ratings in validation.values())

    item_vectors = defaultdict(dict)
    for user, history in train.items():
        for recipe, rating in history.items():
            if WEIGHTS[rating] != 0:
                item_vectors[recipe][user] = WEIGHTS[rating]
    sample_count = 0
    cosine_count = 0
    for row in rows(output / "positive_item_cosine.csv"):
        a, b, actual = row["item_a"], row["item_b"], float(row["similarity"])
        assert 0 < actual <= 1 and math.isfinite(actual)
        if sample_count < 100:
            left, right = item_vectors[a], item_vectors[b]
            dot = sum(weight * right.get(user, 0) for user, weight in left.items())
            norm_a = math.sqrt(sum(weight * weight for weight in left.values()))
            norm_b = math.sqrt(sum(weight * weight for weight in right.values()))
            expected = dot / (norm_a * norm_b)
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            sample_count += 1
        cosine_count += 1
    assert cosine_count == fit["positive_similarity_pairs"] and sample_count == 100

    lists = defaultdict(list)
    source_counts = defaultdict(int)
    top_neighbor_counts = Counter()
    top_score_counts = Counter()
    for row in rows(output / "validation_recommendations_by_grid.csv"):
        user, n, rank, recipe = row["user_id"], int(row["neighbor_n"]), int(row["rank"]), row["recipe_id"]
        assert n in GRID and recipe not in train[user] and row["score_source"] in {"score", "fallback"}
        if row["score_source"] == "score":
            assert math.isfinite(float(row["score"])) and int(row["contributing_neighbors"]) in range(1, n + 1)
            top_neighbor_counts[int(row["contributing_neighbors"])] += 1
            top_score_counts[row["score"]] += 1
        else:
            assert row["score"] == "" and row["contributing_neighbors"] == "0"
        lists[(user, n)].append((rank, recipe, row["score"], row["score_source"]))
        source_counts[row["score_source"]] += 1
    assert len(lists) == 4000 * len(GRID)
    assert all([rank for rank, *_ in value] == list(range(1, 11)) and
               len({recipe for _, recipe, *_ in value}) == 10 for value in lists.values())
    selected_n = selection["selected_n"]
    selected = list(rows(output / "validation_recommendations_selected.csv"))
    assert len(selected) == 40000
    assert all(int(row["neighbor_n"]) == selected_n and
               (int(row["rank"]), row["recipe_id"], row["score"], row["score_source"]) ==
               lists[(row["user_id"], selected_n)][int(row["rank"]) - 1]
               for row in selected)

    per_user = {row["user_id"]: row for row in rows(output / "validation_per_user.csv")}
    assert len(per_user) == 4000
    sums = {n: 0.0 for n in GRID}
    eligible = 0
    distinct_top_10 = 0
    scorable_counts = []
    users_with_positive_hit = 0
    for user in sorted(train):
        ratings = validation[user]
        scorable = int(per_user[user]["scorable_candidates"])
        assert 0 < scorable <= 75972 - 28
        scorable_counts.append(scorable)
        positives = sorted((gain(rating) for rating in ratings.values() if gain(rating) > 0), reverse=True)
        assert int(per_user[user]["positive_validation_targets"]) == len(positives)
        if not positives:
            assert all(per_user[user][f"ndcg_{n}"] == "" for n in GRID)
            continue
        eligible += 1
        ideal = sum(value / math.log2(rank + 2) for rank, value in enumerate(positives))
        for n in GRID:
            actual = sum(gain(ratings.get(recipe, 1)) / math.log2(rank + 2)
                         for rank, (_, recipe, _, _) in enumerate(lists[(user, n)])) / ideal
            assert math.isclose(actual, float(per_user[user][f"ndcg_{n}"]), abs_tol=1e-12)
            sums[n] += actual
            if n == selected_n and actual > 0:
                users_with_positive_hit += 1
        if len({tuple(recipe for _, recipe, _, _ in lists[(user, n)]) for n in GRID}) > 1:
            distinct_top_10 += 1
    assert eligible == selection["users_with_positive_validation"]
    means = {n: sums[n] / eligible for n in GRID}
    assert all(math.isclose(means[n], selection["mean_validation_ndcg_at_10"][str(n)], abs_tol=1e-12)
               for n in GRID)
    assert selected_n == max(GRID, key=lambda n: (means[n], -n))
    return {"condition": "1_item_cf", "status": "fit_and_validation_verified",
            "selected_n": selected_n, "mean_ndcg_at_10": means,
            "users_with_positive_validation": eligible,
            "users_with_grid_dependent_top_10": distinct_top_10,
            "recommendations_by_source": dict(source_counts),
            "top_10_neighbor_counts": dict(sorted(top_neighbor_counts.items())),
            "top_10_score_one_count": top_score_counts["1"],
            "scorable_candidates_min": min(scorable_counts),
            "scorable_candidates_max": max(scorable_counts),
            "scorable_candidates_mean": sum(scorable_counts) / len(scorable_counts),
            "users_with_positive_hit_at_10": users_with_positive_hit,
            "independently_recomputed_cosines": sample_count,
            "verified_similarity_pairs": cosine_count}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
