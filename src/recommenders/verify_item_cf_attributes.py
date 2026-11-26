"""Independent saved-artifact checks for RQ1 condition 2 validation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf_attributes import BASE_OUTPUT, GRID, OUTPUT, load_attributes
from src.recommenders.item_cf import WEIGHTS
from src.recommenders.rotate_ranking import recipe_order


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def gain(rating: int) -> int:
    return 2 if rating == 5 else 1 if rating == 4 else 0


def verify(data: Path = DATA, output: Path = OUTPUT, base: Path = BASE_OUTPUT) -> dict:
    data, output, base = Path(data).resolve(), Path(output).resolve(), Path(base).resolve()
    verify_lock(data)
    fit, recipes, ingredient_matrix, concept_matrix = load_attributes(data, output, base)
    selection = json.loads((output / "validation_selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "validation_selected_no_test_access"
    assert selection["fit_manifest_sha256"] == sha(output / "fit_manifest.json")
    assert selection["base_validation_selection_sha256"] == sha(base / "validation_selection.json")
    assert selection["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json")
    for key, filename in (("validation_per_user_sha256", "validation_per_user.csv"),
                          ("validation_grid_recommendations_sha256", "validation_recommendations_by_grid.csv"),
                          ("validation_selected_recommendations_sha256", "validation_recommendations_selected.csv")):
        assert selection[key] == sha(output / filename)
    assert fit["recipes_with_mapped_ingredient"] == 65984
    assert fit["recipes_with_direct_foodon"] == 65943
    assert ingredient_matrix.nnz == fit["ingredient_incidence_count"] == 586292
    assert concept_matrix.nnz == fit["direct_foodon_incidence_count"] == 506069
    sample_recipes = set(recipes[:20])
    raw_ingredients = defaultdict(set)
    for row in rows(data / "recipe_ingredients.csv"):
        if row["recipe_id"] in sample_recipes:
            raw_ingredients[row["recipe_id"]].add(row["ingredient_id"])
    direct_links = defaultdict(set)
    for row in rows(data / "foodon_links.csv"):
        direct_links[row["ingredient_id"]].add(row["foodon_iri"])
    raw_concepts = {recipe: set().union(*(direct_links[ingredient] for ingredient in values))
                    for recipe, values in raw_ingredients.items()}
    for matrix, raw in ((ingredient_matrix, raw_ingredients),
                        (concept_matrix, raw_concepts)):
        for left in range(0, 20, 2):
            right = left + 1
            a, b = raw.get(recipes[left], set()), raw.get(recipes[right], set())
            expected = len(a & b) / len(a | b) if a and b else 0.0
            intersection = int(matrix[left].multiply(matrix[right]).sum())
            union = matrix[left].nnz + matrix[right].nnz - intersection
            actual = intersection / union if union else 0.0
            assert math.isclose(actual, expected, abs_tol=1e-12)

    train, validation = defaultdict(set), defaultdict(dict)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train":
            train[row["user_id"]].add(row["recipe_id"])
        elif row["split"] == "validation":
            validation[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    assert len(train) == len(validation) == 4000
    assert all(len(history) == 28 for history in train.values())
    assert all(len(targets) == 4 for targets in validation.values())

    baseline = {(row["user_id"], int(row["rank"])): row["recipe_id"]
                for row in rows(base / "validation_recommendations_selected.csv")}
    assert len(baseline) == 40000
    grid_rows = defaultdict(list)
    source_counts = Counter()
    top_score_one = Counter()
    for row in rows(output / "validation_recommendations_by_grid.csv"):
        user, alpha, rank = row["user_id"], float(row["alpha"]), int(row["rank"])
        assert alpha in GRID and int(row["neighbor_n"]) == fit["selected_baseline_n"] == 5
        assert row["recipe_id"] not in train[user]
        assert row["score_source"] in {"score", "fallback"}
        if row["score_source"] == "score":
            assert math.isfinite(float(row["score"]))
            assert 1 <= int(row["contributing_neighbors"]) <= 5
            top_score_one[alpha] += float(row["score"]) == 1
        else:
            assert row["score"] == "" and row["contributing_neighbors"] == "0"
        if alpha == 0:
            assert row["recipe_id"] == baseline[user, rank]
        source_counts[(alpha, row["score_source"])] += 1
        grid_rows[(user, alpha)].append(row)
    assert len(grid_rows) == 4000 * len(GRID)
    assert all(len(value) == 10 and [int(row["rank"]) for row in value] == list(range(1, 11)) and
               len({row["recipe_id"] for row in value}) == 10 for value in grid_rows.values())
    sample_users = sorted(train)[:5]
    sample_history = defaultdict(dict)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train" and row["user_id"] in sample_users:
            sample_history[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    sample_candidates = {row["recipe_id"] for user in sample_users for alpha in GRID
                         for row in grid_rows[(user, alpha)]}
    sample_recipes = sample_candidates | {recipe for history in sample_history.values() for recipe in history}
    sample_ingredients = defaultdict(set)
    for row in rows(data / "recipe_ingredients.csv"):
        if row["recipe_id"] in sample_recipes:
            sample_ingredients[row["recipe_id"]].add(row["ingredient_id"])
    sample_concepts = {recipe: set().union(*(direct_links[ingredient] for ingredient in values))
                       for recipe, values in sample_ingredients.items()}
    needed_pairs = {tuple(sorted((history_recipe, candidate), key=recipe_order))
                    for user in sample_users for history_recipe in sample_history[user]
                    for alpha in GRID for candidate in
                    (row["recipe_id"] for row in grid_rows[(user, alpha)])}
    sample_cf = {}
    for row in rows(base / "positive_item_cosine.csv"):
        pair = (row["item_a"], row["item_b"])
        if pair in needed_pairs:
            sample_cf[pair] = float(row["similarity"])

    def jaccard(left, right):
        return len(left & right) / len(left | right) if left and right else 0.0

    checked_scores = 0
    for user in sample_users:
        for alpha in GRID:
            for row in grid_rows[(user, alpha)]:
                candidate = row["recipe_id"]
                neighbors = []
                for history_recipe, rating in sample_history[user].items():
                    if WEIGHTS[rating] == 0:
                        continue
                    pair = tuple(sorted((history_recipe, candidate), key=recipe_order))
                    attribute = 0.5 * (jaccard(sample_ingredients[history_recipe],
                                                sample_ingredients[candidate]) +
                                       jaccard(sample_concepts.get(history_recipe, set()),
                                               sample_concepts.get(candidate, set())))
                    similarity = (1 - alpha) * sample_cf.get(pair, 0.0) + alpha * attribute
                    if similarity > 0:
                        neighbors.append((similarity, history_recipe, WEIGHTS[rating]))
                neighbors.sort(key=lambda entry: (-entry[0], recipe_order(entry[1])))
                chosen = neighbors[:5]
                denominator = sum(similarity * abs(weight) for similarity, _, weight in chosen)
                expected_score = sum(similarity * weight for similarity, _, weight in chosen) / denominator
                assert row["score_source"] == "score" and int(row["contributing_neighbors"]) == len(chosen)
                assert math.isclose(float(row["score"]), expected_score, abs_tol=1e-12)
                checked_scores += 1

    selected_alpha = selection["selected_alpha"]
    selected_rows = list(rows(output / "validation_recommendations_selected.csv"))
    assert len(selected_rows) == 40000
    assert all(float(row["alpha"]) == selected_alpha and
               row == grid_rows[(row["user_id"], selected_alpha)][int(row["rank"]) - 1]
               for row in selected_rows)

    per_user = {row["user_id"]: row for row in rows(output / "validation_per_user.csv")}
    assert len(per_user) == 4000
    totals = {alpha: 0.0 for alpha in GRID}
    eligible = 0
    hit_users = Counter()
    scorable_ranges = {alpha: [] for alpha in GRID}
    for user in sorted(train):
        positive = sorted((gain(r) for r in validation[user].values() if gain(r) > 0), reverse=True)
        assert int(per_user[user]["positive_validation_targets"]) == len(positive)
        for alpha in GRID:
            key = str(alpha).removesuffix(".0")
            scorable = int(per_user[user][f"scorable_{key}"])
            assert 0 <= scorable <= 75944
            scorable_ranges[alpha].append(scorable)
            metric = per_user[user][f"ndcg_{key}"]
            if not positive:
                assert metric == ""
                continue
            ideal = sum(value / math.log2(position + 2) for position, value in enumerate(positive))
            actual = sum(gain(validation[user].get(row["recipe_id"], 1)) / math.log2(position + 2)
                         for position, row in enumerate(grid_rows[(user, alpha)])) / ideal
            assert math.isclose(actual, float(metric), abs_tol=1e-12)
            totals[alpha] += actual
            hit_users[alpha] += actual > 0
        eligible += bool(positive)
    assert eligible == selection["users_with_positive_validation"] == 3995
    means = {alpha: totals[alpha] / eligible for alpha in GRID}
    assert all(math.isclose(means[alpha], selection["mean_validation_ndcg_at_10"][str(alpha)], abs_tol=1e-12)
               for alpha in GRID)
    assert selected_alpha == max(GRID, key=lambda alpha: (means[alpha], -alpha))
    assert math.isclose(means[0], json.loads((base / "validation_selection.json").read_text(
        encoding="utf-8"))["mean_validation_ndcg_at_10"]["5"], abs_tol=1e-12)
    changed_from_baseline = {str(alpha): sum(
        tuple(row["recipe_id"] for row in grid_rows[(user, alpha)]) !=
        tuple(row["recipe_id"] for row in grid_rows[(user, 0.0)]) for user in train)
        for alpha in GRID if alpha != 0.0}
    return {"status": "fit_and_validation_verified", "selected_alpha": selected_alpha,
            "baseline_n": fit["selected_baseline_n"], "mean_ndcg_at_10": means,
            "users_with_positive_validation": eligible, "positive_hit_users": dict(hit_users),
            "independently_recomputed_top_scores": checked_scores,
            "top_10_score_one_count": dict(top_score_one),
            "users_with_changed_top_10_vs_alpha_zero": changed_from_baseline,
            "top_10_score_source_counts": {str(key): value for key, value in source_counts.items()},
            "scorable_candidate_range": {str(alpha): [min(values), max(values)]
                                         for alpha, values in scorable_ranges.items()}}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
