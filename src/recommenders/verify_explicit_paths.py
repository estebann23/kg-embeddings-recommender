"""Independent checks of condition-3 fitted paths and validation selection."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.explicit_paths import BASE_OUTPUT, GRID, OUTPUT, load_fit_paths


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def gain(rating: int) -> int:
    return 2 if rating == 5 else 1 if rating == 4 else 0


def verify(data: Path = DATA, output: Path = OUTPUT, base: Path = BASE_OUTPUT) -> dict:
    data, output, base = Path(data).resolve(), Path(output).resolve(), Path(base).resolve()
    verify_lock(data)
    fit, recipes, families = load_fit_paths(data, output, base)
    selection = json.loads((output / "validation_selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "validation_selected_no_test_access"
    assert selection["fit_manifest_sha256"] == sha(output / "fit_manifest.json")
    assert selection["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json")
    for key, filename in (("validation_per_user_sha256", "validation_per_user.csv"),
                          ("validation_grid_recommendations_sha256", "validation_recommendations_by_grid.csv"),
                          ("validation_selected_recommendations_sha256", "validation_recommendations_selected.csv")):
        assert selection[key] == sha(output / filename)
    assert fit["mapped_recipes_for_parent_threshold"] == 65984
    assert len(fit["broad_parent_ids_removed"]) == 44
    assert fit["source_asserted_parent_edges"] == 2872
    assert fit["feature_families"]["ingredient"]["incidences"] == 586292
    assert fit["feature_families"]["direct_foodon"]["incidences"] == 506069
    assert fit["feature_families"]["parent_foodon"]["incidences"] == 197539

    train, validation = defaultdict(dict), defaultdict(dict)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train":
            train[row["user_id"]][row["recipe_id"]] = int(row["rating"])
        elif row["split"] == "validation":
            validation[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    assert len(train) == len(validation) == 4000
    assert all(len(history) == 28 for history in train.values())
    assert all(len(targets) == 4 for targets in validation.values())

    grid_rows = defaultdict(list)
    sources = Counter()
    for row in rows(output / "validation_recommendations_by_grid.csv"):
        user, beta = row["user_id"], float(row["beta"])
        assert beta in GRID and row["recipe_id"] not in train[user]
        assert row["score_source"] in {"score", "fallback"}
        if row["score_source"] == "score":
            assert 0 < float(row["score"]) <= 1 + 1e-12
        else:
            assert row["score"] == ""
        for column in ("matched_ingredient_ids", "matched_direct_foodon_ids",
                       "matched_parent_foodon_ids"):
            assert isinstance(json.loads(row[column]), list)
        if beta == 0:
            assert json.loads(row["matched_parent_foodon_ids"]) == []
        sources[(beta, row["score_source"])] += 1
        grid_rows[(user, beta)].append(row)
    assert len(grid_rows) == 4000 * len(GRID)
    assert all(len(value) == 10 and [int(row["rank"]) for row in value] == list(range(1, 11)) and
               len({row["recipe_id"] for row in value}) == 10 for value in grid_rows.values())

    selected_beta = selection["selected_beta"]
    selected_rows = list(rows(output / "validation_recommendations_selected.csv"))
    assert len(selected_rows) == 40000
    assert all(float(row["beta"]) == selected_beta and
               row == grid_rows[(row["user_id"], selected_beta)][int(row["rank"]) - 1]
               for row in selected_rows)
    per_user = {row["user_id"]: row for row in rows(output / "validation_per_user.csv")}
    assert len(per_user) == 4000
    totals = {beta: 0.0 for beta in GRID}
    eligible = 0
    hit_users = Counter()
    scorable_ranges = {beta: [] for beta in GRID}
    for user in sorted(train):
        positive = sorted((gain(r) for r in validation[user].values() if gain(r) > 0), reverse=True)
        assert int(per_user[user]["positive_validation_targets"]) == len(positive)
        assert int(per_user[user]["positive_training_history"]) == sum(r >= 4 for r in train[user].values())
        for beta in GRID:
            key = str(beta).removesuffix(".0")
            scorable = int(per_user[user][f"scorable_{key}"])
            assert 0 <= scorable <= 75944
            scorable_ranges[beta].append(scorable)
            metric = per_user[user][f"ndcg_{key}"]
            if not positive:
                assert metric == ""
                continue
            ideal = sum(value / math.log2(position + 2) for position, value in enumerate(positive))
            actual = sum(gain(validation[user].get(row["recipe_id"], 1)) / math.log2(position + 2)
                         for position, row in enumerate(grid_rows[(user, beta)])) / ideal
            assert math.isclose(actual, float(metric), abs_tol=1e-12)
            totals[beta] += actual
            hit_users[beta] += actual > 0
        eligible += bool(positive)
    assert eligible == selection["users_with_positive_validation"] == 3995
    means = {beta: totals[beta] / eligible for beta in GRID}
    assert all(math.isclose(means[beta], selection["mean_validation_ndcg_at_10"][str(beta)], abs_tol=1e-12)
               for beta in GRID)
    assert selected_beta == max(GRID, key=lambda beta: (means[beta], -beta))

    # Recompute sampled scores and traces with set operations, outside the sparse scorer.
    recipe_index = {recipe: index for index, recipe in enumerate(recipes)}
    set_features = {}
    weights = {}
    for family, (binary, _, _, ids) in families.items():
        set_features[family] = [set(binary[index].indices) for index in range(len(recipes))]
        df = [int(value) for value in binary.getnnz(axis=0)]
        weights[family] = [math.log((1 + len(recipes)) / (1 + count)) for count in df]

    def weighted_jaccard(family, left, right):
        a, b = set_features[family][left], set_features[family][right]
        intersection, union = a & b, a | b
        denominator = sum(weights[family][feature] for feature in union)
        return sum(weights[family][feature] for feature in intersection) / denominator if denominator else 0.0

    score_checks = 0
    trace_checks = 0
    for user in sorted(train)[:5]:
        history = [(recipe_index[recipe], gain(rating)) for recipe, rating in train[user].items()
                   if rating >= 4]
        denominator = sum(weight for _, weight in history)
        for beta in GRID:
            for row in grid_rows[(user, beta)]:
                index = recipe_index[row["recipe_id"]]
                score = sum(weight * (0.5 * weighted_jaccard("ingredient", hist, index) +
                                      0.5 * ((1 - beta) * weighted_jaccard("direct_foodon", hist, index) +
                                             beta * weighted_jaccard("parent_foodon", hist, index)))
                            for hist, weight in history) / denominator if denominator else 0.0
                if score > 0:
                    assert row["score_source"] == "score"
                    assert math.isclose(float(row["score"]), score, abs_tol=1e-12)
                    score_checks += 1
                else:
                    assert row["score_source"] == "fallback"
                for family, column in (("ingredient", "matched_ingredient_ids"),
                                       ("direct_foodon", "matched_direct_foodon_ids"),
                                       ("parent_foodon", "matched_parent_foodon_ids")):
                    expected = set()
                    if family != "parent_foodon" or beta > 0:
                        for hist, _ in history:
                            expected.update(set_features[family][index] & set_features[family][hist])
                    ids = families[family][3]
                    assert json.loads(row[column]) == [ids[col] for col in sorted(expected)]
                    trace_checks += 1
    return {"status": "fit_and_validation_verified", "selected_beta": selected_beta,
            "mean_ndcg_at_10": means, "users_with_positive_validation": eligible,
            "positive_hit_users": dict(hit_users),
            "top_10_score_source_counts": {str(key): value for key, value in sources.items()},
            "scorable_candidate_range": {str(beta): [min(values), max(values)]
                                         for beta, values in scorable_ranges.items()},
            "independently_recomputed_scores": score_checks,
            "independently_checked_trace_columns": trace_checks}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
