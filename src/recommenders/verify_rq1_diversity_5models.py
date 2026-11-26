"""Independent full-row audit of the five-model RQ1 diversity output."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from src.recommenders.contract import DATA, sha, verify_lock
from src.recommenders.item_cf import load_recipes
from src.recommenders.rq1_diversity import load_ingredients
from src.recommenders.rq1_diversity_5models import (CONFIG, K_VALUES, OUTPUT,
                                                   SPLIT, read_json, rows)


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (actual, expected)


def cumulative_gini(counts: list[int]) -> float:
    ordered = sorted(counts)
    running = cumulative = 0
    for count in ordered:
        running += count
        cumulative += running
    assert running > 0
    return (len(ordered) + 1 - 2 * cumulative / running) / len(ordered)


def verify(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    pre = read_json(output / "preflight.json")
    manifest = read_json(output / "evaluation_manifest.json")
    config = read_json(CONFIG)
    assert pre["status"] == "five_model_diversity_preflight_passed"
    assert manifest["status"] == "five_model_diversity_complete"
    assert pre["config_sha256"] == sha(CONFIG)
    assert pre["split_sha256"] == sha(SPLIT)
    assert pre["recipes_sha256"] == sha(data / "recipes.csv")
    assert pre["ingredient_source_sha256"] == sha(data / "recipe_ingredients.csv")
    assert manifest["preflight_sha256"] == sha(output / "preflight.json")
    assert manifest["per_user_sha256"] == sha(output / "per_user_diversity.csv")
    assert manifest["aggregate_sha256"] == sha(output / "aggregate_diversity.csv")
    paths = {model: Path(CONFIG.parent.parent / name) for model, name in config["models"].items()}
    assert len(paths) == 5
    for model, path in paths.items():
        assert pre["ranking_sha256"][model] == sha(path)

    half_users = defaultdict(Counter)
    for row in rows(SPLIT):
        half_users[row["test_half"]][row["user_id"]] += 1
    assert set(half_users) == {"random_50", "semantic_50"}
    assert half_users["random_50"] == half_users["semantic_50"]
    assert len(half_users["random_50"]) == 4000
    assert set(half_users["random_50"].values()) == {4}
    expected_users = set(half_users["random_50"])

    catalog = load_recipes(data)
    catalog_set = set(catalog)
    ingredient_sets = load_ingredients(data, catalog)
    saved = {(row["model"], row["user_id"], int(row["k"])): row
             for row in rows(output / "per_user_diversity.csv")}
    assert len(saved) == 5 * 4000 * 2 == manifest["per_user_rows"]
    aggregate = {(row["model"], int(row["k"])): row
                 for row in rows(output / "aggregate_diversity.csv")}
    assert len(aggregate) == 10 == manifest["aggregate_rows"]
    observed = set()
    checked_pairs = 0
    for model, path in paths.items():
        lists = defaultdict(list)
        ranks = defaultdict(list)
        for row in rows(path):
            user = row["user_id"]
            lists[user].append(row["recipe_id"])
            ranks[user].append(int(row["rank"]))
        assert set(lists) == expected_users
        exposures = {k: Counter() for k in K_VALUES}
        ilda = {k: [] for k in K_VALUES}
        pairs = {k: 0 for k in K_VALUES}
        positions = {k: 0 for k in K_VALUES}
        defined = {k: 0 for k in K_VALUES}
        for user, ranking in lists.items():
            assert ranks[user] == list(range(1, 51))
            assert len(set(ranking)) == 50 and set(ranking) <= catalog_set
            for k in K_VALUES:
                key = (model, user, k)
                assert key in saved and key not in observed
                observed.add(key)
                top = ranking[:k]
                exposures[k].update(top)
                mapped = [ingredient_sets[item] for item in top if item in ingredient_sets]
                position = len(mapped) / k
                pair_count = len(mapped) * (len(mapped) - 1) // 2
                pair_ratio = pair_count / (k * (k - 1) // 2)
                distances = (len(a.symmetric_difference(b)) / len(a.union(b))
                             for a, b in combinations(mapped, 2))
                distance = math.fsum(distances) / pair_count if pair_count else None
                checked_pairs += pair_count
                row = saved[key]
                close(float(row["pair_coverage"]), pair_ratio)
                close(float(row["ingredient_position_coverage"]), position)
                if distance is None:
                    assert row["ingredient_ild"] == ""
                else:
                    close(float(row["ingredient_ild"]), distance)
                    ilda[k].append(distance)
                    defined[k] += 1
                pairs[k] += pair_count
                positions[k] += len(mapped)
        for k in K_VALUES:
            row = aggregate[model, k]
            counts = [exposures[k][item] for item in catalog]
            assert sum(counts) == 4000 * k
            expected = {
                "ingredient_ild": math.fsum(ilda[k]) / len(ilda[k]),
                "exposure_gini": cumulative_gini(counts),
                "catalog_coverage": sum(count > 0 for count in counts) / len(catalog),
                "pair_coverage": pairs[k] / (4000 * k * (k - 1) // 2),
                "ingredient_position_coverage": positions[k] / (4000 * k),
                "defined_user_coverage": defined[k] / 4000,
                "user_list_coverage": len(lists) / 4000,
            }
            for name, value in expected.items():
                close(float(row[name]), value)
                assert 0 <= float(row[name]) <= 1

    assert observed == set(saved)
    # Four rows per K must reproduce the prior diversity experiment on unchanged inputs.
    previous = {(row["model"], int(row["k"])): row for row in
                rows(CONFIG.parent.parent / "results/rq1_diversity_test/aggregate_diversity_exposure.csv")}
    older_names = {"interaction_only_cf": "1_item_cf", "weighted_graph_paths": "3_explicit_paths",
                   "rotate_primary": "4_rotate_primary", "rotate_foodon_taxonomy": "4_rotate_hierarchy"}
    for model, old in older_names.items():
        for k in K_VALUES:
            new = aggregate[model, k]
            old_row = previous[old, k]
            for new_key, old_key in (("ingredient_ild", "mean_user_ingredient_ild"),
                                     ("exposure_gini", "exposure_gini"),
                                     ("catalog_coverage", "catalog_coverage"),
                                     ("pair_coverage", "pair_coverage")):
                close(float(new[new_key]), float(old_row[old_key]))
    result = {"status": "five_model_diversity_verified", "per_user_rows": len(saved),
              "aggregate_rows": len(aggregate), "measurable_pairs_recomputed": checked_pairs,
              "prior_model_rows_crosschecked": len(older_names) * len(K_VALUES)}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
