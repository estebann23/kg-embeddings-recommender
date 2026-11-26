"""Independent checks of saved four-model ingredient diversity and exposure CSVs."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from src.recommenders.contract import DATA, sha, verify_lock
from src.recommenders.rq1_diversity import (K_VALUES, MODELS, OUTPUT, RANKINGS,
                                            load_ingredients, require_preflight)
from src.recommenders.item_cf import load_recipes


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def gini_cumulative(counts: list[int]) -> float:
    ordered = sorted(counts)
    running = 0
    cumulative_sum = 0
    for value in ordered:
        running += value
        cumulative_sum += running
    return (len(ordered) + 1 - 2 * cumulative_sum / running) / len(ordered)


def verify(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    require_preflight(data, output)
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "diversity_exposure_complete"
    assert manifest["preflight_sha256"] == sha(output / "preflight.json")
    for key, filename in (("per_user_ingredient_diversity_sha256", "per_user_ingredient_diversity.csv"),
                          ("aggregate_diversity_exposure_sha256", "aggregate_diversity_exposure.csv"),
                          ("recipe_exposure_sha256", "recipe_exposure.csv"),
                          ("paired_ingredient_ild_differences_sha256", "paired_ingredient_ild_differences.csv")):
        assert manifest[key] == sha(output / filename)
    recipes = load_recipes(data)
    catalog = set(recipes)
    ingredients = load_ingredients(data, recipes)
    saved_user = {(row["model"], int(row["k"]), row["user_id"]): row
                  for row in rows(output / "per_user_ingredient_diversity.csv")}
    assert len(saved_user) == 4 * 2 * 4000
    exposure = {(model, k): Counter() for model in MODELS for k in K_VALUES}
    sampled_ild_checks = 0
    for model in MODELS:
        current_user, current = None, []
        seen_users = 0

        def check_user(user, ranking):
            nonlocal seen_users, sampled_ild_checks
            assert len(ranking) == 50 and len(set(ranking)) == 50
            for k in K_VALUES:
                top = ranking[:k]
                saved = saved_user[model, k, user]
                mapped = [ingredients[recipe] for recipe in top if recipe in ingredients]
                possible = k * (k - 1) // 2
                measurable = len(mapped) * (len(mapped) - 1) // 2
                assert int(saved["list_size"]) == k
                assert int(saved["recipes_with_mapped_ingredients"]) == len(mapped)
                assert int(saved["recipes_without_mapped_ingredients"]) == k - len(mapped)
                assert int(saved["possible_pairs"]) == possible
                assert int(saved["measurable_pairs"]) == measurable
                assert math.isclose(float(saved["pair_coverage"]), measurable / possible, abs_tol=1e-12)
                if seen_users < 100:
                    distances = [len(left ^ right) / len(left | right)
                                 for left, right in combinations(mapped, 2)]
                    if distances:
                        assert math.isclose(float(saved["ingredient_ild"]),
                                            math.fsum(distances) / len(distances), abs_tol=1e-12)
                    else:
                        assert saved["ingredient_ild"] == ""
                    sampled_ild_checks += 1
                exposure[model, k].update(top)
            seen_users += 1

        for row in rows(RANKINGS[model]):
            if current_user is None:
                current_user = row["user_id"]
            if row["user_id"] != current_user:
                check_user(current_user, current)
                current_user, current = row["user_id"], []
            current.append(row["recipe_id"])
        check_user(current_user, current)
        assert seen_users == 4000

    exposure_rows = 0
    exposure_keys = set()
    for row in rows(output / "recipe_exposure.csv"):
        model, k, recipe = row["model"], int(row["k"]), row["recipe_id"]
        assert model in MODELS and k in K_VALUES and recipe in catalog
        assert (model, k, recipe) not in exposure_keys
        exposure_keys.add((model, k, recipe))
        assert int(row["exposure_count"]) == exposure[model, k][recipe]
        exposure_rows += 1
    assert exposure_rows == len(exposure_keys) == 4 * 2 * 75972

    aggregate = list(rows(output / "aggregate_diversity_exposure.csv"))
    assert len(aggregate) == 8
    for row in aggregate:
        model, k = row["model"], int(row["k"])
        values = [saved_user[model, k, user] for user in sorted({key[2] for key in saved_user
                                                                 if key[0] == model and key[1] == k})]
        assert len(values) == int(row["users"]) == 4000
        defined = [value for value in values if value["ingredient_ild"] != ""]
        assert int(row["users_with_defined_ild"]) == len(defined)
        assert int(row["users_without_defined_ild"]) == 4000 - len(defined)
        measurable = sum(int(value["measurable_pairs"]) for value in values)
        possible = sum(int(value["possible_pairs"]) for value in values)
        mapped = sum(int(value["recipes_with_mapped_ingredients"]) for value in values)
        assert int(row["measurable_pairs"]) == measurable
        assert int(row["possible_pairs"]) == possible
        assert math.isclose(float(row["pair_coverage"]), measurable / possible, abs_tol=1e-12)
        assert int(row["mapped_recommendation_positions"]) == mapped
        assert int(row["missing_mapping_positions"]) == 4000 * k - mapped
        assert math.isclose(float(row["ingredient_mapping_position_coverage"]),
                            mapped / (4000 * k), abs_tol=1e-12)
        if defined:
            macro = math.fsum(float(value["ingredient_ild"]) for value in defined) / len(defined)
            pooled = math.fsum(float(value["ingredient_ild"]) * int(value["measurable_pairs"])
                               for value in defined) / measurable
            assert math.isclose(float(row["mean_user_ingredient_ild"]), macro, abs_tol=1e-12)
            assert math.isclose(float(row["pooled_pair_ingredient_ild"]), pooled, abs_tol=1e-12)
        counts = [exposure[model, k][recipe] for recipe in recipes]
        assert sum(counts) == int(row["total_exposures"]) == 4000 * k
        distinct = sum(value > 0 for value in counts)
        assert distinct == int(row["distinct_exposed_recipes"])
        assert math.isclose(float(row["catalog_coverage"]), distinct / len(recipes), abs_tol=1e-12)
        assert math.isclose(float(row["exposure_gini"]), gini_cumulative(counts), abs_tol=1e-12)
        assert int(row["max_recipe_exposure"]) == max(counts)
        top_count = math.ceil(0.01 * len(recipes))
        assert math.isclose(float(row["top_1_percent_exposure_share"]),
                            sum(sorted(counts, reverse=True)[:top_count]) / (4000 * k), abs_tol=1e-12)
    paired = list(rows(output / "paired_ingredient_ild_differences.csv"))
    assert len(paired) == 12
    for row in paired:
        k, a, b = int(row["k"]), row["model_a"], row["model_b"]
        differences = [float(saved_user[a, k, user]["ingredient_ild"]) -
                       float(saved_user[b, k, user]["ingredient_ild"])
                       for user in sorted({key[2] for key in saved_user if key[0] == a and key[1] == k})
                       if saved_user[a, k, user]["ingredient_ild"] != "" and
                       saved_user[b, k, user]["ingredient_ild"] != ""]
        assert int(row["paired_users"]) == len(differences)
        assert math.isclose(float(row["mean_a_minus_b"]),
                            math.fsum(differences) / len(differences), abs_tol=1e-12)
        assert int(row["wins_a"]) == sum(value > 0 for value in differences)
        assert int(row["ties"]) == sum(value == 0 for value in differences)
        assert int(row["wins_b"]) == sum(value < 0 for value in differences)
    return {"status": "diversity_exposure_verified", "per_user_rows": len(saved_user),
            "exposure_rows": exposure_rows, "aggregate_rows": len(aggregate),
            "paired_rows": len(paired), "sampled_ild_lists_recomputed": sampled_ild_checks}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
