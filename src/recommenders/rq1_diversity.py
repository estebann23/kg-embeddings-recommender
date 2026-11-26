"""Ingredient intra-list diversity and catalog exposure for four saved RQ1 lists."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf import load_recipes, require


OUTPUT = ROOT / "results/rq1_diversity_test"
CONFIG = ROOT / "config/rq1_diversity_experiment.json"
ACCURACY = ROOT / "results/rq1_accuracy_test"
TAXONOMY = ROOT / "results/rq1_rotate_taxonomy_test"
MODELS = ("1_item_cf", "3_explicit_paths", "4_rotate_primary", "4_rotate_hierarchy")
K_VALUES = (10, 50)
RANKINGS = {
    "1_item_cf": ACCURACY / "1_item_cf_top50.csv",
    "3_explicit_paths": ACCURACY / "3_explicit_paths_top50.csv",
    "4_rotate_primary": ACCURACY / "4_rotate_primary_top50.csv",
    "4_rotate_hierarchy": TAXONOMY / "4_rotate_hierarchy_top50.csv",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def load_ingredients(data: Path, recipes: list[str]) -> dict[str, frozenset[str]]:
    catalog = set(recipes)
    values = defaultdict(set)
    count = 0
    for row in rows(data / "recipe_ingredients.csv"):
        require(row["recipe_id"] in catalog and row["evidence_state"] == "INFERRED" and
                bool(row["ingredient_id"]), "Unexpected mapped ingredient evidence")
        values[row["recipe_id"]].add(row["ingredient_id"])
        count += 1
    result = {recipe: frozenset(items) for recipe, items in values.items()}
    require(len(result) == 65984 and sum(map(len, result.values())) == count == 586292,
            "Ingredient incidence count changed")
    return result


def preflight(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = read_json(CONFIG)
    require(config["status"] == "fixed_before_diversity_calculation" and
            config["models"] == list(MODELS) and config["k"] == list(K_VALUES),
            "Diversity protocol changed")
    from src.recommenders.verify_rq1_accuracy_rank import verify as verify_accuracy_rank
    from src.recommenders.verify_rq1_rotate_taxonomy_rank import verify as verify_taxonomy_rank
    require(verify_accuracy_rank()["status"] == "test_blind_rankings_verified", "Prior rankings changed")
    require(verify_taxonomy_rank()["status"] == "hierarchy_rankings_verified", "Taxonomy rankings changed")
    source = read_json(data / "manifest.json")
    require(source["output_sha256"]["recipe_ingredients"] == sha(data / "recipe_ingredients.csv"),
            "Ingredient source changed")
    recipes = load_recipes(data)
    ingredients = load_ingredients(data, recipes)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "diversity_preflight_passed", "dataset_id": data.name,
        "models": list(MODELS), "k": list(K_VALUES),
        "users": 4000, "catalog_recipes": len(recipes),
        "recipes_with_mapped_ingredients": len(ingredients),
        "mapped_ingredient_incidences": sum(map(len, ingredients.values())),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "config_sha256": sha(CONFIG), "ingredient_source_sha256": sha(data / "recipe_ingredients.csv"),
        "recipes_sha256": sha(data / "recipes.csv"),
        "ranking_sha256": {model: sha(RANKINGS[model]) for model in MODELS},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "preflight.json", result)
    return result


def require_preflight(data: Path, output: Path) -> dict:
    verify_lock(data)
    value = read_json(output / "preflight.json")
    require(value["status"] == "diversity_preflight_passed" and
            value["config_sha256"] == sha(CONFIG) and
            value["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            value["ingredient_source_sha256"] == sha(data / "recipe_ingredients.csv") and
            value["recipes_sha256"] == sha(data / "recipes.csv") and
            all(value["ranking_sha256"][model] == sha(RANKINGS[model]) for model in MODELS),
            "Diversity preflight seal changed")
    return value


def list_ild(recommended: list[str], ingredients: dict[str, frozenset[str]]) -> tuple[float | None, int, int]:
    mapped = [ingredients[recipe] for recipe in recommended if recipe in ingredients]
    possible = len(recommended) * (len(recommended) - 1) // 2
    measurable = len(mapped) * (len(mapped) - 1) // 2
    if not measurable:
        return None, possible, measurable
    distance_sum = 0.0
    for left, right in combinations(mapped, 2):
        shared = len(left & right)
        distance_sum += 1 - shared / (len(left) + len(right) - shared)
    return distance_sum / measurable, possible, measurable


def gini(counts: list[int]) -> float:
    require(counts and all(count >= 0 for count in counts), "Invalid exposure counts")
    total = sum(counts)
    require(total > 0, "Empty exposure distribution")
    ordered = sorted(counts)
    n = len(ordered)
    weighted = sum(position * count for position, count in enumerate(ordered, 1))
    result = 2 * weighted / (n * total) - (n + 1) / n
    require(-1e-12 <= result <= 1 + 1e-12, "Invalid exposure Gini")
    return max(0.0, min(1.0, result))


def grouped_rankings(path: Path):
    current_user, current = None, []
    for row in rows(path):
        user = row["user_id"]
        if current_user is None:
            current_user = user
        if user != current_user:
            yield current_user, current
            current_user, current = user, []
        current.append(row["recipe_id"])
    if current_user is not None:
        yield current_user, current


def run(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    pre = require_preflight(data, output)
    recipes = load_recipes(data)
    ingredients = load_ingredients(data, recipes)
    per_user = defaultdict(dict)
    exposure = {(model, k): Counter() for model in MODELS for k in K_VALUES}
    per_user_path = output / "per_user_ingredient_diversity.csv"
    with per_user_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "user_id", "k", "list_size", "recipes_with_mapped_ingredients",
                         "recipes_without_mapped_ingredients", "possible_pairs", "measurable_pairs",
                         "pair_coverage", "ingredient_ild"))
        for model in MODELS:
            user_count = 0
            for user, ranking in grouped_rankings(RANKINGS[model]):
                require(len(ranking) == 50 and len(set(ranking)) == 50,
                        "Ranking is not a distinct top-50 list")
                for k in K_VALUES:
                    selected = ranking[:k]
                    ild, possible, measurable = list_ild(selected, ingredients)
                    mapped = sum(recipe in ingredients for recipe in selected)
                    writer.writerow((model, user, k, k, mapped, k - mapped,
                                     possible, measurable, format(measurable / possible, ".17g"),
                                     "" if ild is None else format(ild, ".17g")))
                    per_user[(model, k)][user] = (ild, mapped, measurable, possible)
                    exposure[model, k].update(selected)
                user_count += 1
            require(user_count == pre["users"], "Ranking user count changed")

    aggregate_path = output / "aggregate_diversity_exposure.csv"
    exposure_path = output / "recipe_exposure.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as aggregate_stream, \
         exposure_path.open("w", encoding="utf-8", newline="") as exposure_stream:
        aggregate_writer = csv.writer(aggregate_stream, lineterminator="\n")
        exposure_writer = csv.writer(exposure_stream, lineterminator="\n")
        aggregate_writer.writerow(("model", "k", "users", "users_with_defined_ild",
                                   "users_without_defined_ild", "mean_user_ingredient_ild",
                                   "pooled_pair_ingredient_ild", "measurable_pairs", "possible_pairs",
                                   "pair_coverage", "mapped_recommendation_positions",
                                   "missing_mapping_positions", "ingredient_mapping_position_coverage",
                                   "total_exposures", "distinct_exposed_recipes", "catalog_coverage",
                                   "exposure_gini", "max_recipe_exposure", "top_1_percent_exposure_share"))
        exposure_writer.writerow(("model", "k", "recipe_id", "exposure_count"))
        for model in MODELS:
            for k in K_VALUES:
                values = per_user[model, k]
                require(len(values) == 4000, "Per-user ILD output incomplete")
                defined = [entry for entry in values.values() if entry[0] is not None]
                measurable = sum(entry[2] for entry in values.values())
                possible = sum(entry[3] for entry in values.values())
                mapped = sum(entry[1] for entry in values.values())
                counts = [exposure[model, k][recipe] for recipe in recipes]
                total = sum(counts)
                require(total == 4000 * k and possible == 4000 * k * (k - 1) // 2,
                        "Exposure or pair denominator changed")
                for recipe, count in zip(recipes, counts):
                    exposure_writer.writerow((model, k, recipe, count))
                distinct = sum(count > 0 for count in counts)
                top_one_percent = math.ceil(0.01 * len(recipes))
                top_share = sum(sorted(counts, reverse=True)[:top_one_percent]) / total
                macro = sum(entry[0] for entry in defined) / len(defined) if defined else None
                pooled = sum(entry[0] * entry[2] for entry in defined) / measurable if measurable else None
                aggregate_writer.writerow((model, k, len(values), len(defined), len(values) - len(defined),
                                           "" if macro is None else format(macro, ".17g"),
                                           "" if pooled is None else format(pooled, ".17g"),
                                           measurable, possible, format(measurable / possible, ".17g"),
                                           mapped, total - mapped, format(mapped / total, ".17g"),
                                           total, distinct, format(distinct / len(recipes), ".17g"),
                                           format(gini(counts), ".17g"), max(counts),
                                           format(top_share, ".17g")))

    paired_path = output / "paired_ingredient_ild_differences.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("k", "model_a", "model_b", "paired_users", "mean_a_minus_b",
                         "wins_a", "ties", "wins_b"))
        for k in K_VALUES:
            for a, b in combinations(MODELS, 2):
                differences = [per_user[a, k][user][0] - per_user[b, k][user][0]
                               for user in per_user[a, k]
                               if per_user[a, k][user][0] is not None and
                               per_user[b, k][user][0] is not None]
                writer.writerow((k, a, b, len(differences),
                                 "" if not differences else format(sum(differences) / len(differences), ".17g"),
                                 sum(value > 0 for value in differences),
                                 sum(value == 0 for value in differences),
                                 sum(value < 0 for value in differences)))
    result = {
        "status": "diversity_exposure_complete", "models": list(MODELS), "k": list(K_VALUES),
        "users": 4000, "catalog_recipes": len(recipes),
        "preflight_sha256": sha(output / "preflight.json"),
        "per_user_ingredient_diversity_sha256": sha(per_user_path),
        "aggregate_diversity_exposure_sha256": sha(aggregate_path),
        "recipe_exposure_sha256": sha(exposure_path),
        "paired_ingredient_ild_differences_sha256": sha(paired_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "evaluation_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    args = parser.parse_args()
    print(json.dumps(preflight() if args.command == "preflight" else run(), indent=2))


if __name__ == "__main__":
    main()
