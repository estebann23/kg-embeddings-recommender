"""Descriptive diversity of five frozen, test-blind RQ1 top-50 rankings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf import load_recipes, require
from src.recommenders.rq1_diversity import gini, list_ild, load_ingredients

CONFIG = ROOT / "config/rq1_diversity_5models.json"
OUTPUT = ROOT / "results/rq1_diversity_5models_test"
SPLIT = ROOT / "results/rq1_test_semantic_slices/test_split_assignments.csv"
K_VALUES = (10, 50)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def ranking_paths(config: dict) -> dict[str, Path]:
    paths = {model: ROOT / name for model, name in config["models"].items()}
    require(len(paths) == 5 and all(path.is_file() for path in paths.values()),
            "Five ranking files required")
    return paths


def split_users() -> set[str]:
    by_half = defaultdict(Counter)
    for row in rows(SPLIT):
        require(row["split"] == "test" and row["test_half"] in ("random_50", "semantic_50"),
                "Unexpected test split assignment")
        by_half[row["test_half"]][row["user_id"]] += 1
    require(set(by_half) == {"random_50", "semantic_50"}, "Missing test half")
    require(len(by_half["random_50"]) == len(by_half["semantic_50"]) == 4000,
            "Test-half users changed")
    require(by_half["random_50"] == by_half["semantic_50"] and
            all(count == 4 for count in by_half["random_50"].values()),
            "The halves do not contain the same four-target users")
    return set(by_half["random_50"])


def preflight(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = read_json(CONFIG)
    require(config["status"] == "fixed_before_five_model_diversity_calculation" and
            config["dataset_id"] == data.name and config["k"] == list(K_VALUES),
            "Five-model diversity contract changed")
    paths = ranking_paths(config)
    require(set(paths) == {"interaction_only_cf", "weighted_graph_paths", "semantic_mf",
                           "rotate_primary", "rotate_foodon_taxonomy"},
            "Unexpected model set")
    from src.recommenders.verify_rq1_accuracy_rank import verify as verify_original
    from src.recommenders.verify_rq1_rotate_taxonomy_rank import verify as verify_taxonomy
    from src.recommenders.verify_mf_semantic import verify as verify_mf

    require(verify_original()["status"] == "test_blind_rankings_verified",
            "CF, graph, or primary RotatE rankings failed verification")
    require(verify_taxonomy()["status"] == "hierarchy_rankings_verified",
            "Taxonomy RotatE rankings failed verification")
    require(verify_mf()["status"] == "PASS", "Semantic MF rankings failed verification")
    source = read_json(data / "manifest.json")
    require(source["output_sha256"]["recipe_ingredients"] == sha(data / "recipe_ingredients.csv"),
            "Mapped-ingredient source changed")
    recipes = load_recipes(data)
    ingredients = load_ingredients(data, recipes)
    users = split_users()
    require(len(recipes) == 75972 and len(ingredients) == 65984, "Catalog changed")
    output.mkdir(parents=True, exist_ok=True)
    value = {
        "status": "five_model_diversity_preflight_passed",
        "dataset_id": data.name, "models": list(paths), "k": list(K_VALUES),
        "users": len(users), "catalog_recipes": len(recipes),
        "recipes_with_mapped_ingredients": len(ingredients),
        "config_sha256": sha(CONFIG), "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "recipes_sha256": sha(data / "recipes.csv"),
        "ingredient_source_sha256": sha(data / "recipe_ingredients.csv"),
        "split_sha256": sha(SPLIT),
        "ranking_sha256": {model: sha(path) for model, path in paths.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "preflight.json", value)
    return value


def require_preflight(data: Path, output: Path) -> tuple[dict, dict[str, Path], set[str]]:
    verify_lock(data)
    config = read_json(CONFIG)
    paths = ranking_paths(config)
    value = read_json(output / "preflight.json")
    require(value["status"] == "five_model_diversity_preflight_passed" and
            value["config_sha256"] == sha(CONFIG) and
            value["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            value["recipes_sha256"] == sha(data / "recipes.csv") and
            value["ingredient_source_sha256"] == sha(data / "recipe_ingredients.csv") and
            value["split_sha256"] == sha(SPLIT) and
            all(value["ranking_sha256"][model] == sha(path) for model, path in paths.items()),
            "Five-model preflight seal changed")
    users = split_users()
    require(len(users) == value["users"], "User set changed")
    return value, paths, users


def grouped_rankings(path: Path):
    user, current, ranks = None, [], []
    seen = set()
    for row in rows(path):
        if user is None:
            user = row["user_id"]
        if row["user_id"] != user:
            require(user not in seen and ranks == list(range(1, 51)), "Ranking order changed")
            seen.add(user)
            yield user, current
            user, current, ranks = row["user_id"], [], []
        current.append(row["recipe_id"])
        ranks.append(int(row["rank"]))
    if user is not None:
        require(user not in seen and ranks == list(range(1, 51)), "Ranking order changed")
        yield user, current


def run(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    pre, paths, expected_users = require_preflight(data, output)
    recipes = load_recipes(data)
    catalog = set(recipes)
    ingredients = load_ingredients(data, recipes)
    per_user_path = output / "per_user_diversity.csv"
    aggregate_path = output / "aggregate_diversity.csv"
    aggregate = []
    with per_user_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "user_id", "k", "ingredient_ild", "pair_coverage",
                         "ingredient_position_coverage"))
        for model, path in paths.items():
            exposures = {k: Counter() for k in K_VALUES}
            values = {k: [] for k in K_VALUES}
            seen_users = set()
            for user, ranking in grouped_rankings(path):
                require(user in expected_users and user not in seen_users and len(ranking) == 50 and
                        len(set(ranking)) == 50 and set(ranking) <= catalog,
                        "Invalid ranking user, length, or recipe")
                seen_users.add(user)
                for k in K_VALUES:
                    top = ranking[:k]
                    ild, possible, measurable = list_ild(top, ingredients)
                    mapped = sum(recipe in ingredients for recipe in top)
                    pair_coverage = measurable / possible
                    position_coverage = mapped / k
                    require(0 <= pair_coverage <= 1 and 0 <= position_coverage <= 1 and
                            (ild is None or 0 <= ild <= 1), "Metric outside unit interval")
                    values[k].append((ild, measurable, possible, mapped))
                    exposures[k].update(top)
                    writer.writerow((model, user, k, "" if ild is None else format(ild, ".17g"),
                                     format(pair_coverage, ".17g"),
                                     format(position_coverage, ".17g")))
            require(seen_users == expected_users, "Incomplete model user cohort")
            for k in K_VALUES:
                entries = values[k]
                defined = [entry[0] for entry in entries if entry[0] is not None]
                require(defined, "No measurable ingredient pairs")
                counts = [exposures[k][recipe] for recipe in recipes]
                require(sum(counts) == len(expected_users) * k and
                        sum(entry[2] for entry in entries) == len(expected_users) * k * (k - 1) // 2,
                        "Exposure or pair denominator changed")
                aggregate.append({
                    "model": model, "k": k,
                    "ingredient_ild": sum(defined) / len(defined),
                    "exposure_gini": gini(counts),
                    "catalog_coverage": sum(count > 0 for count in counts) / len(recipes),
                    "pair_coverage": sum(entry[1] for entry in entries) / sum(entry[2] for entry in entries),
                    "ingredient_position_coverage": sum(entry[3] for entry in entries) / (len(entries) * k),
                    "defined_user_coverage": len(defined) / len(entries),
                    "user_list_coverage": len(seen_users) / len(expected_users),
                })
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ("model", "k", "ingredient_ild", "exposure_gini", "catalog_coverage",
                  "pair_coverage", "ingredient_position_coverage", "defined_user_coverage",
                  "user_list_coverage")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in aggregate:
            require(all(0 <= record[key] <= 1 for key in fields[2:]), "Aggregate outside unit interval")
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value
                             for key, value in record.items()})
    result = {
        "status": "five_model_diversity_complete", "models": list(paths), "k": list(K_VALUES),
        "users": pre["users"], "catalog_recipes": len(recipes),
        "per_user_rows": len(paths) * len(expected_users) * len(K_VALUES),
        "aggregate_rows": len(aggregate),
        "preflight_sha256": sha(output / "preflight.json"),
        "per_user_sha256": sha(per_user_path), "aggregate_sha256": sha(aggregate_path),
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
