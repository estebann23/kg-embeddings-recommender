"""Freeze a per-user random half and graph-profiled half of the RQ1 test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock

CONFIG = ROOT / "config/rq1_test_semantic_slices.json"
OUTPUT = ROOT / "results/rq1_test_semantic_slices"
BINS = (
    "exact_ingredient",
    "direct_foodon_only",
    "foodon_parent_only",
    "no_observed_overlap",
    "insufficient_ingredient_mapping",
)
FIELDS = (
    "review_id", "user_id", "recipe_id", "rating", "is_rated", "submit_date",
    "split", "source_file", "test_half", "semantic_bin", "matched_feature_count",
    "matched_history_recipe_count", "evidence_feature_id", "evidence_history_recipe_id",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def features(data: Path):
    ingredients = defaultdict(set)
    for row in read_csv(data / "recipe_ingredients.csv"):
        require(row["evidence_state"] == "INFERRED", "Unexpected ingredient evidence")
        ingredients[row["recipe_id"]].add(row["ingredient_id"])
    direct_by_ingredient = defaultdict(set)
    for row in read_csv(data / "foodon_links.csv"):
        require(row["evidence_state"] == "INFERRED", "Unexpected FoodOn evidence")
        direct_by_ingredient[row["ingredient_id"]].add(row["foodon_iri"])
    direct = {recipe: set().union(*(direct_by_ingredient[i] for i in ids))
              for recipe, ids in ingredients.items()}
    parent_by_child = defaultdict(set)
    for row in read_csv(data / "foodon_hierarchy.csv"):
        require(row["evidence_state"] == "SOURCE_ASSERTED", "Unexpected parent evidence")
        parent_by_child[row["child_iri"]].add(row["parent_iri"])
    raw_parents = {recipe: set().union(*(parent_by_child[c] for c in ids))
                   for recipe, ids in direct.items()}
    parent_frequency = Counter(p for ids in raw_parents.values() for p in ids)
    broad = {p for p, n in parent_frequency.items() if n > 0.05 * len(ingredients)}
    parents = {recipe: ids - broad for recipe, ids in raw_parents.items()}
    return ingredients, direct, parents, broad


def random_key(seed: str, row: dict) -> str:
    return hashlib.sha256(f'{seed}|{row["user_id"]}|{row["review_id"]}'.encode()).hexdigest()


def classify(recipe: str, history: list[str], feature_maps) -> tuple[str, int, int, str, str]:
    ingredients, direct, parents = feature_maps
    if not ingredients.get(recipe) or not any(ingredients.get(h) for h in history):
        return BINS[4], 0, 0, "", ""
    for name, mapping in zip(BINS[:3], (ingredients, direct, parents)):
        candidate = mapping.get(recipe, set())
        if not candidate:
            continue
        matches = [(h, candidate & mapping.get(h, set())) for h in history]
        matched = [(h, shared) for h, shared in matches if shared]
        if matched:
            union = set().union(*(shared for _, shared in matched))
            source, feature = min((h, feature) for h, shared in matched for feature in shared)
            return name, len(union), len(matched), feature, source
    # Disjoint means no overlap in *observed* audited links, not proof of absent ingredients.
    return BINS[3], 0, 0, "", ""


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(config["dataset_id"] == data.name and tuple(config["bin_precedence"]) == BINS,
            "Slice config disagrees with input or code")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in ("reviews", "recipe_ingredients", "foodon_links", "foodon_hierarchy"):
        require(sha(data / f"{name}.csv") == manifest["output_sha256"][name],
                f"{name} does not match source manifest")
    history = defaultdict(list)
    tests = defaultdict(list)
    pair_seen = set()
    for row in read_csv(data / "reviews.csv"):
        pair = row["user_id"], row["recipe_id"]
        require(pair not in pair_seen, "Duplicate user-recipe pair")
        pair_seen.add(pair)
        if row["split"] == "train":
            history[row["user_id"]].append(row["recipe_id"])
        elif row["split"] == "test":
            tests[row["user_id"]].append(row)
    require(len(tests) == 4000 and set(tests) == set(history), "Unexpected user cohort")
    require(all(len(history[u]) == 28 and len(tests[u]) == 8 for u in tests),
            "Unexpected per-user split sizes")
    ingredients, direct, parents, broad = features(data)
    selected = []
    for user in sorted(tests):
        rows = sorted(tests[user], key=lambda r: (random_key(config["seed"], r), r["review_id"]))
        for index, row in enumerate(rows):
            result = dict(row)
            result.update({"test_half": "random_50" if index < 4 else "semantic_50",
                           "semantic_bin": "", "matched_feature_count": "",
                           "matched_history_recipe_count": "", "evidence_feature_id": "",
                           "evidence_history_recipe_id": ""})
            if index >= 4:
                name, count, n_history, feature, source = classify(
                    row["recipe_id"], history[user], (ingredients, direct, parents))
                result.update({"semantic_bin": name, "matched_feature_count": count,
                               "matched_history_recipe_count": n_history,
                               "evidence_feature_id": feature,
                               "evidence_history_recipe_id": source})
            selected.append(result)
    selected.sort(key=lambda r: (r["user_id"], r["review_id"]))
    counts = Counter(r["semantic_bin"] for r in selected if r["test_half"] == "semantic_50")
    require(len(selected) == 32000 and sum(counts.values()) == 16000, "Incorrect slice sizes")
    require(all(counts[b] > 0 for b in BINS), "At least one diagnostic bin is empty")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, output / "experiment_config.json")
    files = {"assignments": "test_split_assignments.csv", "random": "random_50.csv",
             "semantic": "semantic_50.csv", "config_snapshot": "experiment_config.json"}
    write_csv(output / files["assignments"], selected)
    write_csv(output / files["random"], [r for r in selected if r["test_half"] == "random_50"])
    write_csv(output / files["semantic"], [r for r in selected if r["test_half"] == "semantic_50"])
    for name in BINS:
        filename = f"bin_{name}.csv"
        files[name] = filename
        write_csv(output / filename, [r for r in selected if r["semantic_bin"] == name])
    report = {
        "schema_version": 1, "dataset_id": data.name, "config_sha256": sha(CONFIG),
        "source_sha256": {name: sha(data / f"{name}.csv") for name in
                          ("reviews", "recipe_ingredients", "foodon_links", "foodon_hierarchy")},
        "files": {name: {"path": filename, "sha256": sha(output / filename)}
                  for name, filename in files.items()},
        "users": len(tests), "test_rows": len(selected), "random_rows": 16000,
        "semantic_rows": 16000, "bin_counts": {name: counts[name] for name in BINS},
        "mapped_recipes": len(ingredients), "broad_parent_concepts_excluded": len(broad),
        "test_rating_distribution_after_assignment": {
            half: dict(sorted(Counter(r["rating"] for r in selected if r["test_half"] == half).items()))
            for half in ("random_50", "semantic_50")
        },
        "bin_summary_after_assignment": {
            name: {
                "rows": counts[name],
                "users": len({r["user_id"] for r in selected if r["semantic_bin"] == name}),
                "positive_test_ratings_4_or_5": sum(
                    r["rating"] in {"4", "5"} for r in selected if r["semantic_bin"] == name),
            } for name in BINS
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
