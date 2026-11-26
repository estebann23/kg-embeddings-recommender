"""Independently audit every saved RQ1 test subdivision and graph category."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock

OUTPUT = ROOT / "results/rq1_test_semantic_slices"
CONFIG = ROOT / "config/rq1_test_semantic_slices.json"
BINS = ("exact_ingredient", "direct_foodon_only", "foodon_parent_only",
        "no_observed_overlap", "insufficient_ingredient_mapping")


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def verify(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["config_sha256"] == sha(CONFIG), "Config changed")
    check(sha(output / "experiment_config.json") == sha(CONFIG), "Config snapshot differs")
    for name, digest in manifest["source_sha256"].items():
        check(sha(data / f"{name}.csv") == digest, f"Source {name} changed")
    for info in manifest["files"].values():
        check(sha(output / info["path"]) == info["sha256"], f"Output {info['path']} changed")
    source = rows(data / "reviews.csv")
    train = defaultdict(list)
    test = defaultdict(list)
    all_pairs = set()
    for row in source:
        pair = row["user_id"], row["recipe_id"]
        check(pair not in all_pairs, "Duplicate source user-recipe pair")
        all_pairs.add(pair)
        if row["split"] == "train":
            train[row["user_id"]].append(row["recipe_id"])
        if row["split"] == "test":
            test[row["user_id"]].append(row)
    saved = rows(output / "test_split_assignments.csv")
    check(len(test) == 4000 and len(saved) == 32000, "User/test cardinality changed")
    check(len({r["review_id"] for r in saved}) == len(saved), "Duplicate assigned review")
    by_review = {r["review_id"]: r for records in test.values() for r in records}
    check(set(by_review) == {r["review_id"] for r in saved}, "Missing or extra test review")
    by_user = defaultdict(list)
    for row in saved:
        check({k: row[k] for k in by_review[row["review_id"]]} ==
              by_review[row["review_id"]], "Original review fields altered")
        by_user[row["user_id"]].append(row)
    check(set(by_user) == set(test), "User set mismatch")
    for user, user_rows in by_user.items():
        check(len(train[user]) == 28 and len(user_rows) == 8, "Per-user counts changed")
        expected = {r["review_id"] for r in sorted(
            test[user], key=lambda r: (hashlib.sha256(
                (config["seed"] + "|" + user + "|" + r["review_id"]).encode()).hexdigest(),
                r["review_id"]))[:4]}
        actual = {r["review_id"] for r in user_rows if r["test_half"] == "random_50"}
        check(actual == expected, f"Random selection differs for {user}")
        check(sum(r["test_half"] == "semantic_50" for r in user_rows) == 4,
              f"Semantic selection differs for {user}")
    ingredient = defaultdict(set)
    for row in rows(data / "recipe_ingredients.csv"):
        check(row["evidence_state"] == "INFERRED", "Unverified ingredient relation")
        ingredient[row["recipe_id"]].add(row["ingredient_id"])
    concept_for_ingredient = defaultdict(set)
    for row in rows(data / "foodon_links.csv"):
        check(row["evidence_state"] == "INFERRED", "Unverified FoodOn relation")
        concept_for_ingredient[row["ingredient_id"]].add(row["foodon_iri"])
    direct = defaultdict(set)
    for recipe, ids in ingredient.items():
        for item in ids:
            direct[recipe].update(concept_for_ingredient[item])
    parent_for_child = defaultdict(set)
    for row in rows(data / "foodon_hierarchy.csv"):
        check(row["evidence_state"] == "SOURCE_ASSERTED", "Unverified hierarchy relation")
        parent_for_child[row["child_iri"]].add(row["parent_iri"])
    parents = defaultdict(set)
    for recipe, concepts in direct.items():
        for concept in concepts:
            parents[recipe].update(parent_for_child[concept])
    freq = Counter(p for ids in parents.values() for p in ids)
    broad = {p for p, count in freq.items() if count > 0.05 * len(ingredient)}
    for ids in parents.values():
        ids.difference_update(broad)
    observed = Counter()
    for row in saved:
        if row["test_half"] == "random_50":
            check(row["semantic_bin"] == "" and row["matched_feature_count"] == "" and
                  row["evidence_feature_id"] == "", "Random half has diagnostic label")
            continue
        user, recipe = row["user_id"], row["recipe_id"]
        history = train[user]
        if not ingredient[recipe] or not any(ingredient[h] for h in history):
            expected, mapping = BINS[4], None
        else:
            expected, mapping = BINS[3], None
            for bin_name, family in zip(BINS[:3], (ingredient, direct, parents)):
                if any(family[recipe] & family[h] for h in history):
                    expected, mapping = bin_name, family
                    break
        check(row["semantic_bin"] == expected, f"Incorrect bin for {row['review_id']}")
        if mapping is None:
            check(row["matched_feature_count"] == "0" and
                  row["matched_history_recipe_count"] == "0" and
                  row["evidence_feature_id"] == "" and
                  row["evidence_history_recipe_id"] == "", "Spurious overlap trace")
        else:
            matches = [(h, mapping[recipe] & mapping[h]) for h in history]
            present = [(h, ids) for h, ids in matches if ids]
            check(int(row["matched_history_recipe_count"]) == len(present),
                  "History overlap count differs")
            check(int(row["matched_feature_count"]) == len(set().union(*(ids for _, ids in present))),
                  "Feature overlap count differs")
            check(row["evidence_feature_id"] in mapping[row["evidence_history_recipe_id"]] and
                  row["evidence_feature_id"] in mapping[recipe], "Unconnected evidence trace")
        observed[expected] += 1
    check({b: observed[b] for b in BINS} == manifest["bin_counts"], "Manifest bins differ")
    check(sum(observed.values()) == 16000, "Incorrect semantic total")
    check({r["review_id"] for r in rows(output / "random_50.csv")} ==
          {r["review_id"] for r in saved if r["test_half"] == "random_50"},
          "Random file membership differs")
    check({r["review_id"] for r in rows(output / "semantic_50.csv")} ==
          {r["review_id"] for r in saved if r["test_half"] == "semantic_50"},
          "Semantic file membership differs")
    for name in BINS:
        check({r["review_id"] for r in rows(output / f"bin_{name}.csv")} ==
              {r["review_id"] for r in saved if r["semantic_bin"] == name},
              f"Bin file {name} differs")
    return {"status": "PASS", "source_test_rows": len(by_review), "saved_test_rows": len(saved),
            "users_checked": len(by_user), "semantic_rows_reclassified": sum(observed.values()),
            "bin_counts": dict(observed), "source_and_output_hashes_checked": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.output), indent=2))
