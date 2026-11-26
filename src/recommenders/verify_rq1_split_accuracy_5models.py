"""Independently reconcile all five-model split accuracy outputs and prior hit totals."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock

CONFIG = ROOT / "config/rq1_split_accuracy_5models.json"
OUTPUT = ROOT / "results/rq1_split_accuracy_5models"


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def close(value, expected):
    return math.isclose(float(value), float(expected), rel_tol=1e-11, abs_tol=1e-12)


def verify(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seal = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    result = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    check(result["status"] == "split_test_accuracy_complete" and
          result["preflight_sha256"] == sha(output / "preflight.json") and
          result["aggregate_sha256"] == sha(output / "aggregate_accuracy.csv") and
          result["per_user_sha256"] == sha(output / "per_user_accuracy.csv"),
          "Evaluation manifest differs")
    check(seal["config_sha256"] == sha(CONFIG) == sha(output / "experiment_config.json") and
          seal["split_assignments_sha256"] == sha(ROOT / config["split_assignments"]) and
          seal["reviews_sha256"] == sha(data / "reviews.csv") and
          seal["recipes_sha256"] == sha(data / "recipes.csv"),
          "Input seal differs")
    for model, spec in config["models"].items():
        check(seal["ranking_sha256"][model] == sha(ROOT / spec["ranking"]) and
              seal["validation"][model]["selection_sha256"] == sha(ROOT / spec["validation"]),
              f"Model {model} artifact changed")
    source_test = {row["review_id"]: row for row in rows(data / "reviews.csv")
                   if row["split"] == "test"}
    check(len(source_test) == 32000, "Source test count differs")
    labels = {name: defaultdict(dict) for name in config["populations"]}
    seen = set()
    for row in rows(ROOT / config["split_assignments"]):
        key = row["review_id"]
        check(key in source_test and key not in seen, "Duplicate or foreign split review")
        seen.add(key)
        original = source_test[key]
        check(all(row[field] == original[field] for field in
                  ("review_id", "user_id", "recipe_id", "rating", "split")),
              "Source test label differs")
        destinations = [row["test_half"]]
        if row["test_half"] == "semantic_50":
            destinations.append(row["semantic_bin"])
        else:
            check(row["test_half"] == "random_50" and row["semantic_bin"] == "",
                  "Incorrect random/semantic assignment")
        for name in destinations:
            check(name in labels, "Unknown evaluation population")
            user, recipe = row["user_id"], row["recipe_id"]
            check(recipe not in labels[name][user], "Repeated population target")
            labels[name][user][recipe] = int(row["rating"])
    check(seen == set(source_test), "Incomplete split assignment")
    check(sum(map(len, labels["random_50"].values())) ==
          sum(map(len, labels["semantic_50"].values())) == 16000, "Half count differs")
    check(sum(sum(map(len, labels[name].values())) for name in config["populations"][2:]) == 16000,
          "Semantic bin total differs")
    recorded = {}
    for row in rows(output / "per_user_accuracy.csv"):
        key = row["population"], row["model"], row["user_id"]
        check(key not in recorded, "Repeated per-user metric row")
        recorded[key] = row
    check(len(recorded) == result["per_user_rows"] == 72805, "Per-user row total differs")
    aggregate = {(row["population"], row["model"]): row
                 for row in rows(output / "aggregate_accuracy.csv")}
    check(len(aggregate) == result["aggregate_rows"] == 35, "Aggregate row total differs")
    prior = defaultdict(dict)
    for row in rows(ROOT / "results/rq1_accuracy_test/per_user_accuracy.csv"):
        prior[row["condition"]][row["user_id"]] = (int(row["hits_at_10"]),
                                                     int(row["hits_at_50"]))
    for row in rows(ROOT / "results/rq1_rotate_taxonomy_test/per_user_accuracy.csv"):
        if row["condition"] == "4_rotate_hierarchy":
            prior[row["condition"]][row["user_id"]] = (int(row["hits_at_10"]),
                                                         int(row["hits_at_50"]))
    prior_name = {"interaction_only_cf": "1_item_cf",
                  "weighted_graph_paths": "3_explicit_paths",
                  "rotate_primary": "4_rotate_primary",
                  "rotate_foodon_taxonomy": "4_rotate_hierarchy"}
    checked_rows = 0
    prior_hit_checks = 0
    for model, spec in config["models"].items():
        ranks = defaultdict(list)
        for row in rows(ROOT / spec["ranking"]):
            ranks[row["user_id"]].append((int(row["rank"]), row["recipe_id"]))
        check(len(ranks) == 4000 and all(len(entries) == 50 for entries in ranks.values()),
              f"Model {model} incomplete")
        for user, entries in ranks.items():
            check([rank for rank, _ in entries] == list(range(1, 51)), "Rank sequence differs")
            check(len({recipe for _, recipe in entries}) == 50, "Repeated recommendation")
            ranked = [recipe for _, recipe in entries]
            halves = []
            for population in config["populations"]:
                if user not in labels[population]:
                    continue
                targets = labels[population][user]
                positives = sum(r >= 4 for r in targets.values())
                ideal = sorted((2 if r == 5 else 1 for r in targets.values() if r >= 4),
                               reverse=True)
                row = recorded[population, model, user]
                check(int(row["targets"]) == len(targets) and
                      int(row["positive_targets"]) == positives, "Target denominator differs")
                for k in (10, 50):
                    top = ranked[:k]
                    hits = sum(targets.get(recipe, 0) >= 4 for recipe in top)
                    dcg = sum((2 if targets.get(recipe) == 5 else
                               1 if targets.get(recipe) == 4 else 0) / math.log2(index + 2)
                              for index, recipe in enumerate(top))
                    ideal_dcg = sum(gain / math.log2(index + 2)
                                    for index, gain in enumerate(ideal[:k]))
                    check(int(row[f"hits_at_{k}"]) == hits and
                          close(row[f"precision_at_{k}"], hits / k), "Precision/hit differs")
                    if positives:
                        check(close(row[f"recall_at_{k}"], hits / positives) and
                              close(row[f"ndcg_at_{k}"], dcg / ideal_dcg),
                              "Recall/NDCG differs")
                    else:
                        check(row[f"recall_at_{k}"] == row[f"ndcg_at_{k}"] == "",
                              "Undefined metric not blank")
                if population in {"random_50", "semantic_50"}:
                    halves.append((int(row["hits_at_10"]), int(row["hits_at_50"])))
                checked_rows += 1
            check(len(halves) == 2, "Missing user half")
            if model in prior_name:
                expected = prior[prior_name[model]][user]
                check(tuple(sum(part[i] for part in halves) for i in (0, 1)) == expected,
                      "Split hits do not reconcile to old complete-test result")
                prior_hit_checks += 1
        for population in config["populations"]:
            subset = [recorded[population, model, user] for user in labels[population]]
            row = aggregate[population, model]
            positives = [entry for entry in subset if int(entry["positive_targets"]) > 0]
            check(int(row["users_with_targets"]) == len(subset) and
                  int(row["users_with_positive_targets"]) == len(positives) and
                  int(row["targets"]) == sum(int(x["targets"]) for x in subset) and
                  int(row["positive_targets"]) == sum(int(x["positive_targets"]) for x in subset),
                  "Aggregate denominator differs")
            for k in (10, 50):
                check(int(row[f"hits_at_{k}"]) == sum(int(x[f"hits_at_{k}"]) for x in subset),
                      "Aggregate hit total differs")
                for name, population_rows in (("precision", subset), ("recall", positives),
                                              ("ndcg", positives)):
                    mean = sum(float(x[f"{name}_at_{k}"]) for x in population_rows) / len(population_rows)
                    check(close(row[f"{name}_at_{k}"], mean),
                          "Aggregate macro metric differs")
    check(checked_rows == len(recorded), "Per-user table has unvisited rows")
    return {"status": "PASS", "models": len(config["models"]),
            "populations": len(config["populations"]),
            "test_assignments": len(seen), "per_user_rows_checked": checked_rows,
            "aggregate_rows_checked": len(aggregate),
            "prior_full_test_user_hit_reconciliations": prior_hit_checks}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
