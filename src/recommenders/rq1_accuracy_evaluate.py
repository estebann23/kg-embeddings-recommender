"""Evaluate frozen test-blind RQ1 top-50 rankings on held-out test ratings."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.rq1_accuracy_rank import ADDENDUM, CONDITIONS, OUTPUT, require_preflight
from src.recommenders.verify_rq1_accuracy_rank import verify as verify_rankings


K_VALUES = (10, 50)
METRICS = ("precision_at_10", "precision_at_50", "ndcg_at_10", "ndcg_at_50")


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def gain(rating: int) -> int:
    return 2 if rating == 5 else 1 if rating == 4 else 0


def metric_values(ranking: list[str], targets: dict[str, int]) -> dict:
    positive = sum(rating >= 4 for rating in targets.values())
    ideal_gains = sorted((gain(rating) for rating in targets.values() if gain(rating) > 0), reverse=True)
    result = {"positive_test_targets": positive}
    for k in K_VALUES:
        observed = ranking[:k]
        hits = sum(targets.get(recipe, 0) >= 4 for recipe in observed)
        dcg = sum(gain(targets.get(recipe, 0)) / math.log2(position + 2)
                  for position, recipe in enumerate(observed))
        ideal = sum(value / math.log2(position + 2)
                    for position, value in enumerate(ideal_gains[:k]))
        result[f"hits_at_{k}"] = hits
        result[f"precision_at_{k}"] = hits / k
        result[f"dcg_at_{k}"] = dcg
        result[f"idcg_at_{k}"] = ideal
        result[f"ndcg_at_{k}"] = dcg / ideal if ideal else None
    return result


def evaluate(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    preflight = require_preflight(data, output)
    rank_check = verify_rankings(data, output)
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    assert addendum["metrics"] == list(METRICS)
    assert addendum["test_label_access"].startswith("only after all three rank lists")
    assert rank_check["status"] == "test_blind_rankings_verified"
    targets = defaultdict(dict)
    train_pairs = defaultdict(set)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train":
            train_pairs[row["user_id"]].add(row["recipe_id"])
        elif row["split"] == "test":
            targets[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    assert len(targets) == len(train_pairs) == 4000
    assert all(len(targets[user]) == 8 and len(train_pairs[user]) == 28 and
               not set(targets[user]) & train_pairs[user] for user in targets)
    positive_users = sorted(user for user in targets
                            if any(rating >= 4 for rating in targets[user].values()))
    positive_user_set = set(positive_users)
    assert positive_users, "No positive test users"
    generated = datetime.now(timezone.utc).isoformat()
    per_user_path = output / "per_user_accuracy.csv"
    per_user = defaultdict(dict)
    with per_user_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("condition", "user_id", "positive_test_targets", "scorable_candidates",
                         "fallback_top50", "hits_at_10", "hits_at_50", "precision_at_10",
                         "precision_at_50", "dcg_at_10", "dcg_at_50", "idcg_at_10",
                         "idcg_at_50", "ndcg_at_10", "ndcg_at_50"))
        for condition in CONDITIONS:
            current_user, current_rows = None, []

            def commit(user, entries):
                assert len(entries) == 50
                ranking = [row["recipe_id"] for row in entries]
                result = metric_values(ranking, targets[user])
                result["scorable_candidates"] = int(entries[0]["scorable_candidates"])
                result["fallback_top50"] = sum(row["score_source"] == "fallback" for row in entries)
                per_user[user][condition] = result
                writer.writerow((condition, user, result["positive_test_targets"],
                                 result["scorable_candidates"], result["fallback_top50"],
                                 result["hits_at_10"], result["hits_at_50"],
                                 *(format(result[f"precision_at_{k}"], ".17g") for k in K_VALUES),
                                 *(format(result[f"dcg_at_{k}"], ".17g") for k in K_VALUES),
                                 *(format(result[f"idcg_at_{k}"], ".17g") for k in K_VALUES),
                                 *("" if result[f"ndcg_at_{k}"] is None else
                                   format(result[f"ndcg_at_{k}"], ".17g") for k in K_VALUES)))

            for row in rows(output / f"{condition}_top50.csv"):
                if current_user is None:
                    current_user = row["user_id"]
                if row["user_id"] != current_user:
                    commit(current_user, current_rows)
                    current_user, current_rows = row["user_id"], []
                current_rows.append(row)
            commit(current_user, current_rows)
    assert len(per_user) == 4000 and all(set(value) == set(CONDITIONS) for value in per_user.values())

    aggregate_path = output / "aggregate_accuracy.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("population", "condition", "metric", "users", "mean", "sum_hits",
                         "users_with_hit", "positive_targets", "definition"))
        for population, users in (("positive_test_users", positive_users),
                                  ("all_users", sorted(targets))):
            for condition in CONDITIONS:
                for metric in METRICS:
                    if population == "all_users" and metric.startswith("ndcg"):
                        continue
                    k = 10 if metric.endswith("10") else 50
                    values = [per_user[user][condition][metric] for user in users]
                    assert all(value is not None for value in values)
                    hits = sum(per_user[user][condition][f"hits_at_{k}"] for user in users)
                    hit_users = sum(per_user[user][condition][f"hits_at_{k}"] > 0 for user in users)
                    positive_targets = sum(per_user[user][condition]["positive_test_targets"] for user in users)
                    definition = ("mean per-user observed positive hits / K" if metric.startswith("precision")
                                  else "mean per-user DCG / ideal DCG using observed test gains 0,0,0,1,2")
                    writer.writerow((population, condition, metric, len(users),
                                     format(sum(values) / len(values), ".17g"), hits, hit_users,
                                     positive_targets, definition))

    paired_path = output / "paired_accuracy_differences.csv"
    pairs = (("3_explicit_paths", "1_item_cf"), ("4_rotate_primary", "1_item_cf"),
             ("4_rotate_primary", "3_explicit_paths"))
    with paired_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("population", "metric", "model_a", "model_b", "users", "mean_a_minus_b",
                         "wins_a", "ties", "wins_b"))
        for metric in METRICS:
            for a, b in pairs:
                differences = [per_user[user][a][metric] - per_user[user][b][metric]
                               for user in positive_users]
                writer.writerow(("positive_test_users", metric, a, b, len(differences),
                                 format(sum(differences) / len(differences), ".17g"),
                                 sum(value > 0 for value in differences),
                                 sum(value == 0 for value in differences),
                                 sum(value < 0 for value in differences)))

    result = {
        "status": "test_accuracy_complete", "conditions": list(CONDITIONS),
        "k": list(K_VALUES), "all_users": len(targets),
        "positive_test_users": len(positive_users),
        "users_without_positive_test_target": len(targets) - len(positive_users),
        "test_ratings": sum(map(len, targets.values())),
        "preflight_sha256": sha(output / "preflight.json"),
        "addendum_sha256": sha(ADDENDUM),
        "rank_manifest_sha256": {condition: sha(output / f"{condition}_rank_manifest.json")
                                 for condition in CONDITIONS},
        "per_user_accuracy_sha256": sha(per_user_path),
        "aggregate_accuracy_sha256": sha(aggregate_path),
        "paired_accuracy_differences_sha256": sha(paired_path),
        "generated_utc": generated,
        "note": "No model or setting was changed using test ratings; this is an accuracy-only three-condition pass."
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
