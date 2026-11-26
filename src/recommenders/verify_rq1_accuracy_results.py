"""Independent full-cohort recalculation of saved RQ1 test accuracy CSVs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, sha, verify_lock
from src.recommenders.rq1_accuracy_rank import ADDENDUM, CONDITIONS, OUTPUT


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def verify(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "test_accuracy_complete"
    assert manifest["addendum_sha256"] == sha(ADDENDUM)
    assert manifest["preflight_sha256"] == sha(output / "preflight.json")
    for condition in CONDITIONS:
        assert manifest["rank_manifest_sha256"][condition] == sha(output / f"{condition}_rank_manifest.json")
    for key, filename in (("per_user_accuracy_sha256", "per_user_accuracy.csv"),
                          ("aggregate_accuracy_sha256", "aggregate_accuracy.csv"),
                          ("paired_accuracy_differences_sha256", "paired_accuracy_differences.csv")):
        assert manifest[key] == sha(output / filename)
    targets = defaultdict(dict)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "test":
            targets[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    assert len(targets) == 4000 and all(len(ratings) == 8 for ratings in targets.values())
    positive_users = {user for user, ratings in targets.items() if any(value >= 4 for value in ratings.values())}
    assert len(positive_users) == manifest["positive_test_users"]
    assert 4000 - len(positive_users) == manifest["users_without_positive_test_target"]
    saved_per_user = {(row["user_id"], row["condition"]): row
                      for row in rows(output / "per_user_accuracy.csv")}
    assert len(saved_per_user) == 4000 * len(CONDITIONS)
    recalculated = {}
    for condition in CONDITIONS:
        by_user = defaultdict(list)
        for row in rows(output / f"{condition}_top50.csv"):
            by_user[row["user_id"]].append(row)
        assert len(by_user) == 4000
        for user, entries in by_user.items():
            assert len(entries) == 50
            recorded = saved_per_user[user, condition]
            ratings = targets[user]
            positives = sum(value >= 4 for value in ratings.values())
            ideal_gains = sorted((2 if value == 5 else 1 for value in ratings.values()
                                  if value >= 4), reverse=True)
            assert int(recorded["positive_test_targets"]) == positives
            assert int(recorded["scorable_candidates"]) == int(entries[0]["scorable_candidates"])
            assert int(recorded["fallback_top50"]) == sum(row["score_source"] == "fallback"
                                                       for row in entries)
            values = {}
            for k in (10, 50):
                top = entries[:k]
                hits = sum(ratings.get(row["recipe_id"], 0) in (4, 5) for row in top)
                actual = math.fsum((2 if ratings.get(row["recipe_id"]) == 5 else
                                    1 if ratings.get(row["recipe_id"]) == 4 else 0) /
                                   math.log2(index + 2) for index, row in enumerate(top))
                ideal = math.fsum(value / math.log2(index + 2)
                                  for index, value in enumerate(ideal_gains[:k]))
                assert int(recorded[f"hits_at_{k}"]) == hits
                for column, expected in ((f"precision_at_{k}", hits / k),
                                         (f"dcg_at_{k}", actual), (f"idcg_at_{k}", ideal)):
                    assert math.isclose(float(recorded[column]), expected, abs_tol=1e-12)
                if ideal:
                    assert math.isclose(float(recorded[f"ndcg_at_{k}"]), actual / ideal, abs_tol=1e-12)
                else:
                    assert recorded[f"ndcg_at_{k}"] == ""
                values[f"precision_at_{k}"] = hits / k
                values[f"ndcg_at_{k}"] = actual / ideal if ideal else None
                values[f"hits_at_{k}"] = hits
            recalculated[user, condition] = values
    aggregate = list(rows(output / "aggregate_accuracy.csv"))
    assert len(aggregate) == 18
    for row in aggregate:
        users = positive_users if row["population"] == "positive_test_users" else set(targets)
        condition, metric = row["condition"], row["metric"]
        assert condition in CONDITIONS and metric in ("precision_at_10", "precision_at_50",
                                                      "ndcg_at_10", "ndcg_at_50")
        expected = math.fsum(recalculated[user, condition][metric] for user in users) / len(users)
        assert int(row["users"]) == len(users)
        assert math.isclose(float(row["mean"]), expected, abs_tol=1e-12)
        k = 10 if metric.endswith("10") else 50
        assert int(row["sum_hits"]) == sum(recalculated[user, condition][f"hits_at_{k}"] for user in users)
        assert int(row["users_with_hit"]) == sum(recalculated[user, condition][f"hits_at_{k}"] > 0
                                                for user in users)
        assert int(row["positive_targets"]) == sum(value >= 4 for user in users
                                                   for value in targets[user].values())
    paired = list(rows(output / "paired_accuracy_differences.csv"))
    assert len(paired) == 12
    for row in paired:
        metric, a, b = row["metric"], row["model_a"], row["model_b"]
        assert row["population"] == "positive_test_users" and a != b
        differences = [recalculated[user, a][metric] - recalculated[user, b][metric]
                       for user in positive_users]
        assert int(row["users"]) == len(positive_users)
        assert math.isclose(float(row["mean_a_minus_b"]),
                            math.fsum(differences) / len(differences), abs_tol=1e-12)
        assert int(row["wins_a"]) == sum(value > 0 for value in differences)
        assert int(row["ties"]) == sum(value == 0 for value in differences)
        assert int(row["wins_b"]) == sum(value < 0 for value in differences)
    return {"status": "accuracy_results_verified", "users": len(targets),
            "positive_test_users": len(positive_users), "per_user_rows": len(saved_per_user),
            "aggregate_rows": len(aggregate), "paired_rows": len(paired)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
