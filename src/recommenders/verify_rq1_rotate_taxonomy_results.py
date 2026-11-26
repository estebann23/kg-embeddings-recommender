"""Independent recalculation of primary versus FoodOn-taxonomy RotatE metrics."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, sha, verify_lock
from src.recommenders.rq1_rotate_taxonomy_test import (ADDENDUM, CONDITIONS, OUTPUT,
                                                       PRIOR, require_preflight)


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def verify(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    require_preflight(data, output)
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "taxonomy_test_accuracy_complete"
    assert manifest["preflight_sha256"] == sha(output / "preflight.json")
    assert manifest["rank_manifest_sha256"] == sha(output / "rank_manifest.json")
    assert manifest["prior_evaluation_manifest_sha256"] == sha(PRIOR / "evaluation_manifest.json")
    for key, name in (("per_user_accuracy_sha256", "per_user_accuracy.csv"),
                      ("aggregate_accuracy_sha256", "aggregate_accuracy.csv"),
                      ("paired_differences_sha256", "paired_differences.csv")):
        assert manifest[key] == sha(output / name)
    targets = defaultdict(dict)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "test":
            targets[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    assert len(targets) == 4000 and all(len(value) == 8 for value in targets.values())
    saved = {(row["user_id"], row["condition"]): row
             for row in rows(output / "per_user_accuracy.csv")}
    assert len(saved) == 8000
    prior_primary = {row["user_id"]: row for row in rows(PRIOR / "per_user_accuracy.csv")
                     if row["condition"] == "4_rotate_primary"}
    assert len(prior_primary) == 4000
    recalculated = {}
    for condition, path in (("4_rotate_primary", PRIOR / "4_rotate_primary_top50.csv"),
                            ("4_rotate_hierarchy", output / "4_rotate_hierarchy_top50.csv")):
        grouped = defaultdict(list)
        for row in rows(path):
            grouped[row["user_id"]].append(row["recipe_id"])
        assert len(grouped) == 4000
        for user, ranked in grouped.items():
            assert len(ranked) == 50
            observed = targets[user]
            ideal_gains = sorted((2 if rating == 5 else 1 for rating in observed.values()
                                  if rating >= 4), reverse=True)
            assert ideal_gains
            record = saved[user, condition]
            assert int(record["positive_test_targets"]) == len(ideal_gains)
            result = {}
            for k in (10, 50):
                hits = sum(observed.get(recipe) in (4, 5) for recipe in ranked[:k])
                dcg = math.fsum((2 if observed.get(recipe) == 5 else
                                 1 if observed.get(recipe) == 4 else 0) / math.log2(position + 2)
                                for position, recipe in enumerate(ranked[:k]))
                ideal = math.fsum(gain / math.log2(position + 2)
                                  for position, gain in enumerate(ideal_gains[:k]))
                result[f"precision_at_{k}"] = hits / k
                result[f"ndcg_at_{k}"] = dcg / ideal
                result[f"hits_at_{k}"] = hits
                assert int(record[f"hits_at_{k}"]) == hits
                assert math.isclose(float(record[f"precision_at_{k}"]), hits / k, abs_tol=1e-12)
                assert math.isclose(float(record[f"ndcg_at_{k}"]), dcg / ideal, abs_tol=1e-12)
                if condition == "4_rotate_primary":
                    assert record[f"hits_at_{k}"] == prior_primary[user][f"hits_at_{k}"]
                    assert record[f"precision_at_{k}"] == prior_primary[user][f"precision_at_{k}"]
                    assert record[f"ndcg_at_{k}"] == prior_primary[user][f"ndcg_at_{k}"]
            recalculated[user, condition] = result
    aggregate = list(rows(output / "aggregate_accuracy.csv"))
    assert len(aggregate) == 8
    for row in aggregate:
        condition, metric = row["condition"], row["metric"]
        assert condition in CONDITIONS
        values = [recalculated[user, condition][metric] for user in targets]
        k = 10 if metric.endswith("10") else 50
        hits = [recalculated[user, condition][f"hits_at_{k}"] for user in targets]
        assert int(row["users"]) == 4000
        assert math.isclose(float(row["mean"]), math.fsum(values) / 4000, abs_tol=1e-12)
        assert int(row["sum_hits"]) == sum(hits)
        assert int(row["users_with_hit"]) == sum(value > 0 for value in hits)
    paired = list(rows(output / "paired_differences.csv"))
    assert len(paired) == 4
    for row in paired:
        metric = row["metric"]
        diff = [recalculated[user, "4_rotate_hierarchy"][metric] -
                recalculated[user, "4_rotate_primary"][metric] for user in targets]
        assert int(row["users"]) == 4000
        assert math.isclose(float(row["mean_hierarchy_minus_primary"]),
                            math.fsum(diff) / 4000, abs_tol=1e-12)
        assert int(row["hierarchy_wins"]) == sum(value > 0 for value in diff)
        assert int(row["ties"]) == sum(value == 0 for value in diff)
        assert int(row["primary_wins"]) == sum(value < 0 for value in diff)
    return {"status": "taxonomy_accuracy_verified", "users": len(targets),
            "per_user_rows": len(saved), "aggregate_rows": len(aggregate), "paired_rows": len(paired)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
