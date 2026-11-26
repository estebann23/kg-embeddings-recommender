"""Test-blind verification of primary RotatE compatible-only recommendations."""

import json
import math
from collections import defaultdict

from src.recommenders.contract import DATA, ROOT, sha
from src.recommenders.rq2_constraint_midlayer import OUTPUT, rows


def verify():
    manifest = json.loads((OUTPUT / "ranking_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rq2_compatible_only_rankings_complete"
    assert manifest["rankings_sha256"] == sha(OUTPUT / "primary_rotate_compatible_top10.csv")
    assert manifest["midlayer_manifest_sha256"] == sha(OUTPUT / "midlayer_manifest.json")
    assert manifest["model_sha256"] == sha(ROOT / "results/rotate_primary_full/model.pt")
    profiles = {row["user_id"]: int(row["profile_number"])
                for row in rows(OUTPUT / "profile_assignments.csv")}
    status = defaultdict(dict)
    for row in rows(OUTPUT / "recipe_profile_compatibility.csv"):
        status[int(row["profile_number"])][row["recipe_id"]] = row["status"]
    train = defaultdict(set)
    for row in rows(DATA / "reviews.csv"):
        if row["split"] == "train" and row["user_id"] in profiles:
            train[row["user_id"]].add(row["recipe_id"])
    recommend = defaultdict(list)
    for row in rows(OUTPUT / "primary_rotate_compatible_top10.csv"):
        recommend[row["user_id"]].append(row)
    assert len(recommend) == 40
    for user, entries in recommend.items():
        assert user in profiles and len(entries) == 10 and len(train[user]) == 28
        assert [int(row["rank"]) for row in entries] == list(range(1, 11))
        recipes = [row["recipe_id"] for row in entries]
        assert len(set(recipes)) == 10 and not set(recipes) & train[user]
        assert all(status[profiles[user]][recipe] == "COMPATIBLE" for recipe in recipes)
        assert all(row["score_source"] == "score" for row in entries)
        scores = [float(row["preference_score"]) for row in entries]
        assert all(math.isfinite(value) for value in scores)
        assert all(left >= right for left, right in zip(scores, scores[1:]))
    return {"status": "rq2_rankings_verified_test_blind", "users": 40, "rows": 400,
            "noncompatible_recommendations": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
