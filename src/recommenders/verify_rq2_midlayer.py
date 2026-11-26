"""Independent source, classification, and candidate-coverage checks for RQ2."""

import csv
import json
from collections import Counter, defaultdict

from src.recommenders.contract import DATA, ROOT, sha
from src.recommenders.rq2_constraint_midlayer import OUTPUT, rows


def verify():
    manifest = json.loads((OUTPUT / "midlayer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rq2_kg_compatibility_complete"
    assert manifest["classification_sha256"] == sha(OUTPUT / "recipe_profile_compatibility.csv")
    assert manifest["candidate_coverage_sha256"] == sha(OUTPUT / "candidate_coverage_by_user.csv")
    assert manifest["preflight_sha256"] == sha(OUTPUT / "midlayer_preflight.json")
    names = {row["id"]: row["name"] for row in rows(DATA / "graph_nodes.csv")
             if row["label"] == "Ingredient"}
    edges = defaultdict(set)
    for row in rows(DATA / "graph_edges.csv"):
        if row["relation"] == "CONTAINS_MAPPED_INGREDIENT":
            edges[row["source_id"]].add(row["target_id"])
    assert sum(map(len, edges.values())) == 586292
    assignments = {row["user_id"]: int(row["profile_number"])
                   for row in rows(OUTPUT / "profile_assignments.csv")}
    assert len(assignments) == 40 and Counter(assignments.values()) == {i: 8 for i in range(1, 6)}
    counts = defaultdict(Counter)
    statuses = defaultdict(dict)
    checked = 0
    for row in rows(OUTPUT / "recipe_profile_compatibility.csv"):
        p, recipe, status = int(row["profile_number"]), row["recipe_id"], row["status"]
        assert status in {"COMPATIBLE", "INCOMPATIBLE", "UNKNOWN"}
        assert recipe not in statuses[p]
        statuses[p][recipe] = status
        counts[p][status] += 1
        assert int(row["graph_ingredient_count"]) == len(edges.get(recipe, ()))
        if status == "COMPATIBLE":
            assert row["reconciled_list"] == "True" and not row["conflict_ingredient_id"]
            assert not row["ambiguous_ingredient_name"]
        elif status == "INCOMPATIBLE":
            ingredient = row["conflict_ingredient_id"]
            assert ingredient in edges[recipe]
            assert names[ingredient] == row["conflict_ingredient_name"]
            assert row["conflict_rule_id"]
        else:
            assert not row["conflict_ingredient_id"]
        checked += 1
    assert checked == 5 * 75972
    assert all(len(statuses[p]) == 75972 for p in range(1, 6))
    assert {str(p): dict(counts[p]) for p in range(1, 6)} == manifest["profile_status_counts"]
    # Reference labels were recorded from source text and reconciliation evidence,
    # separately from the classifier output.
    cases = list(rows(ROOT / "config/rq2_reference_cases.csv"))
    assert len(cases) == 13
    for case in cases:
        assert statuses[int(case["profile_number"])][case["recipe_id"]] == case["expected_status"]
    train = defaultdict(set)
    for row in rows(DATA / "reviews.csv"):
        if row["split"] == "train" and row["user_id"] in assignments:
            train[row["user_id"]].add(row["recipe_id"])
    cover = list(rows(OUTPUT / "candidate_coverage_by_user.csv"))
    assert len(cover) == 40
    for row in cover:
        user, p = row["user_id"], int(row["profile_number"])
        assert assignments[user] == p and len(train[user]) == 28
        expected = counts[p].copy()
        for recipe in train[user]:
            expected[statuses[p][recipe]] -= 1
        assert int(row["base_candidates"]) == 75944
        for status, column in (("COMPATIBLE", "compatible_candidates"),
                               ("INCOMPATIBLE", "incompatible_candidates"),
                               ("UNKNOWN", "unknown_candidates")):
            assert int(row[column]) == expected[status]
        assert int(row["allow_unknown_candidates"]) == expected["COMPATIBLE"] + expected["UNKNOWN"]
    return {"status": "rq2_midlayer_verified", "classification_rows": checked,
            "users": len(assignments), "reference_cases": len(cases),
            "candidate_rows": len(cover)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
