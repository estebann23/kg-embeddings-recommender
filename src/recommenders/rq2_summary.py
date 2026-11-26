"""Save and independently check RQ2 profile coverage and test-label eligibility."""

import csv
import json
from collections import Counter, defaultdict

from src.recommenders.contract import DATA, sha
from src.recommenders.rq2_constraint_midlayer import OUTPUT, rows, write_json
from src.recommenders.verify_rq2_midlayer import verify as verify_midlayer
from src.recommenders.verify_rq2_rank import verify as verify_rank


def run():
    assert verify_midlayer()["status"] == "rq2_midlayer_verified"
    assert verify_rank()["status"] == "rq2_rankings_verified_test_blind"
    evaluation = json.loads((OUTPUT / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert evaluation["per_user_sha256"] == sha(OUTPUT / "per_user_coverage_and_relevance.csv")
    per_user = list(rows(OUTPUT / "per_user_coverage_and_relevance.csv"))
    assert len(per_user) == 40
    assignments = {row["user_id"]: int(row["profile_number"])
                   for row in rows(OUTPUT / "profile_assignments.csv")}
    statuses = defaultdict(dict)
    for row in rows(OUTPUT / "recipe_profile_compatibility.csv"):
        statuses[int(row["profile_number"])][row["recipe_id"]] = row["status"]
    target_counts = defaultdict(Counter)
    user_targets = defaultdict(dict)
    for row in rows(DATA / "reviews.csv"):
        if row["split"] == "test" and row["user_id"] in assignments:
            user, recipe, rating = row["user_id"], row["recipe_id"], int(row["rating"])
            p = assignments[user]
            status = statuses[p][recipe]
            target_counts[p, status]["all"] += 1
            target_counts[p, status]["positive"] += rating >= 4
            user_targets[user][recipe] = rating
    assert sum(counts["all"] for counts in target_counts.values()) == 320
    recommendations = defaultdict(list)
    for row in rows(OUTPUT / "primary_rotate_compatible_top10.csv"):
        recommendations[row["user_id"]].append(row["recipe_id"])
    hits = 0
    for row in per_user:
        user, p = row["user_id"], int(row["profile_number"])
        assert assignments[user] == p and len(user_targets[user]) == 8
        assert len(recommendations[user]) == int(row["compatible_top10_slots_filled"]) == 10
        assert all(statuses[p][recipe] == "COMPATIBLE" for recipe in recommendations[user])
        actual = sum(user_targets[user].get(recipe, 0) >= 4 for recipe in recommendations[user])
        assert int(row["observed_positive_hits_at_10"]) == actual
        assert int(row["retained_positive_test_targets"]) == sum(
            rating >= 4 for rating in user_targets[user].values())
        hits += actual
    assert hits == evaluation["observed_positive_test_hits_at_10"]
    summary_path = OUTPUT / "aggregate_coverage_by_profile.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("profile_number", "users", "mean_compatible_candidates",
                         "mean_incompatible_candidates", "mean_unknown_candidates",
                         "mean_compatible_candidate_coverage", "mean_allow_unknown_candidate_coverage",
                         "users_with_full_compatible_top10", "compatible_top10_positions",
                         "observed_positive_test_hits"))
        for p in range(1, 6):
            group = [row for row in per_user if int(row["profile_number"]) == p]
            assert len(group) == 8
            mean = lambda name: sum(float(row[name]) for row in group) / 8
            writer.writerow((p, 8, *(format(mean(name), ".17g") for name in
                                     ("compatible_candidates", "incompatible_candidates",
                                      "unknown_candidates", "compatible_candidate_coverage",
                                      "allow_unknown_candidate_coverage")),
                             sum(int(row["compatible_top10_slots_filled"]) == 10 for row in group),
                             sum(int(row["compatible_top10_slots_filled"]) for row in group),
                             sum(int(row["observed_positive_hits_at_10"]) for row in group)))
    targets_path = OUTPUT / "test_target_eligibility.csv"
    with targets_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("profile_number", "status", "test_targets", "positive_test_targets"))
        for p in range(1, 6):
            for status in ("COMPATIBLE", "INCOMPATIBLE", "UNKNOWN"):
                c = target_counts[p, status]
                writer.writerow((p, status, c["all"], c["positive"]))
    result = {"status": "rq2_summary_verified", "users": 40,
              "test_targets": 320, "positive_test_targets": sum(
                  c["positive"] for c in target_counts.values()),
              "compatible_positive_test_targets": sum(
                  target_counts[p, "COMPATIBLE"]["positive"] for p in range(1, 6)),
              "observed_positive_hits_at_10": hits,
              "aggregate_sha256": sha(summary_path), "target_eligibility_sha256": sha(targets_path),
              "evaluation_manifest_sha256": sha(OUTPUT / "evaluation_manifest.json")}
    write_json(OUTPUT / "summary_manifest.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
