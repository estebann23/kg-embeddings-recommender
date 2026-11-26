"""Apply primary RotatE only after RQ2 KG compatibility filtering."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from src.recommenders.contract import DATA, ROOT, ROTATE, sha
from src.recommenders.item_cf import load_recipes, read_reviews, require
from src.recommenders.rotate import load_factory
from src.recommenders.rq2_constraint_midlayer import OUTPUT, rows, write_json
from src.recommenders.verify_rq2_midlayer import verify as verify_midlayer


PRIMARY = ROOT / "results/rotate_primary_full"


def rank():
    require(verify_midlayer()["status"] == "rq2_midlayer_verified",
            "KG compatibility gate failed")
    import torch
    from pykeen.losses import MarginRankingLoss
    from pykeen.models import RotatE
    from scipy.stats import rankdata

    assignments = {row["user_id"]: int(row["profile_number"])
                   for row in rows(OUTPUT / "profile_assignments.csv")}
    compatible = defaultdict(set)
    for row in rows(OUTPUT / "recipe_profile_compatibility.csv"):
        if row["status"] == "COMPATIBLE":
            compatible[int(row["profile_number"])].add(row["recipe_id"])
    recipes = load_recipes(DATA)
    titles = {row["recipe_id"]: row["title"] for row in rows(DATA / "recipes.csv")}
    train, _ = read_reviews(DATA)
    selection = json.loads((PRIMARY / "validation_selection.json").read_text(encoding="utf-8"))
    weight = selection["selected_weight_five"]
    require(weight == 0.5, "Primary RotatE fusion weight changed")
    factory = load_factory(DATA, "primary")
    config = json.loads(ROTATE.read_text(encoding="utf-8"))
    model = RotatE(triples_factory=factory,
                   embedding_dim=config["training"]["embedding_dim"],
                   loss=MarginRankingLoss(margin=config["training"]["margin"]),
                   random_seed=config["training"]["seed"])
    model.load_state(PRIMARY / "model.pt")
    model.eval()
    timestamp = datetime.now(timezone.utc).isoformat()
    path = OUTPUT / "primary_rotate_compatible_top10.csv"
    counts = Counter()
    with path.open("w", encoding="utf-8", newline="") as stream, torch.inference_mode():
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("generated_utc", "user_id", "profile_number", "rank", "recipe_id",
                         "title", "compatibility_status", "preference_score", "score_source"))
        for user in sorted(assignments):
            profile = assignments[user]
            excluded = {recipe for recipe, _ in train[user]}
            eligible = [recipe for recipe in recipes if recipe in compatible[profile]
                        and recipe not in excluded]
            embedded = [recipe for recipe in eligible if recipe in factory.entity_to_id]
            unembedded = [recipe for recipe in eligible if recipe not in factory.entity_to_id]
            require(len(eligible) >= 10 and len(embedded) >= 10, "Insufficient compatible candidates")
            tails = torch.tensor([factory.entity_to_id[recipe] for recipe in embedded],
                                 dtype=torch.long)
            percentiles = []
            for relation in ("RATED_4", "RATED_5"):
                query = torch.tensor([[factory.entity_to_id[user],
                                       factory.relation_to_id[relation]]], dtype=torch.long)
                raw = model.predict_t(query, tails=tails).reshape(-1).cpu().numpy()
                require(np.isfinite(raw).all(), "Non-finite compatible RotatE score")
                percentiles.append(rankdata(raw, method="average") / len(raw))
            combined = (1 - weight) * percentiles[0] + weight * percentiles[1]
            order = np.lexsort((np.arange(len(embedded)), -combined))[:10]
            ranked = [(embedded[index], float(combined[index]), "score") for index in order]
            if len(ranked) < 10:
                ranked.extend((recipe, None, "fallback") for recipe in unembedded[:10 - len(ranked)])
            require(len(ranked) == 10 and len({item[0] for item in ranked}) == 10,
                    "Incomplete compatible top-10 list")
            for position, (recipe, score, source) in enumerate(ranked, 1):
                writer.writerow((timestamp, user, profile, position, recipe, titles[recipe],
                                 "COMPATIBLE", "" if score is None else format(score, ".17g"), source))
                counts[source] += 1
    result = {"status": "rq2_compatible_only_rankings_complete", "users": len(assignments),
              "rows": sum(counts.values()), "scored_rows": counts["score"],
              "fallback_rows": counts["fallback"], "rankings_sha256": sha(path),
              "midlayer_manifest_sha256": sha(OUTPUT / "midlayer_manifest.json"),
              "model_sha256": sha(PRIMARY / "model.pt"),
              "validation_sha256": sha(PRIMARY / "validation_selection.json"),
              "created_utc": timestamp}
    require(result["rows"] == 400, "Missing user recommendations")
    write_json(OUTPUT / "ranking_manifest.json", result)
    return result


def evaluate():
    manifest = json.loads((OUTPUT / "ranking_manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "rq2_compatible_only_rankings_complete" and
            manifest["rankings_sha256"] == sha(OUTPUT / "primary_rotate_compatible_top10.csv") and
            manifest["midlayer_manifest_sha256"] == sha(OUTPUT / "midlayer_manifest.json"),
            "RQ2 ranking seal changed")
    assignments = {row["user_id"]: int(row["profile_number"])
                   for row in rows(OUTPUT / "profile_assignments.csv")}
    coverage = {row["user_id"]: row for row in rows(OUTPUT / "candidate_coverage_by_user.csv")}
    targets = defaultdict(dict)
    for row in rows(DATA / "reviews.csv"):
        if row["split"] == "test" and row["user_id"] in assignments:
            targets[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    recommend = defaultdict(list)
    for row in rows(OUTPUT / "primary_rotate_compatible_top10.csv"):
        recommend[row["user_id"]].append(row)
    require(set(recommend) == set(assignments), "Recommendation cohort changed")
    path = OUTPUT / "per_user_coverage_and_relevance.csv"
    total_hits = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "profile_number", "base_candidates", "compatible_candidates",
                         "incompatible_candidates", "unknown_candidates", "allow_unknown_candidates",
                         "compatible_candidate_coverage", "allow_unknown_candidate_coverage",
                         "compatible_top10_slots_filled", "compatible_top10_coverage",
                         "retained_positive_test_targets", "observed_positive_hits_at_10",
                         "conditional_precision_at_10"))
        for user in sorted(assignments):
            entries = recommend[user]
            require(len(entries) == 10 and
                    [int(row["rank"]) for row in entries] == list(range(1, 11)) and
                    all(row["compatibility_status"] == "COMPATIBLE" for row in entries),
                    "Noncompatible or incomplete list")
            c = coverage[user]
            hits = sum(targets[user].get(row["recipe_id"], 0) >= 4 for row in entries)
            total_hits += hits
            writer.writerow((user, assignments[user], c["base_candidates"],
                             c["compatible_candidates"], c["incompatible_candidates"],
                             c["unknown_candidates"], c["allow_unknown_candidates"],
                             c["compatible_fraction"], c["allow_unknown_fraction"],
                             len(entries), len(entries) / 10,
                             sum(rating >= 4 for rating in targets[user].values()), hits,
                             format(hits / 10, ".17g")))
    result = {"status": "rq2_compatible_only_evaluation_complete", "users": len(assignments),
              "users_with_full_top10": sum(len(entries) == 10 for entries in recommend.values()),
              "recommended_positions": sum(map(len, recommend.values())),
              "incompatible_or_unknown_recommendations": 0,
              "observed_positive_test_hits_at_10": total_hits,
              "mean_conditional_precision_at_10": total_hits / 400,
              "per_user_sha256": sha(path),
              "ranking_manifest_sha256": sha(OUTPUT / "ranking_manifest.json")}
    write_json(OUTPUT / "evaluation_manifest.json", result)
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(rank() if sys.argv[1:] == ["rank"] else evaluate(), indent=2))
