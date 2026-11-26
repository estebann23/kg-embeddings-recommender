"""Descriptive paired comparisons for the top-100 candidate exclusion test."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from itertools import combinations

from src.recommenders.contract import ROOT, sha
from src.recommenders.rq1_top100_exclusion import MODELS, OUTPUT, rows, write_json


PRIOR = ROOT / "results/rq1_accuracy_test"
TAXONOMY = ROOT / "results/rq1_rotate_taxonomy_test"
DIVERSITY = ROOT / "results/rq1_diversity_test"
METRICS = ("precision_at_10", "precision_at_50", "ndcg_at_10", "ndcg_at_50",
           "ingredient_ild_at_10", "ingredient_ild_at_50")


def save(path, header, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(values)


def compare():
    result = json.loads((OUTPUT / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert result["status"] == "reduced_catalog_evaluation_complete"
    assert result["per_user_sha256"] == sha(OUTPUT / "per_user_metrics.csv")
    current = {(row["model"], row["user_id"]): row
               for row in rows(OUTPUT / "per_user_metrics.csv")}
    assert len(current) == 16000
    prior_accuracy = {}
    for path in (PRIOR / "per_user_accuracy.csv", TAXONOMY / "per_user_accuracy.csv"):
        for row in rows(path):
            prior_accuracy[row["condition"], row["user_id"]] = row
    prior_ild = {(row["model"], row["user_id"], int(row["k"])): row
                 for row in rows(DIVERSITY / "per_user_ingredient_diversity.csv")}
    assert len(prior_ild) == 32000
    users = sorted({user for _, user in current})
    assert len(users) == 4000

    def value(table, model, user, metric):
        text = table[model, user][metric]
        return None if text == "" else float(text)

    model_pairs = []
    for metric in METRICS:
        for a, b in combinations(MODELS, 2):
            differences = []
            for user in users:
                left, right = value(current, a, user, metric), value(current, b, user, metric)
                if left is not None and right is not None:
                    differences.append(left - right)
            assert differences
            model_pairs.append((metric, a, b, len(differences),
                                format(math.fsum(differences) / len(differences), ".17g"),
                                sum(delta > 0 for delta in differences),
                                sum(delta == 0 for delta in differences),
                                sum(delta < 0 for delta in differences)))
    model_path = OUTPUT / "paired_model_differences.csv"
    save(model_path, ("metric", "model_a", "model_b", "paired_users", "mean_a_minus_b",
                      "wins_a", "ties", "wins_b"), model_pairs)

    catalog_pairs = []
    for model in MODELS:
        for metric in METRICS:
            differences = []
            for user in users:
                now = value(current, model, user, metric)
                if metric.startswith("ingredient_ild"):
                    k = int(metric.rsplit("_", 1)[-1])
                    old_text = prior_ild[model, user, k]["ingredient_ild"]
                else:
                    old_text = prior_accuracy[model, user][metric]
                old = None if old_text == "" else float(old_text)
                if now is not None and old is not None:
                    differences.append(now - old)
            assert differences
            catalog_pairs.append((model, metric, len(differences),
                                  format(math.fsum(differences) / len(differences), ".17g"),
                                  sum(delta > 0 for delta in differences),
                                  sum(delta == 0 for delta in differences),
                                  sum(delta < 0 for delta in differences)))
    catalog_path = OUTPUT / "paired_full_catalog_differences.csv"
    save(catalog_path, ("model", "metric", "paired_users", "mean_reduced_minus_full",
                        "improved", "unchanged", "worsened"), catalog_pairs)
    removed = {row["recipe_id"] for row in rows(OUTPUT / "excluded_top100_training_mean.csv")}
    prior_rank_paths = {model: (TAXONOMY if model == "4_rotate_hierarchy" else PRIOR) /
                        f"{model}_top50.csv" for model in MODELS}
    prior_exposure = []
    for model in MODELS:
        for k in (10, 50):
            removed_counts = Counter(row["recipe_id"] for row in rows(prior_rank_paths[model])
                                     if int(row["rank"]) <= k and row["recipe_id"] in removed)
            prior_exposure.append((model, k, sum(removed_counts.values()),
                                   len(removed_counts)))
    prior_exposure_path = OUTPUT / "removed_recipes_in_prior_lists.csv"
    save(prior_exposure_path,
         ("model", "k", "removed_recipe_exposure_positions", "distinct_removed_recipes_exposed"),
         prior_exposure)
    output = {"status": "paired_comparisons_complete", "model_pair_rows": len(model_pairs),
              "full_catalog_pair_rows": len(catalog_pairs),
              "prior_removed_exposure_rows": len(prior_exposure),
              "new_evaluation_manifest_sha256": sha(OUTPUT / "evaluation_manifest.json"),
              "prior_accuracy_sha256": sha(PRIOR / "per_user_accuracy.csv"),
              "prior_taxonomy_accuracy_sha256": sha(TAXONOMY / "per_user_accuracy.csv"),
              "prior_diversity_sha256": sha(DIVERSITY / "per_user_ingredient_diversity.csv"),
              "paired_model_sha256": sha(model_path),
              "paired_full_catalog_sha256": sha(catalog_path),
              "prior_removed_exposure_sha256": sha(prior_exposure_path)}
    write_json(OUTPUT / "comparison_manifest.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
