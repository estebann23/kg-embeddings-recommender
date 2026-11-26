"""Fixed FoodOn taxonomy RotatE test ablation against saved primary RotatE."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.recommenders.contract import DATA, ROOT, ROTATE, sha, verify_lock
from src.recommenders.item_cf import load_recipes, read_reviews, require
from src.recommenders.rotate import load_factory
from src.recommenders.rq1_accuracy_evaluate import METRICS, metric_values


OUTPUT = ROOT / "results/rq1_rotate_taxonomy_test"
PRIOR = ROOT / "results/rq1_accuracy_test"
PRIMARY = ROOT / "results/rotate_primary_full"
HIERARCHY = ROOT / "results/rotate_hierarchy_full"
ADDENDUM = ROOT / "config/rq1_rotate_taxonomy_test_addendum.json"
PRIOR_ADDENDUM = ROOT / "config/rq1_accuracy_test_addendum.json"
CONDITIONS = ("4_rotate_primary", "4_rotate_hierarchy")
RANK_HEADER = ("generated_utc", "condition", "user_id", "rank", "recipe_id", "score",
               "score_source", "scorable_candidates")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def preflight(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = read_json(ADDENDUM)
    require(config["status"] == "fixed_before_hierarchy_test_scoring" and
            config["metrics"] == list(METRICS) and config["ranking_depth"] == 50,
            "Taxonomy test addendum changed")
    from src.recommenders.verify_rotate_validation import verify as verify_rotate
    from src.recommenders.verify_rq1_accuracy_results import verify as verify_prior
    require(verify_rotate("primary")["weight_five"] == config["primary_weight_five"] == 0.5,
            "Primary selection changed")
    require(verify_rotate("hierarchy")["weight_five"] == config["hierarchy_weight_five"] == 0.75,
            "Hierarchy selection changed")
    require(verify_prior()["status"] == "accuracy_results_verified", "Prior test results changed")
    p = read_json(data / "rotate_primary_preflight.json")
    h = read_json(data / "rotate_hierarchy_preflight.json")
    p_counts = p["relation_counts"]
    h_counts = h["relation_counts"]
    require(h_counts["SUBCLASS_OF"] == 2872 and
            {key: value for key, value in h_counts.items() if key != "SUBCLASS_OF"} == p_counts and
            h["triple_count"] - p["triple_count"] == 2872 and
            h["config_sha256"] == p["config_sha256"] == sha(ROTATE) and
            h["training_users"] == p["training_users"] == 4000 and
            h["scorable_recipes"] == p["scorable_recipes"] == 74055,
            "Taxonomy triple ablation is not isolated as declared")
    p_entities = read_json(PRIMARY / "entity_to_id.json")
    h_entities = read_json(HIERARCHY / "entity_to_id.json")
    p_recipes = {entity for entity in p_entities if entity.startswith("hummus_recipe_")}
    h_recipes = {entity for entity in h_entities if entity.startswith("hummus_recipe_")}
    require(p_recipes == h_recipes and len(p_recipes) == 74055,
            "Recipe embedding vocabulary differs across variants")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "taxonomy_test_preflight_passed", "dataset_id": data.name,
        "added_relation": "SUBCLASS_OF", "added_training_triples": 2872,
        "same_embedded_recipes": len(p_recipes), "catalog_recipes": 75972,
        "primary_weight_five": 0.5, "hierarchy_weight_five": 0.75,
        "users": 4000, "ranking_depth": 50,
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "addendum_sha256": sha(ADDENDUM), "prior_addendum_sha256": sha(PRIOR_ADDENDUM),
        "primary_rank_manifest_sha256": sha(PRIOR / "4_rotate_primary_rank_manifest.json"),
        "primary_evaluation_manifest_sha256": sha(PRIOR / "evaluation_manifest.json"),
        "primary_model_sha256": sha(PRIMARY / "model.pt"),
        "hierarchy_model_sha256": sha(HIERARCHY / "model.pt"),
        "hierarchy_validation_sha256": sha(HIERARCHY / "validation_selection.json"),
        "hierarchy_training_sha256": sha(HIERARCHY / "training_manifest.json"),
        "reviews_sha256": sha(data / "reviews.csv"), "recipes_sha256": sha(data / "recipes.csv"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "preflight.json", result)
    return result


def require_preflight(data: Path, output: Path) -> dict:
    verify_lock(data)
    value = read_json(output / "preflight.json")
    require(value["status"] == "taxonomy_test_preflight_passed" and
            value["addendum_sha256"] == sha(ADDENDUM) and
            value["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            value["primary_rank_manifest_sha256"] == sha(PRIOR / "4_rotate_primary_rank_manifest.json") and
            value["primary_evaluation_manifest_sha256"] == sha(PRIOR / "evaluation_manifest.json") and
            value["primary_model_sha256"] == sha(PRIMARY / "model.pt") and
            value["hierarchy_model_sha256"] == sha(HIERARCHY / "model.pt") and
            value["hierarchy_validation_sha256"] == sha(HIERARCHY / "validation_selection.json") and
            value["hierarchy_training_sha256"] == sha(HIERARCHY / "training_manifest.json") and
            value["reviews_sha256"] == sha(data / "reviews.csv") and
            value["recipes_sha256"] == sha(data / "recipes.csv"),
            "Taxonomy test preflight seal changed")
    return value


def rank_hierarchy(data: Path = DATA, output: Path = OUTPUT) -> dict:
    import torch
    from pykeen.losses import MarginRankingLoss
    from pykeen.models import RotatE
    from scipy.stats import rankdata

    data, output = Path(data).resolve(), Path(output).resolve()
    pre = require_preflight(data, output)
    config = read_json(ROTATE)
    weight = read_json(HIERARCHY / "validation_selection.json")["selected_weight_five"]
    require(weight == pre["hierarchy_weight_five"] == 0.75, "Hierarchy weight changed")
    factory = load_factory(data, "hierarchy")
    model = RotatE(triples_factory=factory, embedding_dim=config["training"]["embedding_dim"],
                   loss=MarginRankingLoss(margin=config["training"]["margin"]),
                   random_seed=config["training"]["seed"])
    model.load_state(HIERARCHY / "model.pt")
    model.eval()
    recipes = load_recipes(data)
    embedded = [recipe for recipe in recipes if recipe in factory.entity_to_id]
    embedded_ids = torch.tensor([factory.entity_to_id[recipe] for recipe in embedded], dtype=torch.long)
    index = {recipe: position for position, recipe in enumerate(embedded)}
    unembedded = [recipe for recipe in recipes if recipe not in index]
    require(len(embedded) == 74055 and len(unembedded) == 1917, "Hierarchy recipe coverage changed")
    train, _ = read_reviews(data)
    require(len(train) == 4000, "Training users changed")
    timestamp = datetime.now(timezone.utc).isoformat()
    path = output / "4_rotate_hierarchy_top50.csv"
    temp = path.with_suffix(".csv.tmp")
    count = 0
    fallback_count = 0
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream, torch.inference_mode():
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(RANK_HEADER)
            for user in sorted(train):
                excluded = {recipe for recipe, _ in train[user]}
                eligible_mask = np.array([recipe not in excluded for recipe in embedded])
                eligible = np.flatnonzero(eligible_mask)
                require(len(eligible) == 74027, "Eligible embedded recipe count changed")
                user_index = factory.entity_to_id[user]
                percentiles = []
                for relation in ("RATED_4", "RATED_5"):
                    query = torch.tensor([[user_index, factory.relation_to_id[relation]]], dtype=torch.long)
                    raw = model.predict_t(query, tails=embedded_ids).reshape(-1).cpu().numpy()
                    require(np.isfinite(raw).all(), "Non-finite hierarchy prediction")
                    percentiles.append(rankdata(raw[eligible], method="average") / len(eligible))
                combined = (1 - weight) * percentiles[0] + weight * percentiles[1]
                cutoff = np.partition(combined, -50)[-50]
                contenders = np.flatnonzero(combined >= cutoff)
                top = contenders[np.lexsort((eligible[contenders], -combined[contenders]))[:50]]
                ranked = [(embedded[eligible[position]], float(combined[position]), "score")
                          for position in top]
                if len(ranked) < 50:
                    ranked.extend((recipe, None, "fallback") for recipe in unembedded
                                  if recipe not in excluded)
                    ranked = ranked[:50]
                require(len(ranked) == 50 and len({recipe for recipe, _, _ in ranked}) == 50,
                        "Hierarchy top-50 incomplete")
                for position, (recipe, score, source) in enumerate(ranked, 1):
                    writer.writerow((timestamp, "4_rotate_hierarchy", user, position, recipe,
                                     "" if score is None else format(score, ".17g"),
                                     source, len(eligible)))
                    fallback_count += source == "fallback"
                    count += 1
        require(count == 200000, "Hierarchy ranking row count changed")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    result = {"status": "hierarchy_test_rankings_complete", "users": len(train), "rows": count,
              "fallback_top50_rows": fallback_count, "scorable_candidates_per_user": 74027,
              "weight_five": weight, "ranking_sha256": sha(path),
              "preflight_sha256": sha(output / "preflight.json"),
              "model_sha256": sha(HIERARCHY / "model.pt"),
              "generated_utc": timestamp}
    write_json(output / "rank_manifest.json", result)
    return result


def evaluate(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    require_preflight(data, output)
    from src.recommenders.verify_rq1_rotate_taxonomy_rank import verify as verify_rank
    require(verify_rank(data, output)["status"] == "hierarchy_rankings_verified",
            "Hierarchy test-blind ranks failed")
    targets = defaultdict(dict)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "test":
            targets[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    require(len(targets) == 4000 and all(len(value) == 8 for value in targets.values()),
            "Test target cohort changed")
    require(all(any(rating >= 4 for rating in values.values()) for values in targets.values()),
            "Expected positive-test user population changed")
    per_user = defaultdict(dict)
    per_user_path = output / "per_user_accuracy.csv"
    with per_user_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("condition", "user_id", "positive_test_targets", "hits_at_10", "hits_at_50",
                         "precision_at_10", "precision_at_50", "ndcg_at_10", "ndcg_at_50"))
        for condition, path in (("4_rotate_primary", PRIOR / "4_rotate_primary_top50.csv"),
                                ("4_rotate_hierarchy", output / "4_rotate_hierarchy_top50.csv")):
            current_user, current_rows = None, []

            def commit(user, entries):
                require(len(entries) == 50, "Incomplete paired ranking")
                value = metric_values([row["recipe_id"] for row in entries], targets[user])
                per_user[user][condition] = value
                writer.writerow((condition, user, value["positive_test_targets"],
                                 value["hits_at_10"], value["hits_at_50"],
                                 *(format(value[metric], ".17g") for metric in METRICS)))

            for row in rows(path):
                if current_user is None:
                    current_user = row["user_id"]
                if row["user_id"] != current_user:
                    commit(current_user, current_rows)
                    current_user, current_rows = row["user_id"], []
                current_rows.append(row)
            commit(current_user, current_rows)
    require(len(per_user) == 4000 and all(set(value) == set(CONDITIONS) for value in per_user.values()),
            "Paired per-user results incomplete")

    aggregate_path = output / "aggregate_accuracy.csv"
    paired_path = output / "paired_differences.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("condition", "metric", "users", "mean", "sum_hits", "users_with_hit"))
        for condition in CONDITIONS:
            for metric in METRICS:
                k = 10 if metric.endswith("10") else 50
                values = [per_user[user][condition][metric] for user in sorted(per_user)]
                hits = [per_user[user][condition][f"hits_at_{k}"] for user in sorted(per_user)]
                writer.writerow((condition, metric, len(values),
                                 format(sum(values) / len(values), ".17g"),
                                 sum(hits), sum(value > 0 for value in hits)))
    with paired_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("metric", "users", "mean_hierarchy_minus_primary", "hierarchy_wins",
                         "ties", "primary_wins"))
        for metric in METRICS:
            diff = [per_user[user]["4_rotate_hierarchy"][metric] -
                    per_user[user]["4_rotate_primary"][metric] for user in sorted(per_user)]
            writer.writerow((metric, len(diff), format(sum(diff) / len(diff), ".17g"),
                             sum(value > 0 for value in diff), sum(value == 0 for value in diff),
                             sum(value < 0 for value in diff)))
    result = {"status": "taxonomy_test_accuracy_complete", "users": len(per_user),
              "conditions": list(CONDITIONS), "k": [10, 50],
              "preflight_sha256": sha(output / "preflight.json"),
              "rank_manifest_sha256": sha(output / "rank_manifest.json"),
              "prior_evaluation_manifest_sha256": sha(PRIOR / "evaluation_manifest.json"),
              "per_user_accuracy_sha256": sha(per_user_path),
              "aggregate_accuracy_sha256": sha(aggregate_path),
              "paired_differences_sha256": sha(paired_path),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "evaluation_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "rank", "evaluate"))
    args = parser.parse_args()
    action = {"preflight": preflight, "rank": rank_hierarchy, "evaluate": evaluate}[args.command]
    print(json.dumps(action(), indent=2))


if __name__ == "__main__":
    main()
