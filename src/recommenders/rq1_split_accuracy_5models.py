"""Evaluate five frozen test-blind top-50 lists on random and semantic test slices."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock

CONFIG = ROOT / "config/rq1_split_accuracy_5models.json"
OUTPUT = ROOT / "results/rq1_split_accuracy_5models"
SPLIT = ROOT / "results/rq1_test_semantic_slices"
GAIN = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}
METRICS = ("precision_at_10", "precision_at_50", "recall_at_10", "recall_at_50",
           "ndcg_at_10", "ndcg_at_50")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def selected_value(model, selection):
    if model == "interaction_only_cf":
        grid = selection["mean_validation_ndcg_at_10"]
        chosen = max(map(int, grid), key=lambda n: (grid[str(n)], -n))
        return {"neighbor_n": selection["selected_n"]}, grid, chosen == selection["selected_n"]
    if model == "weighted_graph_paths":
        grid = selection["mean_validation_ndcg_at_10"]
        chosen = max(map(float, grid), key=lambda beta: (grid[str(beta)], -beta))
        return {"beta": selection["selected_beta"]}, grid, chosen == selection["selected_beta"]
    if model == "semantic_mf":
        candidates = selection["grid_results"]
        priority = lambda r: (r["validation_mean_ndcg_at_10"], -r["validation_rmse_clipped"],
                              -r["content_scale"], r["regularization"], -r["epoch"])
        selected = selection["selected"]
        fields = ("factors", "content_scale", "regularization", "epoch")
        return {field: selected[field] for field in fields}, candidates, max(candidates, key=priority) == selected
    grid = selection["mean_ndcg_at_10_by_weight"]
    chosen = max(map(float, grid), key=lambda w: (grid[str(w)], -abs(w - 0.5), -w))
    return {"weight_five": selection["selected_weight_five"]}, grid, chosen == selection["selected_weight_five"]


def preflight(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = read_json(CONFIG)
    require(config["status"] == "frozen_before_split_test_accuracy" and
            config["dataset_id"] == data.name and config["k"] == [10, 50] and
            len(config["models"]) == 5 and len(config["populations"]) == 7,
            "Incorrect split accuracy contract")
    split = read_json(SPLIT / "manifest.json")
    require(split["users"] == 4000 and split["random_rows"] == split["semantic_rows"] == 16000 and
            split["files"]["assignments"]["sha256"] == sha(ROOT / config["split_assignments"]),
            "Split assignment changed")
    validation = {}
    ranking_hashes = {}
    for model, spec in config["models"].items():
        selection = read_json(ROOT / spec["validation"])
        require(selection["status"] == "validation_selected_no_test_access", f"{model} not selected")
        actual, grid, is_best = selected_value(model, selection)
        require(is_best and actual == spec["selected"], f"{model} validation choice changed")
        validation[model] = {"selection_sha256": sha(ROOT / spec["validation"]),
                             "selected": actual, "grid": grid}
        ranking_path = ROOT / spec["ranking"]
        require(ranking_path.is_file(), f"{model} top-50 file missing")
        ranking_hashes[model] = sha(ranking_path)
    # Existing ranking manifests independently fix these four outputs. Semantic MF
    # has its own validation manifest, which also hashes its test-blind ranking.
    old = ROOT / "results/rq1_accuracy_test"
    for model, condition in (("interaction_only_cf", "1_item_cf"),
                             ("weighted_graph_paths", "3_explicit_paths"),
                             ("rotate_primary", "4_rotate_primary")):
        require(read_json(old / f"{condition}_rank_manifest.json")["ranking_sha256"] ==
                ranking_hashes[model], f"{model} ranking manifest changed")
    require(read_json(ROOT / "results/rq1_rotate_taxonomy_test/rank_manifest.json")
            ["ranking_sha256"] == ranking_hashes["rotate_foodon_taxonomy"],
            "Taxonomy ranking manifest changed")
    require(read_json(ROOT / config["models"]["semantic_mf"]["validation"])
            ["test_blind_top50_sha256"] == ranking_hashes["semantic_mf"],
            "Semantic MF ranking manifest changed")
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_bytes(CONFIG.read_bytes())
    with (output / "validation_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "parameter_setting", "validation_mean_ndcg_at_10", "selected"))
        for model, info in validation.items():
            if model == "semantic_mf":
                for record in info["grid"]:
                    parameters = {k: record[k] for k in ("factors", "content_scale",
                                                           "regularization", "epoch")}
                    writer.writerow((model, json.dumps(parameters, sort_keys=True),
                                     format(record["validation_mean_ndcg_at_10"], ".17g"),
                                     int(parameters == info["selected"])))
            else:
                param = next(iter(info["selected"]))
                for value, ndcg in info["grid"].items():
                    writer.writerow((model, json.dumps({param: float(value) if "." in value
                                                     else int(value)}, sort_keys=True),
                                     format(ndcg, ".17g"),
                                     int(float(value) == float(info["selected"][param]))))
    result = {"status": "preflight_passed_before_split_test_metrics",
              "dataset_id": data.name, "models": list(config["models"]),
              "populations": config["populations"], "validation": validation,
              "config_sha256": sha(CONFIG), "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
              "reviews_sha256": sha(data / "reviews.csv"), "recipes_sha256": sha(data / "recipes.csv"),
              "split_manifest_sha256": sha(SPLIT / "manifest.json"),
              "split_assignments_sha256": sha(ROOT / config["split_assignments"]),
              "ranking_sha256": ranking_hashes,
              "validation_grid_sha256": sha(output / "validation_grid.csv"),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    (output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def metric_values(ranking, targets):
    positive = sum(rating >= 4 for rating in targets.values())
    ideal = sorted((GAIN[rating] for rating in targets.values() if rating >= 4), reverse=True)
    result = {"target_count": len(targets), "positive_targets": positive}
    for k in (10, 50):
        top = ranking[:k]
        hits = sum(targets.get(recipe, 0) >= 4 for recipe in top)
        dcg = sum(GAIN[targets[recipe]] / math.log2(position + 2)
                  for position, recipe in enumerate(top) if recipe in targets)
        idcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(ideal[:k]))
        result[f"hits_at_{k}"] = hits
        result[f"precision_at_{k}"] = hits / k
        result[f"recall_at_{k}"] = hits / positive if positive else None
        result[f"ndcg_at_{k}"] = dcg / idcg if idcg else None
    return result


def evaluate(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    config = read_json(CONFIG)
    seal = read_json(output / "preflight.json")
    require(seal["status"] == "preflight_passed_before_split_test_metrics" and
            seal["config_sha256"] == sha(CONFIG) and
            seal["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            seal["reviews_sha256"] == sha(data / "reviews.csv") and
            seal["recipes_sha256"] == sha(data / "recipes.csv") and
            seal["split_manifest_sha256"] == sha(SPLIT / "manifest.json") and
            seal["split_assignments_sha256"] == sha(ROOT / config["split_assignments"]) and
            seal["validation_grid_sha256"] == sha(output / "validation_grid.csv"),
            "Split test preflight seal changed")
    for model, spec in config["models"].items():
        require(seal["validation"][model]["selection_sha256"] == sha(ROOT / spec["validation"]) and
                seal["ranking_sha256"][model] == sha(ROOT / spec["ranking"]),
                f"{model} selection or ranking changed")
    train = defaultdict(set)
    source_test = {}
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train":
            train[row["user_id"]].add(row["recipe_id"])
        elif row["split"] == "test":
            source_test[row["review_id"]] = row
    require(len(train) == 4000 and len(source_test) == 32000,
            "Source cohort changed")
    targets = {population: defaultdict(dict) for population in config["populations"]}
    seen_reviews = set()
    for row in rows(ROOT / config["split_assignments"]):
        review = row["review_id"]
        require(review in source_test and review not in seen_reviews,
                "Duplicate or non-test split assignment")
        seen_reviews.add(review)
        source = source_test[review]
        require(all(row[key] == source[key] for key in ("user_id", "recipe_id", "rating", "split")),
                "Split assignment differs from source test rating")
        population = row["test_half"]
        require(population in {"random_50", "semantic_50"}, "Unknown split half")
        destinations = [population]
        if population == "semantic_50":
            require(row["semantic_bin"] in targets, "Unknown semantic bin")
            destinations.append(row["semantic_bin"])
        else:
            require(row["semantic_bin"] == "", "Random review has semantic bin")
        for name in destinations:
            user, recipe = row["user_id"], row["recipe_id"]
            require(recipe not in train[user] and recipe not in targets[name][user],
                    "Held-out target duplicate or in training")
            targets[name][user][recipe] = int(row["rating"])
    require(seen_reviews == set(source_test), "Missing test review assignment")
    require(sum(map(len, targets["random_50"].values())) == 16000 and
            sum(map(len, targets["semantic_50"].values())) == 16000 and
            all(len(targets[half][user]) == 4 for half in ("random_50", "semantic_50")
                for user in train), "Incorrect per-user half sizes")
    require(sum(sum(map(len, targets[name].values())) for name in config["populations"][2:]) == 16000,
            "Semantic bins do not partition the half")
    catalog = {row["recipe_id"] for row in rows(data / "recipes.csv")}
    require(len(catalog) == 75972, "Candidate universe changed")
    per_user_path = output / "per_user_accuracy.csv"
    aggregate_path = output / "aggregate_accuracy.csv"
    aggregate = []
    with per_user_path.open("w", encoding="utf-8", newline="") as per_stream:
        writer = csv.writer(per_stream, lineterminator="\n")
        writer.writerow(("population", "model", "user_id", "targets", "positive_targets",
                         "hits_at_10", "hits_at_50", *METRICS))
        for model, spec in config["models"].items():
            saved = defaultdict(list)
            for row in rows(ROOT / spec["ranking"]):
                user, recipe = row["user_id"], row["recipe_id"]
                require(user in train and recipe in catalog and recipe not in train[user],
                        f"{model} ranking eligibility changed")
                require(int(row["rank"]) == len(saved[user]) + 1 and len(saved[user]) < 50,
                        f"{model} ranking order changed")
                saved[user].append(recipe)
            require(len(saved) == 4000 and all(len(saved[user]) == 50 and
                                               len(set(saved[user])) == 50 for user in train),
                    f"{model} ranking incomplete")
            for population in config["populations"]:
                sums = Counter()
                n_users = n_positive_users = n_targets = n_positive_targets = 0
                for user in sorted(targets[population]):
                    value = metric_values(saved[user], targets[population][user])
                    n_users += 1
                    n_targets += value["target_count"]
                    n_positive_targets += value["positive_targets"]
                    if value["positive_targets"]:
                        n_positive_users += 1
                    writer.writerow((population, model, user, value["target_count"],
                                     value["positive_targets"], value["hits_at_10"],
                                     value["hits_at_50"],
                                     *("" if value[name] is None else format(value[name], ".17g")
                                       for name in METRICS)))
                    for name in METRICS:
                        if value[name] is not None:
                            sums[name] += value[name]
                    sums["hits_at_10"] += value["hits_at_10"]
                    sums["hits_at_50"] += value["hits_at_50"]
                require(n_users > 0 and n_positive_users > 0, "Empty evaluation population")
                record = {"population": population, "model": model, "users_with_targets": n_users,
                          "users_with_positive_targets": n_positive_users,
                          "targets": n_targets, "positive_targets": n_positive_targets,
                          "hits_at_10": int(sums["hits_at_10"]),
                          "hits_at_50": int(sums["hits_at_50"])}
                for name in METRICS:
                    denominator = n_users if name.startswith("precision") else n_positive_users
                    record[name] = sums[name] / denominator
                aggregate.append(record)
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ("population", "model", "users_with_targets", "users_with_positive_targets",
                  "targets", "positive_targets", "hits_at_10", "hits_at_50", *METRICS)
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in aggregate:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value
                             for key, value in record.items()})
    result = {"status": "split_test_accuracy_complete", "models": list(config["models"]),
              "populations": config["populations"], "k": config["k"],
              "source_test_reviews": len(source_test), "aggregate_rows": len(aggregate),
              "per_user_rows": sum(len(targets[name]) for name in config["populations"]) * 5,
              "preflight_sha256": sha(output / "preflight.json"),
              "aggregate_sha256": sha(aggregate_path),
              "per_user_sha256": sha(per_user_path),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    (output / "evaluation_manifest.json").write_text(json.dumps(result, indent=2) + "\n",
                                                     encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "evaluate"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(preflight(args.data, args.output) if args.stage == "preflight"
                     else evaluate(args.data, args.output), indent=2))
