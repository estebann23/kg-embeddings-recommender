"""Frozen-model RQ1 sensitivity test excluding 100 training-top-rated recipes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

from src.recommenders.contract import DATA, ROOT, ROTATE, sha, verify_lock
from src.recommenders.explicit_paths import (OUTPUT as PATH_OUTPUT, POSITIVE_GAIN,
                                             load_fit_paths, rank, weighted_jaccard_profile)
from src.recommenders.item_cf import (OUTPUT as CF_OUTPUT, load_fit, load_recipes,
                                      rank_for_n, read_reviews, require, score_candidates)
from src.recommenders.rotate import load_factory
from src.recommenders.rotate_ranking import recipe_order
from src.recommenders.rq1_accuracy_evaluate import metric_values
from src.recommenders.rq1_diversity import gini, list_ild, load_ingredients


OUTPUT = ROOT / "results/rq1_top100_exclusion_test"
CONFIG = ROOT / "config/rq1_top100_exclusion_experiment.json"
PRIMARY = ROOT / "results/rotate_primary_full"
HIERARCHY = ROOT / "results/rotate_hierarchy_full"
MODELS = ("1_item_cf", "3_explicit_paths", "4_rotate_primary", "4_rotate_hierarchy")
HEADER = ("generated_utc", "condition", "user_id", "rank", "recipe_id", "score",
          "score_source", "scorable_candidates")
K_VALUES = (10, 50)


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def selected_recipes(data: Path = DATA) -> list[tuple[str, int, int]]:
    """Return (recipe, training rating sum, training count), with exact-ratio sorting."""
    train, _ = read_reviews(data)
    counts = Counter()
    sums = Counter()
    for history in train.values():
        for recipe, rating in history:
            counts[recipe] += 1
            sums[recipe] += rating
    require(sum(counts.values()) == 112000 and len(train) == 4000,
            "Unexpected training-only rating population")
    chosen = sorted(counts, key=lambda recipe: (-Fraction(sums[recipe], counts[recipe]),
                                                 -counts[recipe], recipe_order(recipe)))[:100]
    require(len(chosen) == 100 and len(set(chosen)) == 100, "Invalid top-100 selection")
    return [(recipe, sums[recipe], counts[recipe]) for recipe in chosen]


def preflight(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = read_json(CONFIG)
    require(config["status"] == "fixed_before_top100_selection_and_reduced_ranking" and
            config["models"] == list(MODELS) and config["k"] == list(K_VALUES) and
            config["removed_recipes"] == 100, "Top-100 protocol changed")
    recipes = load_recipes(data)
    require(len(recipes) == 75972, "Catalog changed")
    from src.recommenders.verify_rq1_accuracy_rank import verify as verify_primary_ranks
    from src.recommenders.verify_rq1_rotate_taxonomy_rank import verify as verify_hierarchy_ranks
    require(verify_primary_ranks()["status"] == "test_blind_rankings_verified" and
            verify_hierarchy_ranks()["status"] == "hierarchy_rankings_verified",
            "Saved model ranking evidence changed")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "excluded_top100_training_mean.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("selection_rank", "recipe_id", "training_rating_sum",
                         "training_rating_count", "training_mean_stars"))
        for position, (recipe, total, count) in enumerate(selected_recipes(data), 1):
            writer.writerow((position, recipe, total, count, format(total / count, ".17g")))
    selected = {row["recipe_id"] for row in rows(path)}
    require(len(selected) == 100 and selected <= set(recipes), "Selection outside catalog")
    result = {
        "status": "training_only_selection_passed", "users": 4000,
        "original_catalog": len(recipes), "reduced_catalog": len(recipes) - len(selected),
        "selected_count": len(selected), "selection_sha256": sha(path),
        "config_sha256": sha(CONFIG), "reviews_sha256": sha(data / "reviews.csv"),
        "recipes_sha256": sha(data / "recipes.csv"),
        "ingredient_source_sha256": sha(data / "recipe_ingredients.csv"),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "model_sha256": {"4_rotate_primary": sha(PRIMARY / "model.pt"),
                          "4_rotate_hierarchy": sha(HIERARCHY / "model.pt")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "preflight.json", result)
    return result


def require_preflight(data: Path = DATA, output: Path = OUTPUT) -> set[str]:
    verify_lock(data)
    pre = read_json(output / "preflight.json")
    require(pre["status"] == "training_only_selection_passed" and
            pre["selection_sha256"] == sha(output / "excluded_top100_training_mean.csv") and
            pre["config_sha256"] == sha(CONFIG) and
            pre["reviews_sha256"] == sha(data / "reviews.csv") and
            pre["recipes_sha256"] == sha(data / "recipes.csv") and
            pre["ingredient_source_sha256"] == sha(data / "recipe_ingredients.csv") and
            pre["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            pre["model_sha256"]["4_rotate_primary"] == sha(PRIMARY / "model.pt") and
            pre["model_sha256"]["4_rotate_hierarchy"] == sha(HIERARCHY / "model.pt"),
            "Top-100 preflight seal changed")
    chosen = list(rows(output / "excluded_top100_training_mean.csv"))
    require(len(chosen) == 100 and
            [(row["recipe_id"], int(row["training_rating_sum"]),
              int(row["training_rating_count"])) for row in chosen] == selected_recipes(data),
            "Training-only top-100 selection changed")
    return {row["recipe_id"] for row in chosen}


def write_rankings(model: str, generated, data: Path = DATA, output: Path = OUTPUT) -> dict:
    removed = require_preflight(data, output)
    path = output / f"{model}_top50.csv"
    temp = path.with_suffix(".csv.tmp")
    timestamp = datetime.now(timezone.utc).isoformat()
    count = users = fallback = 0
    min_scorable, max_scorable = 75972, 0
    train, _ = read_reviews(data)
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(HEADER)
            for user, ranked, scorable in generated:
                require(user in train and len(ranked) == 50 and
                        len({recipe for recipe, _, _ in ranked}) == 50 and
                        not ({recipe for recipe, _, _ in ranked} & removed) and
                        not ({recipe for recipe, _, _ in ranked} &
                             {recipe for recipe, _ in train[user]}), "Invalid reduced ranking")
                for position, (recipe, score, source) in enumerate(ranked, 1):
                    writer.writerow((timestamp, model, user, position, recipe,
                                     "" if score is None else format(score, ".17g"),
                                     source, scorable))
                    count += 1
                    fallback += source == "fallback"
                users += 1
                min_scorable = min(min_scorable, scorable)
                max_scorable = max(max_scorable, scorable)
        require(users == 4000 and count == 200000, "Incomplete reduced rankings")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    result = {"status": "reduced_catalog_ranking_complete", "model": model,
              "users": users, "rows": count, "fallback_rows": fallback,
              "min_scorable_candidates": min_scorable,
              "max_scorable_candidates": max_scorable,
              "ranking_sha256": sha(path),
              "preflight_sha256": sha(output / "preflight.json"),
              "created_utc": timestamp}
    write_json(output / f"{model}_rank_manifest.json", result)
    return result


def rank_cf(data: Path = DATA, output: Path = OUTPUT) -> dict:
    removed = require_preflight(data, output)
    _, adjacency, popularity = load_fit(data, CF_OUTPUT)
    fallback = sorted((recipe for recipe in load_recipes(data) if recipe not in removed),
                      key=lambda recipe: (-popularity[recipe], recipe_order(recipe)))
    train, _ = read_reviews(data)
    require(read_json(CF_OUTPUT / "validation_selection.json")["selected_n"] == 5,
            "CF parameter changed")

    def generate():
        for user in sorted(train):
            history = train[user]
            contributions = score_candidates(history, adjacency)
            for recipe in removed:
                contributions.pop(recipe, None)
            ranked, scorable = rank_for_n(history, contributions, 5, fallback, k=50)
            yield user, [(recipe, score, source) for recipe, score, source, _ in ranked], scorable

    return write_rankings("1_item_cf", generate(), data, output)


_PATH_WORKER = None


def _init_path_worker(data: Path, removed: set[str]):
    global _PATH_WORKER
    _, recipes, families = load_fit_paths(data, PATH_OUTPUT, CF_OUTPUT)
    _, _, popularity = load_fit(data, CF_OUTPUT)
    index = {recipe: position for position, recipe in enumerate(recipes)}
    _PATH_WORKER = (recipes, families, index,
                    sorted(range(len(recipes)), key=lambda i: (-popularity[recipes[i]], i)),
                    {index[recipe] for recipe in removed})


def _score_path_user(payload):
    user, history = payload
    recipes, families, index, fallback, removed = _PATH_WORKER
    positive = [(index[recipe], POSITIVE_GAIN[rating]) for recipe, rating in history
                if rating in POSITIVE_GAIN]
    excluded = removed | {index[recipe] for recipe, _ in history}
    if positive:
        indices = [position for position, _ in positive]
        gains = np.asarray([gain for _, gain in positive], dtype=np.float64)
        ingredient = weighted_jaccard_profile(families["ingredient"], indices, gains)
        direct = weighted_jaccard_profile(families["direct_foodon"], indices, gains)
        parent = weighted_jaccard_profile(families["parent_foodon"], indices, gains)
        scores = 0.5 * ingredient + 0.25 * direct + 0.25 * parent
    else:
        scores = np.zeros(len(recipes))
    ranked, scorable = rank(scores, excluded, fallback, k=50)
    return user, [(recipes[position], score, source) for position, score, source in ranked], scorable


def rank_paths(data: Path = DATA, output: Path = OUTPUT) -> dict:
    removed = require_preflight(data, output)
    require(read_json(PATH_OUTPUT / "validation_selection.json")["selected_beta"] == 0.5,
            "Explicit path parameter changed")
    train, _ = read_reviews(data)
    payloads = ((user, train[user]) for user in sorted(train))
    with ProcessPoolExecutor(max_workers=6, initializer=_init_path_worker,
                             initargs=(data, removed)) as pool:
        return write_rankings("3_explicit_paths",
                              pool.map(_score_path_user, payloads, chunksize=4), data, output)


def rank_rotate(variant: str, data: Path = DATA, output: Path = OUTPUT) -> dict:
    import torch
    from pykeen.losses import MarginRankingLoss
    from pykeen.models import RotatE
    from scipy.stats import rankdata

    require(variant in {"primary", "hierarchy"}, "Invalid RotatE variant")
    removed = require_preflight(data, output)
    base = PRIMARY if variant == "primary" else HIERARCHY
    weight = read_json(base / "validation_selection.json")["selected_weight_five"]
    require(weight == (0.5 if variant == "primary" else 0.75), "Fusion weight changed")
    factory = load_factory(data, variant)
    config = read_json(ROTATE)
    model = RotatE(triples_factory=factory,
                   embedding_dim=config["training"]["embedding_dim"],
                   loss=MarginRankingLoss(margin=config["training"]["margin"]),
                   random_seed=config["training"]["seed"])
    model.load_state(base / "model.pt")
    model.eval()
    recipes = load_recipes(data)
    embedded = [recipe for recipe in recipes if recipe in factory.entity_to_id]
    embedded_ids = torch.tensor([factory.entity_to_id[recipe] for recipe in embedded],
                                dtype=torch.long)
    unembedded = [recipe for recipe in recipes
                  if recipe not in factory.entity_to_id and recipe not in removed]
    train, _ = read_reviews(data)
    name = "4_rotate_" + variant

    def generate():
        with torch.inference_mode():
            allowed = np.array([recipe not in removed for recipe in embedded])
            for user in sorted(train):
                excluded = {recipe for recipe, _ in train[user]}
                mask = allowed & np.array([recipe not in excluded for recipe in embedded])
                eligible = np.flatnonzero(mask)
                user_id = factory.entity_to_id[user]
                percentiles = []
                for relation in ("RATED_4", "RATED_5"):
                    query = torch.tensor([[user_id, factory.relation_to_id[relation]]],
                                         dtype=torch.long)
                    raw = model.predict_t(query, tails=embedded_ids).reshape(-1).cpu().numpy()
                    require(np.isfinite(raw).all(), "Non-finite RotatE score")
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
                yield user, ranked, len(eligible)

    return write_rankings(name, generate(), data, output)


def evaluate(data: Path = DATA, output: Path = OUTPUT) -> dict:
    removed = require_preflight(data, output)
    for model in MODELS:
        manifest = read_json(output / f"{model}_rank_manifest.json")
        require(manifest["status"] == "reduced_catalog_ranking_complete" and
                manifest["ranking_sha256"] == sha(output / f"{model}_top50.csv") and
                manifest["preflight_sha256"] == sha(output / "preflight.json"),
                f"Unverified rank source: {model}")
    targets = defaultdict(dict)
    removed_targets = Counter()
    for row in rows(data / "reviews.csv"):
        if row["split"] == "test":
            if row["recipe_id"] in removed:
                removed_targets["all"] += 1
                removed_targets["positive"] += int(row["rating"]) >= 4
            else:
                targets[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    train, _ = read_reviews(data)
    for user in train:
        targets[user]  # keep users whose eight test targets were all removed
    ingredients = load_ingredients(data, load_recipes(data))
    catalog = [recipe for recipe in load_recipes(data) if recipe not in removed]
    require(len(catalog) == 75872 and len(targets) == 4000, "Reduced population changed")
    timestamp = datetime.now(timezone.utc).isoformat()
    per_user_path = output / "per_user_metrics.csv"
    agg_path = output / "aggregate_metrics.csv"
    exposure_path = output / "recipe_exposure.csv"
    per_user = defaultdict(dict)
    exposure = {}
    with per_user_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "user_id", "retained_test_targets", "retained_positive_targets",
                         "scorable_candidates", "fallback_top50", "hits_at_10", "hits_at_50",
                         "precision_at_10", "precision_at_50", "ndcg_at_10", "ndcg_at_50",
                         "ingredient_ild_at_10", "ingredient_ild_at_50",
                         "measurable_pairs_at_10", "measurable_pairs_at_50",
                         "mapped_positions_at_10", "mapped_positions_at_50"))
        for model in MODELS:
            current_user, current = None, []
            exposure[model] = {k: Counter() for k in K_VALUES}

            def commit(user, entries):
                require(user in targets and len(entries) == 50 and
                        [int(row["rank"]) for row in entries] == list(range(1, 51)) and
                        not ({row["recipe_id"] for row in entries} & removed),
                        "Invalid ranked user list")
                ranked = [row["recipe_id"] for row in entries]
                values = metric_values(ranked, targets[user])
                diversity = {}
                for k in K_VALUES:
                    ild, possible, measurable = list_ild(ranked[:k], ingredients)
                    require(possible == k * (k - 1) // 2, "Pair denominator changed")
                    diversity[k] = (ild, measurable,
                                    sum(recipe in ingredients for recipe in ranked[:k]))
                    exposure[model][k].update(ranked[:k])
                per_user[model, user] = (values, diversity)
                writer.writerow((model, user, len(targets[user]), values["positive_test_targets"],
                                 entries[0]["scorable_candidates"],
                                 sum(row["score_source"] == "fallback" for row in entries),
                                 values["hits_at_10"], values["hits_at_50"],
                                 *(format(values[f"precision_at_{k}"], ".17g") for k in K_VALUES),
                                 *("" if values[f"ndcg_at_{k}"] is None else
                                   format(values[f"ndcg_at_{k}"], ".17g") for k in K_VALUES),
                                 *("" if diversity[k][0] is None else
                                   format(diversity[k][0], ".17g") for k in K_VALUES),
                                 *(diversity[k][1] for k in K_VALUES),
                                 *(diversity[k][2] for k in K_VALUES)))

            for row in rows(output / f"{model}_top50.csv"):
                if current_user is not None and row["user_id"] != current_user:
                    commit(current_user, current)
                    current = []
                current_user = row["user_id"]
                current.append(row)
            commit(current_user, current)
    require(len(per_user) == 16000, "Incomplete per-user metrics")
    with exposure_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "k", "recipe_id", "exposure_count"))
        for model in MODELS:
            for k in K_VALUES:
                require(sum(exposure[model][k].values()) == 4000 * k,
                        "Exposure positions changed")
                for recipe in catalog:
                    writer.writerow((model, k, recipe, exposure[model][k][recipe]))
    positive_users = [user for user in train
                      if any(rating >= 4 for rating in targets[user].values())]
    with agg_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "k", "users", "positive_test_users", "retained_test_targets",
                         "retained_positive_targets", "observed_positive_hits", "precision",
                         "ndcg_positive_users", "defined_ild_users", "mean_user_ingredient_ild",
                         "measurable_pairs", "possible_pairs", "pair_coverage",
                         "mapped_positions", "ingredient_position_coverage", "fallback_top50_rows",
                         "distinct_exposed_recipes", "catalog_coverage", "exposure_gini",
                         "max_recipe_exposure", "top_1_percent_exposure_share"))
        for model in MODELS:
            model_users = [per_user[model, user] for user in sorted(train)]
            for k in K_VALUES:
                counts = [exposure[model][k][recipe] for recipe in catalog]
                ilds = [diversity[k][0] for _, diversity in model_users
                        if diversity[k][0] is not None]
                measurable = sum(diversity[k][1] for _, diversity in model_users)
                mapped = sum(diversity[k][2] for _, diversity in model_users)
                possible = 4000 * k * (k - 1) // 2
                hits = sum(values[f"hits_at_{k}"] for values, _ in model_users)
                precision = sum(values[f"precision_at_{k}"] for values, _ in model_users) / 4000
                ndcgs = [per_user[model, user][0][f"ndcg_at_{k}"] for user in positive_users]
                distinct = sum(count > 0 for count in counts)
                top_count = math.ceil(len(catalog) * 0.01)
                writer.writerow((model, k, 4000, len(positive_users),
                                 sum(len(targets[user]) for user in train),
                                 sum(per_user[model, user][0]["positive_test_targets"] for user in train),
                                 hits, format(precision, ".17g"),
                                 format(sum(ndcgs) / len(ndcgs), ".17g") if ndcgs else "",
                                 len(ilds), format(sum(ilds) / len(ilds), ".17g") if ilds else "",
                                 measurable, possible, format(measurable / possible, ".17g"),
                                 mapped, format(mapped / (4000 * k), ".17g"),
                                 sum(int(row["fallback_rows"]) for row in
                                     [read_json(output / f"{model}_rank_manifest.json")]) if k == 50 else "",
                                 distinct, format(distinct / len(catalog), ".17g"),
                                 format(gini(counts), ".17g"), max(counts),
                                 format(sum(sorted(counts, reverse=True)[:top_count]) / (4000 * k), ".17g")))
    result = {"status": "reduced_catalog_evaluation_complete", "models": list(MODELS),
              "k": list(K_VALUES), "users": 4000, "catalog_recipes": len(catalog),
              "removed_test_targets": removed_targets["all"],
              "removed_positive_test_targets": removed_targets["positive"],
              "positive_test_users": len(positive_users),
              "rank_sha256": {model: sha(output / f"{model}_top50.csv") for model in MODELS},
              "selection_sha256": sha(output / "excluded_top100_training_mean.csv"),
              "per_user_sha256": sha(per_user_path), "aggregate_sha256": sha(agg_path),
              "exposure_sha256": sha(exposure_path),
              "created_utc": timestamp}
    write_json(output / "evaluation_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "rank-cf", "rank-paths",
                                            "rank-primary", "rank-hierarchy", "evaluate"))
    command = parser.parse_args().command
    functions = {"preflight": preflight, "rank-cf": rank_cf,
                 "rank-paths": rank_paths,
                 "rank-primary": lambda: rank_rotate("primary"),
                 "rank-hierarchy": lambda: rank_rotate("hierarchy"),
                 "evaluate": evaluate}
    print(json.dumps(functions[command](), indent=2))


if __name__ == "__main__":
    main()
