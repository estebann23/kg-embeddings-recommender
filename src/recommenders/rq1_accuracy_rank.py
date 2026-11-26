"""Freeze and generate test-blind top-50 rankings for RQ1 conditions 1, 3, 4."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.recommenders.contract import DATA, ROOT, ROTATE, sha, verify_lock
from src.recommenders.explicit_paths import (OUTPUT as PATH_OUTPUT, GRID as BETA_GRID,
                                             POSITIVE_GAIN, load_fit_paths, rank,
                                             weighted_jaccard_profile)
from src.recommenders.item_cf import (OUTPUT as CF_OUTPUT, load_fit, load_recipes,
                                      rank_for_n, read_reviews, require, score_candidates)
from src.recommenders.rotate import load_factory
from src.recommenders.rotate_ranking import recipe_order


OUTPUT = ROOT / "results/rq1_accuracy_test"
ROTATE_OUTPUT = ROOT / "results/rotate_primary_full"
ADDENDUM = ROOT / "config/rq1_accuracy_test_addendum.json"
CONDITIONS = ("1_item_cf", "3_explicit_paths", "4_rotate_primary")
HEADER = ("generated_utc", "condition", "user_id", "rank", "recipe_id", "score",
          "score_source", "scorable_candidates")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_popularity(data: Path, base: Path) -> dict[str, int]:
    manifest = read_json(base / "fit_manifest.json")
    require(manifest["popularity_sha256"] == sha(base / "training_positive_popularity.csv") and
            manifest["reviews_sha256"] == sha(data / "reviews.csv"), "Popularity source changed")
    with (base / "training_positive_popularity.csv").open(encoding="utf-8", newline="") as stream:
        result = {row["recipe_id"]: int(row["training_ratings_4_or_5"])
                  for row in csv.DictReader(stream)}
    require(len(result) == 75972, "Popularity catalog changed")
    return result


def preflight(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    addendum = read_json(ADDENDUM)
    require(addendum["status"] == "frozen_before_test_labels" and
            addendum["dataset_id"] == data.name and
            addendum["ranking_depth"] == 50 and
            addendum["metrics"] == ["precision_at_10", "precision_at_50", "ndcg_at_10", "ndcg_at_50"],
            "Accuracy addendum changed")
    from src.recommenders.verify_item_cf import verify as verify_cf
    from src.recommenders.verify_explicit_paths import verify as verify_paths
    from src.recommenders.verify_item_cf_attributes import verify as verify_attributes
    from src.recommenders.verify_rotate_validation import verify as verify_rotate
    require(verify_cf()["selected_n"] == 5, "Condition 1 selection changed")
    require(verify_attributes()["selected_alpha"] == 0, "Condition 2 selection gate changed")
    require(verify_paths()["selected_beta"] == 0.5, "Condition 3 selection changed")
    require(verify_rotate("primary")["weight_five"] == 0.5 and
            verify_rotate("hierarchy")["weight_five"] == 0.75,
            "Condition 4 validation gate changed")
    recipes = load_recipes(data)
    train, _ = read_reviews(data, include_validation=False)
    require(len(train) == 4000 and all(len(history) == 28 for history in train.values()),
            "Training cohort changed")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "preflight_passed_no_test_rating_access", "dataset_id": data.name,
        "conditions": list(CONDITIONS), "ranking_depth": 50,
        "cohort_users": len(train), "candidate_catalog": len(recipes),
        "eligible_candidates_per_user": len(recipes) - 28,
        "selected_neighbor_n": 5, "selected_beta": 0.5,
        "rotate_variant": "primary", "rotate_weight_five": 0.5,
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "addendum_sha256": sha(ADDENDUM),
        "reviews_sha256": sha(data / "reviews.csv"), "recipes_sha256": sha(data / "recipes.csv"),
        "cf_validation_sha256": sha(CF_OUTPUT / "validation_selection.json"),
        "path_validation_sha256": sha(PATH_OUTPUT / "validation_selection.json"),
        "rotate_validation_sha256": sha(ROTATE_OUTPUT / "validation_selection.json"),
        "rotate_model_sha256": sha(ROTATE_OUTPUT / "model.pt"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "preflight.json", result)
    return result


def require_preflight(data: Path, output: Path) -> dict:
    verify_lock(data)
    value = read_json(output / "preflight.json")
    require(value["status"] == "preflight_passed_no_test_rating_access" and
            value["addendum_sha256"] == sha(ADDENDUM) and
            value["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            value["reviews_sha256"] == sha(data / "reviews.csv") and
            value["recipes_sha256"] == sha(data / "recipes.csv") and
            value["cf_validation_sha256"] == sha(CF_OUTPUT / "validation_selection.json") and
            value["path_validation_sha256"] == sha(PATH_OUTPUT / "validation_selection.json") and
            value["rotate_validation_sha256"] == sha(ROTATE_OUTPUT / "validation_selection.json") and
            value["rotate_model_sha256"] == sha(ROTATE_OUTPUT / "model.pt"),
            "Test preflight seal changed")
    return value


def write_rankings(condition: str, generator, data: Path, output: Path, source_hashes: dict) -> dict:
    require_preflight(data, output)
    path = output / f"{condition}_top50.csv"
    temp = path.with_suffix(".csv.tmp")
    timestamp = datetime.now(timezone.utc).isoformat()
    user_count = row_count = fallback_count = 0
    min_scorable, max_scorable = 75972, 0
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(HEADER)
            for user, ranked, scorable in generator:
                require(len(ranked) == 50 and len({row[0] for row in ranked}) == 50,
                        "Incomplete top-50 ranking")
                for position, (recipe, score, source) in enumerate(ranked, 1):
                    require(source in {"score", "fallback"}, "Unknown score source")
                    writer.writerow((timestamp, condition, user, position, recipe,
                                     "" if score is None else format(score, ".17g"), source, scorable))
                    fallback_count += source == "fallback"
                    row_count += 1
                user_count += 1
                min_scorable = min(min_scorable, scorable)
                max_scorable = max(max_scorable, scorable)
        require(user_count == 4000 and row_count == 200000, "Test ranking cohort incomplete")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    result = {
        "status": "test_blind_rankings_complete", "condition": condition,
        "users": user_count, "rows": row_count, "fallback_top50_rows": fallback_count,
        "scorable_candidates_min": min_scorable, "scorable_candidates_max": max_scorable,
        "ranking_sha256": sha(path), "preflight_sha256": sha(output / "preflight.json"),
        "source_sha256": source_hashes, "generated_utc": timestamp,
        "note": "Ranking generated from training, static features and validation-selected parameters only."
    }
    write_json(output / f"{condition}_rank_manifest.json", result)
    return result


def rank_cf(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    _, adjacency, popularity = load_fit(data, CF_OUTPUT)
    recipes = load_recipes(data)
    fallback = sorted(recipes, key=lambda recipe: (-popularity[recipe], recipe_order(recipe)))
    train, _ = read_reviews(data)
    n = read_json(CF_OUTPUT / "validation_selection.json")["selected_n"]
    require(n == read_json(output / "preflight.json")["selected_neighbor_n"], "CF N changed")

    def generate():
        for user in sorted(train):
            history = train[user]
            contributions = score_candidates(history, adjacency)
            ranked, scorable = rank_for_n(history, contributions, n, fallback, k=50)
            yield user, [(recipe, score, source) for recipe, score, source, _ in ranked], scorable

    return write_rankings("1_item_cf", generate(), data, output,
                          {"fit_manifest": sha(CF_OUTPUT / "fit_manifest.json"),
                           "validation_selection": sha(CF_OUTPUT / "validation_selection.json")})


_PATH_WORKER = None


def _init_path_worker(data: Path):
    global _PATH_WORKER
    _, recipes, families = load_fit_paths(data, PATH_OUTPUT, CF_OUTPUT)
    popularity = read_popularity(data, CF_OUTPUT)
    _PATH_WORKER = (recipes, families, {recipe: index for index, recipe in enumerate(recipes)},
                    sorted(range(len(recipes)), key=lambda index: (-popularity[recipes[index]], index)))


def _score_path_user(payload):
    user, history, beta = payload
    recipes, families, recipe_index, fallback = _PATH_WORKER
    positive = [(recipe_index[recipe], POSITIVE_GAIN[rating]) for recipe, rating in history
                if rating in POSITIVE_GAIN]
    excluded = {recipe_index[recipe] for recipe, _ in history}
    if positive:
        indices = [index for index, _ in positive]
        gains = np.asarray([gain for _, gain in positive], dtype=np.float64)
        ingredient = weighted_jaccard_profile(families["ingredient"], indices, gains)
        direct = weighted_jaccard_profile(families["direct_foodon"], indices, gains)
        parent = weighted_jaccard_profile(families["parent_foodon"], indices, gains)
        scores = 0.5 * ingredient + 0.5 * ((1 - beta) * direct + beta * parent)
    else:
        scores = np.zeros(len(recipes))
    ranked, scorable = rank(scores, excluded, fallback, k=50)
    return user, [(recipes[index], score, source) for index, score, source in ranked], scorable


def rank_paths(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    train, _ = read_reviews(data)
    beta = read_json(PATH_OUTPUT / "validation_selection.json")["selected_beta"]
    require(beta == 0.5 == read_json(output / "preflight.json")["selected_beta"] and
            beta in BETA_GRID, "Path beta changed")
    payloads = ((user, train[user], beta) for user in sorted(train))
    with ProcessPoolExecutor(max_workers=6, initializer=_init_path_worker,
                             initargs=(data,)) as pool:
        return write_rankings("3_explicit_paths", pool.map(_score_path_user, payloads, chunksize=4),
                              data, output,
                              {"fit_manifest": sha(PATH_OUTPUT / "fit_manifest.json"),
                               "validation_selection": sha(PATH_OUTPUT / "validation_selection.json")})


def rank_rotate(data: Path = DATA, output: Path = OUTPUT) -> dict:
    import torch
    from pykeen.losses import MarginRankingLoss
    from pykeen.models import RotatE
    from scipy.stats import rankdata

    data, output = Path(data).resolve(), Path(output).resolve()
    require_preflight(data, output)
    config = read_json(ROTATE)
    selection = read_json(ROTATE_OUTPUT / "validation_selection.json")
    weight = selection["selected_weight_five"]
    require(weight == 0.5 == read_json(output / "preflight.json")["rotate_weight_five"],
            "RotatE fusion weight changed")
    factory = load_factory(data, "primary")
    model = RotatE(triples_factory=factory, embedding_dim=config["training"]["embedding_dim"],
                   loss=MarginRankingLoss(margin=config["training"]["margin"]),
                   random_seed=config["training"]["seed"])
    model.load_state(ROTATE_OUTPUT / "model.pt")
    model.eval()
    recipes = load_recipes(data)
    scorable = [recipe for recipe in recipes if recipe in factory.entity_to_id]
    recipe_ids = torch.tensor([factory.entity_to_id[recipe] for recipe in scorable], dtype=torch.long)
    index = {recipe: offset for offset, recipe in enumerate(scorable)}
    unembedded = [recipe for recipe in recipes if recipe not in index]
    require(len(scorable) + len(unembedded) == 75972 and len(unembedded) == 1917,
            "RotatE recipe vocabulary changed")
    train, _ = read_reviews(data)

    def generate():
        with torch.inference_mode():
            for user in sorted(train):
                eligible_mask = np.ones(len(scorable), dtype=bool)
                for recipe, _ in train[user]:
                    if recipe in index:
                        eligible_mask[index[recipe]] = False
                eligible_index = np.flatnonzero(eligible_mask)
                user_index = factory.entity_to_id[user]
                scores = []
                for relation in ("RATED_4", "RATED_5"):
                    query = torch.tensor([[user_index, factory.relation_to_id[relation]]], dtype=torch.long)
                    raw = model.predict_t(query, tails=recipe_ids).reshape(-1).cpu().numpy()
                    require(np.isfinite(raw).all(), "Non-finite RotatE test-blind score")
                    scores.append(rankdata(raw[eligible_index], method="average") / len(eligible_index))
                combined = (1 - weight) * scores[0] + weight * scores[1]
                cutoff = np.partition(combined, -50)[-50]
                contenders = np.flatnonzero(combined >= cutoff)
                top = contenders[np.lexsort((eligible_index[contenders], -combined[contenders]))[:50]]
                ranked = [(scorable[eligible_index[position]], float(combined[position]), "score")
                          for position in top]
                if len(ranked) < 50:
                    excluded = {recipe for recipe, _ in train[user]}
                    ranked.extend((recipe, None, "fallback") for recipe in unembedded
                                  if recipe not in excluded)
                    ranked = ranked[:50]
                yield user, ranked, len(eligible_index)

    return write_rankings("4_rotate_primary", generate(), data, output,
                          {"training_manifest": sha(ROTATE_OUTPUT / "training_manifest.json"),
                           "validation_selection": sha(ROTATE_OUTPUT / "validation_selection.json"),
                           "model": sha(ROTATE_OUTPUT / "model.pt")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "rank-cf", "rank-paths", "rank-rotate"))
    args = parser.parse_args()
    action = {"preflight": preflight, "rank-cf": rank_cf,
              "rank-paths": rank_paths, "rank-rotate": rank_rotate}[args.command]
    print(json.dumps(action(), indent=2))


if __name__ == "__main__":
    main()
