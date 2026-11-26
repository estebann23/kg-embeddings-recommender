"""Frozen RQ1 condition 1: rating-aware item-based collaborative filtering."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.rotate_ranking import recipe_order


OUTPUT = ROOT / "results/item_cf_hummus_4000_v1"
WEIGHTS = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
GAINS = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}
GRID = (5, 10, 20)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_reviews(data: Path, include_validation: bool = False):
    train = defaultdict(list)
    validation = defaultdict(dict)
    with (data / "reviews.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split == "train":
                train[row["user_id"]].append((row["recipe_id"], int(row["rating"])))
            elif split == "validation" and include_validation:
                validation[row["user_id"]][row["recipe_id"]] = int(row["rating"])
            # Test rows are never converted to rating values here.
    return train, validation


def load_recipes(data: Path) -> list[str]:
    with (data / "recipes.csv").open(encoding="utf-8", newline="") as stream:
        recipes = [row["recipe_id"] for row in csv.DictReader(stream)]
    require(len(recipes) == len(set(recipes)) == 75972, "Recipe universe changed")
    return sorted(recipes, key=recipe_order)


def fit(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    lock = verify_lock(data)
    contract = read_json(ROOT / "config/rq1_experiment.json")
    require(contract["conditions"]["1_item_cf"]["validation_grid"]["N"] == list(GRID),
            "Neighbor grid changed")
    require(contract["rating_policy"]["training_preference_weight"] ==
            {str(r): WEIGHTS[r] for r in WEIGHTS}, "Rating weights changed")
    recipes = load_recipes(data)
    train, _ = read_reviews(data)
    require(len(train) == 4000 and sum(map(len, train.values())) == 112000,
            "Training cohort size changed")
    for history in train.values():
        require(len(history) == 28 and len({recipe for recipe, _ in history}) == 28,
                "Unexpected training history")

    norm_sq = defaultdict(float)
    pair_dot = defaultdict(float)
    positive_count = Counter()
    for user in sorted(train):
        entries = sorted(((recipe, WEIGHTS[rating]) for recipe, rating in train[user]
                          if WEIGHTS[rating] != 0), key=lambda x: recipe_order(x[0]))
        for recipe, weight in entries:
            norm_sq[recipe] += weight * weight
        for i, (left, left_weight) in enumerate(entries):
            for right, right_weight in entries[i + 1:]:
                pair_dot[(left, right)] += left_weight * right_weight
        positive_count.update(recipe for recipe, rating in train[user] if rating >= 4)

    output.mkdir(parents=True, exist_ok=True)
    similarity_path = output / "positive_item_cosine.csv"
    count = 0
    with similarity_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("item_a", "item_b", "similarity"))
        for (left, right), dot in sorted(pair_dot.items(),
                                         key=lambda pair: (recipe_order(pair[0][0]),
                                                           recipe_order(pair[0][1]))):
            if dot <= 0:
                continue
            similarity = dot / math.sqrt(norm_sq[left] * norm_sq[right])
            require(math.isfinite(similarity) and 0 < similarity <= 1 + 1e-12,
                    "Invalid fitted cosine")
            writer.writerow((left, right, format(min(1.0, similarity), ".17g")))
            count += 1
    popularity_path = output / "training_positive_popularity.csv"
    with popularity_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("recipe_id", "training_ratings_4_or_5"))
        for recipe in recipes:
            writer.writerow((recipe, positive_count[recipe]))
    result = {
        "status": "fitted_training_only", "condition": "1_item_cf",
        "dataset_id": data.name, "algorithm": "signed_rating_sparse_item_cosine_positive_part",
        "training_users": len(train), "training_ratings": sum(map(len, train.values())),
        "catalog_recipes": len(recipes), "nonzero_norm_items": len(norm_sq),
        "observed_item_pairs": len(pair_dot), "positive_similarity_pairs": count,
        "rating_weights": {str(r): WEIGHTS[r] for r in WEIGHTS},
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "reviews_sha256": sha(data / "reviews.csv"), "recipes_sha256": sha(data / "recipes.csv"),
        "similarity_sha256": sha(similarity_path), "popularity_sha256": sha(popularity_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "No validation or test rating was used to fit item similarities or popularity."
    }
    write_json(output / "fit_manifest.json", result)
    return result


def load_fit(data: Path = DATA, output: Path = OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    manifest = read_json(output / "fit_manifest.json")
    require(manifest["status"] == "fitted_training_only" and
            manifest["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            manifest["reviews_sha256"] == sha(data / "reviews.csv") and
            manifest["recipes_sha256"] == sha(data / "recipes.csv"), "Fit source mismatch")
    require(manifest["similarity_sha256"] == sha(output / "positive_item_cosine.csv") and
            manifest["popularity_sha256"] == sha(output / "training_positive_popularity.csv"),
            "Fitted artifact changed")
    adjacency = defaultdict(list)
    count = 0
    with (output / "positive_item_cosine.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            left, right, similarity = row["item_a"], row["item_b"], float(row["similarity"])
            require(math.isfinite(similarity) and 0 < similarity <= 1, "Invalid similarity artifact")
            adjacency[left].append((right, similarity))
            adjacency[right].append((left, similarity))
            count += 1
    require(count == manifest["positive_similarity_pairs"], "Similarity pair count changed")
    popularity = {}
    with (output / "training_positive_popularity.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            popularity[row["recipe_id"]] = int(row["training_ratings_4_or_5"])
    require(len(popularity) == 75972, "Popularity catalog changed")
    return manifest, adjacency, popularity


def score_candidates(history: list[tuple[str, int]], adjacency) -> dict[str, list[tuple[float, str, float]]]:
    """Collect positive item links, retaining signed user rating contributions."""
    excluded = {recipe for recipe, _ in history}
    contributions = defaultdict(list)
    for item, rating in sorted(history, key=lambda x: recipe_order(x[0])):
        weight = WEIGHTS[rating]
        if weight == 0:
            continue
        for candidate, similarity in adjacency.get(item, ()):
            if candidate not in excluded:
                contributions[candidate].append((similarity, item, weight))
    for items in contributions.values():
        items.sort(key=lambda x: (-x[0], recipe_order(x[1])))
    return contributions


def rank_for_n(history, contributions, n: int, fallback: list[str], k: int = 10):
    require(n in GRID and k > 0, "Unsupported neighbor size or K")
    excluded = {recipe for recipe, _ in history}
    scored = {}
    for candidate, items in contributions.items():
        selected = items[:n]
        numerator = sum(sim * weight for sim, _, weight in selected)
        denominator = sum(sim * abs(weight) for sim, _, weight in selected)
        if denominator > 0:
            scored[candidate] = (numerator / denominator, len(selected))
    top = heapq.nsmallest(k, scored,
                          key=lambda recipe: (-scored[recipe][0], recipe_order(recipe)))
    ranked = [(recipe, scored[recipe][0], "score", scored[recipe][1]) for recipe in top]
    if len(ranked) < k:
        for recipe in fallback:
            if recipe not in excluded and recipe not in scored:
                ranked.append((recipe, None, "fallback", 0))
                if len(ranked) == k:
                    break
    require(len(ranked) == k and len({row[0] for row in ranked}) == k and
            not {row[0] for row in ranked} & excluded, "Invalid ranked list")
    return ranked, len(scored)


def ndcg_at_10(ranked, ratings: dict[str, int]) -> float | None:
    ideal = sorted((GAINS[rating] for rating in ratings.values() if GAINS[rating] > 0), reverse=True)
    if not ideal:
        return None
    denominator = sum(gain / math.log2(position + 2) for position, gain in enumerate(ideal[:10]))
    actual = sum(GAINS[ratings.get(recipe, 1)] / math.log2(position + 2)
                 for position, (recipe, _, _, _) in enumerate(ranked[:10]))
    return actual / denominator


def validate(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    manifest, adjacency, popularity = load_fit(data, output)
    recipes = load_recipes(data)
    fallback = sorted(recipes, key=lambda recipe: (-popularity[recipe], recipe_order(recipe)))
    train, validation = read_reviews(data, include_validation=True)
    require(len(train) == len(validation) == 4000 and
            sum(map(len, validation.values())) == 16000, "Validation cohort changed")
    totals = {n: 0.0 for n in GRID}
    eligible_users = 0
    generated = datetime.now(timezone.utc).isoformat()
    metrics_path = output / "validation_per_user.csv"
    rankings_path = output / "validation_recommendations_by_grid.csv"
    selected_rows = {n: [] for n in GRID}
    with metrics_path.open("w", encoding="utf-8", newline="") as metric_stream, \
         rankings_path.open("w", encoding="utf-8", newline="") as ranking_stream:
        metric_writer = csv.writer(metric_stream, lineterminator="\n")
        ranking_writer = csv.writer(ranking_stream, lineterminator="\n")
        metric_writer.writerow(("user_id", "positive_validation_targets", "scorable_candidates",
                                "ndcg_5", "ndcg_10", "ndcg_20"))
        ranking_writer.writerow(("generated_utc", "neighbor_n", "user_id", "rank", "recipe_id",
                                 "score", "score_source", "contributing_neighbors"))
        for user in sorted(train):
            history = train[user]
            require(len(history) == 28 and len(validation[user]) == 4, "User split size changed")
            contributions = score_candidates(history, adjacency)
            metrics = []
            scorable_counts = []
            positive_targets = sum(rating >= 4 for rating in validation[user].values())
            if positive_targets:
                eligible_users += 1
            for n in GRID:
                ranked, scorable_count = rank_for_n(history, contributions, n, fallback)
                scorable_counts.append(scorable_count)
                metric = ndcg_at_10(ranked, validation[user])
                metrics.append(metric)
                if metric is not None:
                    totals[n] += metric
                for position, (recipe, score, source, neighbor_count) in enumerate(ranked, 1):
                    row = (generated, n, user, position, recipe,
                           "" if score is None else format(score, ".17g"), source, neighbor_count)
                    ranking_writer.writerow(row)
                    selected_rows[n].append(row)
            require(len(set(scorable_counts)) == 1, "Scorable set changed across N grid")
            metric_writer.writerow((user, positive_targets, scorable_counts[0],
                                    *("" if value is None else format(value, ".17g") for value in metrics)))
    require(eligible_users > 0, "No positive validation users")
    means = {n: totals[n] / eligible_users for n in GRID}
    selected_n = max(GRID, key=lambda n: (means[n], -n))
    selected_path = output / "validation_recommendations_selected.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("generated_utc", "neighbor_n", "user_id", "rank", "recipe_id",
                         "score", "score_source", "contributing_neighbors"))
        writer.writerows(selected_rows[selected_n])
    result = {
        "status": "validation_selected_no_test_access", "condition": "1_item_cf",
        "selected_n": selected_n, "grid": list(GRID),
        "mean_validation_ndcg_at_10": {str(n): means[n] for n in GRID},
        "users_with_positive_validation": eligible_users,
        "training_users": len(train), "candidate_catalog": len(recipes),
        "per_user_eligible_candidates": len(recipes) - 28,
        "fit_manifest_sha256": sha(output / "fit_manifest.json"),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "validation_per_user_sha256": sha(metrics_path),
        "validation_grid_recommendations_sha256": sha(rankings_path),
        "validation_selected_recommendations_sha256": sha(selected_path),
        "generated_utc": generated,
        "note": "Selection used validation NDCG@10 only; no test rating was read."
    }
    write_json(output / "validation_selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "validate"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = fit(args.data, args.output) if args.command == "fit" else validate(args.data, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
