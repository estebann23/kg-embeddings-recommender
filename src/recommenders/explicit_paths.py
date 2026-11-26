"""RQ1 condition 3: deterministic weighted matching of audited food paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf import (GAINS, OUTPUT as BASE_OUTPUT, load_fit,
                                      load_recipes, read_reviews, require)
from src.recommenders.rotate_ranking import recipe_order


OUTPUT = ROOT / "results/explicit_paths_hummus_4000_v1"
GRID = (0.0, 0.25, 0.5)
POSITIVE_GAIN = {4: 1, 5: 2}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def feature_matrix(recipes: list[str], feature_sets: dict[str, set[str]]):
    feature_ids = sorted(set().union(*feature_sets.values())) if feature_sets else []
    columns = {value: index for index, value in enumerate(feature_ids)}
    recipe_index = {recipe: index for index, recipe in enumerate(recipes)}
    rows, cols = [], []
    for recipe, values in feature_sets.items():
        for value in values:
            rows.append(recipe_index[recipe])
            cols.append(columns[value])
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                               shape=(len(recipes), len(feature_ids))).tocsr()
    require(matrix.nnz == len(rows), "Repeated feature incidence")
    return matrix, feature_ids


def fit(data: Path = DATA, output: Path = OUTPUT, base: Path = BASE_OUTPUT) -> dict:
    data, output, base = Path(data).resolve(), Path(output).resolve(), Path(base).resolve()
    verify_lock(data)
    contract = read_json(ROOT / "config/rq1_experiment.json")
    require(contract["conditions"]["3_explicit_graph"]["validation_grid"]["beta"] == list(GRID),
            "Hierarchy grid changed")
    _, _, popularity = load_fit(data, base)
    recipes = load_recipes(data)
    recipe_set = set(recipes)
    ingredient_sets = defaultdict(set)
    with (data / "recipe_ingredients.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["recipe_id"] in recipe_set and row["evidence_state"] == "INFERRED",
                    "Unexpected ingredient path evidence")
            ingredient_sets[row["recipe_id"]].add(row["ingredient_id"])
    direct_by_ingredient = defaultdict(set)
    with (data / "foodon_links.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["evidence_state"] == "INFERRED" and row["foodon_iri"].startswith(
                "http://purl.obolibrary.org/obo/FOODON_"), "Unexpected direct concept evidence")
            direct_by_ingredient[row["ingredient_id"]].add(row["foodon_iri"])
    direct_sets = {recipe: set().union(*(direct_by_ingredient[ingredient] for ingredient in values))
                   for recipe, values in ingredient_sets.items()}
    parent_by_child = defaultdict(set)
    with (data / "foodon_hierarchy.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["evidence_state"] == "SOURCE_ASSERTED" and
                    all(row[key].startswith("http://purl.obolibrary.org/obo/FOODON_")
                        for key in ("child_iri", "parent_iri")), "Unexpected parent path evidence")
            parent_by_child[row["child_iri"]].add(row["parent_iri"])
    raw_parents = {recipe: set().union(*(parent_by_child[concept] for concept in values))
                   for recipe, values in direct_sets.items()}
    mapped_recipe_count = sum(bool(values) for values in ingredient_sets.values())
    parent_df = Counter(parent for values in raw_parents.values() for parent in values)
    broad_parents = {parent for parent, count in parent_df.items()
                     if count > 0.05 * mapped_recipe_count}
    parent_sets = {recipe: values - broad_parents for recipe, values in raw_parents.items()}
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for family, values in (("ingredient", ingredient_sets), ("direct_foodon", direct_sets),
                           ("parent_foodon", parent_sets)):
        matrix, feature_ids = feature_matrix(recipes, values)
        sparse.save_npz(output / f"{family}_binary.npz", matrix)
        write_json(output / f"{family}_ids.json", feature_ids)
        counts[family] = {"recipes_with_features": int(np.count_nonzero(matrix.getnnz(axis=1))),
                          "feature_ids": len(feature_ids), "incidences": matrix.nnz,
                          "binary_sha256": sha(output / f"{family}_binary.npz"),
                          "ids_sha256": sha(output / f"{family}_ids.json")}
    write_json(output / "recipe_order.json", recipes)
    result = {
        "status": "fitted_static_paths_and_training_popularity", "condition": "3_explicit_graph",
        "dataset_id": data.name, "recipes": len(recipes), "grid": list(GRID),
        "mapped_recipes_for_parent_threshold": mapped_recipe_count,
        "broad_parent_threshold_fraction": 0.05, "broad_parent_ids_removed": sorted(broad_parents),
        "source_asserted_parent_edges": sum(map(len, parent_by_child.values())),
        "feature_families": counts,
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "base_fit_manifest_sha256": sha(base / "fit_manifest.json"),
        "recipe_ingredients_sha256": sha(data / "recipe_ingredients.csv"),
        "foodon_links_sha256": sha(data / "foodon_links.csv"),
        "foodon_hierarchy_sha256": sha(data / "foodon_hierarchy.csv"),
        "recipe_order_sha256": sha(output / "recipe_order.json"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "One source-asserted named superclass step only; static feature counts and IDF use recipe data, not held-out ratings."
    }
    require(len(popularity) == len(recipes), "Training popularity catalog mismatch")
    write_json(output / "fit_manifest.json", result)
    return result


def load_fit_paths(data: Path, output: Path, base: Path):
    verify_lock(data)
    manifest = read_json(output / "fit_manifest.json")
    require(manifest["status"] == "fitted_static_paths_and_training_popularity" and
            manifest["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            manifest["base_fit_manifest_sha256"] == sha(base / "fit_manifest.json") and
            manifest["recipe_ingredients_sha256"] == sha(data / "recipe_ingredients.csv") and
            manifest["foodon_links_sha256"] == sha(data / "foodon_links.csv") and
            manifest["foodon_hierarchy_sha256"] == sha(data / "foodon_hierarchy.csv") and
            manifest["recipe_order_sha256"] == sha(output / "recipe_order.json"), "Path source mismatch")
    recipes = read_json(output / "recipe_order.json")
    require(recipes == load_recipes(data), "Recipe order changed")
    families = {}
    for family in ("ingredient", "direct_foodon", "parent_foodon"):
        entry = manifest["feature_families"][family]
        require(entry["binary_sha256"] == sha(output / f"{family}_binary.npz") and
                entry["ids_sha256"] == sha(output / f"{family}_ids.json"), "Path feature artifact changed")
        binary = sparse.load_npz(output / f"{family}_binary.npz").tocsr()
        ids = read_json(output / f"{family}_ids.json")
        require(binary.shape == (len(recipes), len(ids)) and binary.nnz == entry["incidences"],
                "Path feature shape changed")
        df = np.asarray(binary.getnnz(axis=0)).ravel()
        weights = np.log((1 + len(recipes)) / (1 + df))
        weighted = binary.multiply(weights).tocsr()
        masses = np.asarray(weighted.sum(axis=1)).ravel()
        families[family] = (binary, weighted, masses, ids)
    return manifest, recipes, families


def weighted_jaccard_profile(family, history_indices: list[int], gains: np.ndarray) -> np.ndarray:
    binary, weighted, masses, _ = family
    intersections = (weighted[history_indices] @ binary.T).toarray()
    union = masses[history_indices, None] + masses[None, :] - intersections
    np.divide(intersections, union, out=intersections, where=union > 0)
    return gains @ intersections / gains.sum()


def rank(scores: np.ndarray, excluded: set[int], fallback: list[int], k: int = 10):
    scores = scores.copy()
    scores[list(excluded)] = 0
    scorable = np.flatnonzero(scores > 0)
    if len(scorable) >= k:
        threshold = np.partition(scores[scorable], -k)[-k]
        above = scorable[scores[scorable] > threshold]
        above = above[np.argsort(-scores[above], kind="stable")]
        equal = scorable[scores[scorable] == threshold]
        top = np.concatenate((above, equal[:k - len(above)]))
    else:
        top = scorable[np.argsort(-scores[scorable], kind="stable")]
    ranked = [(int(index), float(scores[index]), "score") for index in top]
    if len(ranked) < k:
        for index in fallback:
            if index not in excluded and scores[index] <= 0:
                ranked.append((index, None, "fallback"))
                if len(ranked) == k:
                    break
    require(len(ranked) == k and len({row[0] for row in ranked}) == k,
            "Invalid explicit path ranking")
    return ranked, len(scorable)


def ndcg(ranked, targets: dict[str, int], recipes: list[str]) -> float | None:
    ideal = sorted((GAINS[r] for r in targets.values() if GAINS[r] > 0), reverse=True)
    if not ideal:
        return None
    denominator = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
    actual = sum(GAINS[targets.get(recipes[index], 1)] / math.log2(rank + 2)
                 for rank, (index, _, _) in enumerate(ranked))
    return actual / denominator


_WORKER = None


def _init_worker(data: Path, output: Path, base: Path):
    global _WORKER
    manifest, recipes, families = load_fit_paths(data, output, base)
    _, _, popularity = load_fit(data, base)
    _WORKER = (manifest, recipes, families, {recipe: index for index, recipe in enumerate(recipes)},
               sorted(range(len(recipes)), key=lambda index: (-popularity[recipes[index]], index)))


def _trace(index: int, history_indices: list[int], families, beta: float):
    result = []
    for family in ("ingredient", "direct_foodon", "parent_foodon"):
        if family == "parent_foodon" and beta == 0:
            result.append([])
            continue
        matrix, _, _, ids = families[family]
        candidate = set(matrix[index].indices)
        matches = set()
        for history_index in history_indices:
            matches.update(candidate.intersection(matrix[history_index].indices))
        result.append([ids[col] for col in sorted(matches)])
    return result


def _score_user(payload):
    user, history, targets = payload
    _, recipes, families, recipe_index, fallback = _WORKER
    positive = [(recipe_index[recipe], POSITIVE_GAIN[rating]) for recipe, rating in history
                if rating in POSITIVE_GAIN]
    excluded = {recipe_index[recipe] for recipe, _ in history}
    if positive:
        history_indices = [index for index, _ in positive]
        gains = np.asarray([gain for _, gain in positive], dtype=np.float64)
        ingredient_score = weighted_jaccard_profile(families["ingredient"], history_indices, gains)
        direct_score = weighted_jaccard_profile(families["direct_foodon"], history_indices, gains)
        parent_score = weighted_jaccard_profile(families["parent_foodon"], history_indices, gains)
    else:
        history_indices = []
        ingredient_score = direct_score = parent_score = np.zeros(len(recipes))
    recommendations, metrics, scorable_counts = [], [], []
    for beta in GRID:
        scores = 0.5 * ingredient_score + 0.5 * ((1 - beta) * direct_score + beta * parent_score)
        ranked, scorable = rank(scores, excluded, fallback)
        scorable_counts.append(scorable)
        metrics.append(ndcg(ranked, targets, recipes))
        entries = [(index, score, source, *_trace(index, history_indices, families, beta))
                   for index, score, source in ranked]
        recommendations.append((beta, entries))
    return user, sum(rating >= 4 for rating in targets.values()), len(positive), metrics, scorable_counts, recommendations


def validate(data: Path = DATA, output: Path = OUTPUT, base: Path = BASE_OUTPUT) -> dict:
    data, output, base = Path(data).resolve(), Path(output).resolve(), Path(base).resolve()
    manifest, recipes, _ = load_fit_paths(data, output, base)
    train, validation = read_reviews(data, include_validation=True)
    require(len(train) == len(validation) == 4000 and sum(map(len, validation.values())) == 16000,
            "Unexpected validation cohort")
    totals, eligible = {beta: 0.0 for beta in GRID}, 0
    generated = datetime.now(timezone.utc).isoformat()
    grid_path = output / "validation_recommendations_by_grid.csv"
    per_user_path = output / "validation_per_user.csv"
    selected_rows = {beta: [] for beta in GRID}
    header = ("generated_utc", "beta", "user_id", "rank", "recipe_id", "score", "score_source",
              "matched_ingredient_ids", "matched_direct_foodon_ids", "matched_parent_foodon_ids")
    with grid_path.open("w", encoding="utf-8", newline="") as grid_stream, \
         per_user_path.open("w", encoding="utf-8", newline="") as user_stream:
        grid_writer = csv.writer(grid_stream, lineterminator="\n")
        user_writer = csv.writer(user_stream, lineterminator="\n")
        grid_writer.writerow(header)
        user_writer.writerow(("user_id", "positive_validation_targets", "positive_training_history",
                              "scorable_0", "scorable_0.25", "scorable_0.5",
                              "ndcg_0", "ndcg_0.25", "ndcg_0.5"))
        payloads = ((user, train[user], validation[user]) for user in sorted(train))
        with ProcessPoolExecutor(max_workers=6, initializer=_init_worker,
                                 initargs=(data, output, base)) as pool:
            for user, positive_targets, positive_history, metrics, counts, recommendations in pool.map(
                    _score_user, payloads, chunksize=4):
                eligible += positive_targets > 0
                for beta, entries in recommendations:
                    metric = metrics[GRID.index(beta)]
                    if metric is not None:
                        totals[beta] += metric
                    for position, (index, score, source, ingredient_ids, direct_ids, parent_ids) in enumerate(entries, 1):
                        row = (generated, beta, user, position, recipes[index],
                               "" if score is None else format(score, ".17g"), source,
                               json.dumps(ingredient_ids, separators=(",", ":")),
                               json.dumps(direct_ids, separators=(",", ":")),
                               json.dumps(parent_ids, separators=(",", ":")))
                        grid_writer.writerow(row)
                        selected_rows[beta].append(row)
                user_writer.writerow((user, positive_targets, positive_history, *counts,
                                      *("" if metric is None else format(metric, ".17g") for metric in metrics)))
    require(eligible > 0, "No positive validation users")
    means = {beta: totals[beta] / eligible for beta in GRID}
    selected = max(GRID, key=lambda beta: (means[beta], -beta))
    selected_path = output / "validation_recommendations_selected.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(selected_rows[selected])
    result = {
        "status": "validation_selected_no_test_access", "condition": "3_explicit_graph",
        "selected_beta": selected, "grid": list(GRID),
        "mean_validation_ndcg_at_10": {str(beta): means[beta] for beta in GRID},
        "users_with_positive_validation": eligible, "training_users": len(train),
        "candidate_catalog": len(recipes), "per_user_eligible_candidates": len(recipes) - 28,
        "fit_manifest_sha256": sha(output / "fit_manifest.json"),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "validation_per_user_sha256": sha(per_user_path),
        "validation_grid_recommendations_sha256": sha(grid_path),
        "validation_selected_recommendations_sha256": sha(selected_path),
        "generated_utc": generated,
        "note": "Beta selected using validation NDCG only; exact and hierarchy variants retained; no test rating read."
    }
    write_json(output / "validation_selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "validate"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    args = parser.parse_args()
    result = fit(args.data, args.output, args.base) if args.command == "fit" else validate(args.data, args.output, args.base)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
