"""RQ1 condition 2: item CF fused with flat audited recipe attributes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf import (OUTPUT as BASE_OUTPUT, GAINS, WEIGHTS,
                                      load_fit, load_recipes, read_reviews, require)
from src.recommenders.rotate_ranking import recipe_order


OUTPUT = ROOT / "results/item_cf_attributes_hummus_4000_v1"
GRID = (0.0, 0.25, 0.5, 0.75)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def feature_matrix(recipes: list[str], features: dict[str, set[str]]) -> sparse.csr_matrix:
    feature_ids = {feature: col for col, feature in enumerate(sorted(set().union(*features.values())))} if features else {}
    recipe_index = {recipe: row for row, recipe in enumerate(recipes)}
    row_indices, col_indices = [], []
    for recipe, values in features.items():
        for value in values:
            row_indices.append(recipe_index[recipe])
            col_indices.append(feature_ids[value])
    matrix = sparse.coo_matrix((np.ones(len(row_indices), dtype=np.int32),
                                (row_indices, col_indices)),
                               shape=(len(recipes), len(feature_ids))).tocsr()
    require(matrix.nnz == len(row_indices), "Duplicate feature entry")
    return matrix


def fit(data: Path = DATA, output: Path = OUTPUT, base: Path = BASE_OUTPUT) -> dict:
    data, output, base = Path(data).resolve(), Path(output).resolve(), Path(base).resolve()
    verify_lock(data)
    contract = read_json(ROOT / "config/rq1_experiment.json")
    require(contract["conditions"]["2_item_cf_attributes"]["validation_grid"]["alpha"] == list(GRID),
            "Alpha grid changed")
    baseline_selection = read_json(base / "validation_selection.json")
    base_manifest, _, _ = load_fit(data, base)
    n = baseline_selection["selected_n"]
    require(baseline_selection["fit_manifest_sha256"] == sha(base / "fit_manifest.json") and
            baseline_selection["status"] == "validation_selected_no_test_access" and
            n in contract["conditions"]["1_item_cf"]["validation_grid"]["N"],
            "Baseline setting or fit changed")
    recipes = load_recipes(data)
    recipe_set = set(recipes)
    ingredients = defaultdict(set)
    with (data / "recipe_ingredients.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["recipe_id"] in recipe_set and row["evidence_state"] == "INFERRED",
                    "Unexpected mapped ingredient evidence")
            ingredients[row["recipe_id"]].add(row["ingredient_id"])
    ingredient_concepts = defaultdict(set)
    with (data / "foodon_links.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["evidence_state"] == "INFERRED" and row["foodon_iri"].startswith(
                "http://purl.obolibrary.org/obo/FOODON_"), "Unexpected direct FoodOn evidence")
            ingredient_concepts[row["ingredient_id"]].add(row["foodon_iri"])
    concepts = {recipe: set().union(*(ingredient_concepts[ingredient] for ingredient in values))
                for recipe, values in ingredients.items()}
    ingredient_matrix = feature_matrix(recipes, ingredients)
    concept_matrix = feature_matrix(recipes, concepts)
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "ingredient_binary.npz", ingredient_matrix)
    sparse.save_npz(output / "direct_foodon_binary.npz", concept_matrix)
    write_json(output / "recipe_order.json", recipes)
    result = {
        "status": "fitted_static_attributes_and_baseline_cf", "condition": "2_item_cf_attributes",
        "dataset_id": data.name, "selected_baseline_n": n, "grid": list(GRID),
        "recipes": len(recipes), "recipes_with_mapped_ingredient": int(np.count_nonzero(ingredient_matrix.getnnz(axis=1))),
        "recipes_with_direct_foodon": int(np.count_nonzero(concept_matrix.getnnz(axis=1))),
        "ingredient_id_count": ingredient_matrix.shape[1], "direct_foodon_id_count": concept_matrix.shape[1],
        "ingredient_incidence_count": ingredient_matrix.nnz, "direct_foodon_incidence_count": concept_matrix.nnz,
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "base_fit_manifest_sha256": sha(base / "fit_manifest.json"),
        "base_validation_selection_sha256": sha(base / "validation_selection.json"),
        "recipe_ingredients_sha256": sha(data / "recipe_ingredients.csv"),
        "foodon_links_sha256": sha(data / "foodon_links.csv"),
        "recipe_order_sha256": sha(output / "recipe_order.json"),
        "ingredient_matrix_sha256": sha(output / "ingredient_binary.npz"),
        "direct_foodon_matrix_sha256": sha(output / "direct_foodon_binary.npz"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Static mapped ingredient and direct FoodOn identity sets; no hierarchy, nutrition, or held-out rating in fit."
    }
    require(base_manifest["catalog_recipes"] == result["recipes"], "Baseline catalog mismatch")
    write_json(output / "fit_manifest.json", result)
    return result


def load_attributes(data: Path, output: Path, base: Path):
    verify_lock(data)
    manifest = read_json(output / "fit_manifest.json")
    require(manifest["status"] == "fitted_static_attributes_and_baseline_cf" and
            manifest["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json") and
            manifest["base_fit_manifest_sha256"] == sha(base / "fit_manifest.json") and
            manifest["base_validation_selection_sha256"] == sha(base / "validation_selection.json") and
            manifest["recipe_ingredients_sha256"] == sha(data / "recipe_ingredients.csv") and
            manifest["foodon_links_sha256"] == sha(data / "foodon_links.csv"), "Feature source mismatch")
    for key, filename in (("recipe_order_sha256", "recipe_order.json"),
                          ("ingredient_matrix_sha256", "ingredient_binary.npz"),
                          ("direct_foodon_matrix_sha256", "direct_foodon_binary.npz")):
        require(manifest[key] == sha(output / filename), "Feature artifact changed")
    recipes = read_json(output / "recipe_order.json")
    require(recipes == load_recipes(data), "Feature recipe order changed")
    ingredients = sparse.load_npz(output / "ingredient_binary.npz").tocsr()
    concepts = sparse.load_npz(output / "direct_foodon_binary.npz").tocsr()
    require(ingredients.shape[0] == concepts.shape[0] == len(recipes), "Feature rows changed")
    return manifest, recipes, ingredients, concepts


def jaccard_rows(matrix: sparse.csr_matrix, history_indices: list[int], sizes: np.ndarray) -> np.ndarray:
    intersection = (matrix[history_indices] @ matrix.T).toarray().astype(np.float64)
    union = sizes[history_indices, None] + sizes[None, :] - intersection
    np.divide(intersection, union, out=intersection, where=union > 0)
    return intersection


def rank_scores(similarity: np.ndarray, weights: np.ndarray, excluded: np.ndarray,
                fallback_indices: list[int], n: int, k: int = 10):
    """Stable top-N history choice followed by signed weighted scoring."""
    require(similarity.shape[0] == len(weights) and n > 0, "Invalid score shape")
    work = similarity.copy()
    work[weights == 0] = 0
    work[:, excluded] = 0
    # Stable sort preserves numeric history-ID order when similarities tie.
    chosen = np.argsort(-work, axis=0, kind="stable")[:min(n, work.shape[0])]
    values = np.take_along_axis(work, chosen, axis=0)
    selected_weights = weights[chosen]
    numerator = np.sum(values * selected_weights, axis=0)
    denominator = np.sum(values * np.abs(selected_weights), axis=0)
    neighbor_count = np.count_nonzero(values > 0, axis=0)
    scores = np.full(work.shape[1], -np.inf)
    np.divide(numerator, denominator, out=scores, where=denominator > 0)
    scorable = np.flatnonzero(denominator > 0)
    if len(scorable) >= k:
        threshold = np.partition(scores[scorable], -k)[-k]
        above = scorable[scores[scorable] > threshold]
        above = above[np.argsort(-scores[above], kind="stable")]
        equal = scorable[scores[scorable] == threshold]
        top = np.concatenate((above, equal[:k - len(above)]))
    else:
        top = scorable[np.argsort(-scores[scorable], kind="stable")]
    ranked = [(int(index), float(scores[index]), "score", int(neighbor_count[index])) for index in top]
    if len(ranked) < k:
        for index in fallback_indices:
            if index not in excluded and not np.isfinite(scores[index]):
                ranked.append((index, None, "fallback", 0))
                if len(ranked) == k:
                    break
    require(len(ranked) == k and len({entry[0] for entry in ranked}) == k, "Invalid top-K")
    return ranked, len(scorable)


def ndcg(ranked, validation: dict[str, int], recipes: list[str]) -> float | None:
    positive = sorted((GAINS[rating] for rating in validation.values() if GAINS[rating] > 0), reverse=True)
    if not positive:
        return None
    ideal = sum(value / math.log2(position + 2) for position, value in enumerate(positive))
    actual = sum(GAINS[validation.get(recipes[index], 1)] / math.log2(position + 2)
                 for position, (index, _, _, _) in enumerate(ranked))
    return actual / ideal


_WORKER = None


def _init_worker(data: Path, output: Path, base: Path) -> None:
    global _WORKER
    manifest, recipes, ingredients, concepts = load_attributes(data, output, base)
    _, adjacency, popularity = load_fit(data, base)
    _WORKER = (manifest, recipes, ingredients, concepts, adjacency,
               {recipe: index for index, recipe in enumerate(recipes)},
               sorted(range(len(recipes)), key=lambda index: (-popularity[recipes[index]], index)),
               np.asarray(ingredients.getnnz(axis=1), dtype=np.float64),
               np.asarray(concepts.getnnz(axis=1), dtype=np.float64))


def _score_user(payload):
    user, history, targets = payload
    manifest, recipes, ingredients, concepts, adjacency, recipe_index, fallback, ingredient_sizes, concept_sizes = _WORKER
    history = sorted(history, key=lambda entry: recipe_order(entry[0]))
    history_indices = [recipe_index[recipe] for recipe, _ in history]
    weights = np.asarray([WEIGHTS[rating] for _, rating in history], dtype=np.float64)
    excluded = np.asarray(history_indices, dtype=np.int64)
    ingredient_j = jaccard_rows(ingredients, history_indices, ingredient_sizes)
    concept_j = jaccard_rows(concepts, history_indices, concept_sizes)
    attribute = 0.5 * (ingredient_j + concept_j)
    cf = np.zeros_like(attribute)
    for row_index, (recipe, _) in enumerate(history):
        for candidate, similarity in adjacency.get(recipe, ()):
            cf[row_index, recipe_index[candidate]] = similarity
    metrics, scorable_counts, recommendations = [], [], []
    for alpha in GRID:
        similarity = (1 - alpha) * cf + alpha * attribute
        ranked, scorable = rank_scores(similarity, weights, excluded, fallback,
                                        manifest["selected_baseline_n"])
        scorable_counts.append(scorable)
        metrics.append(ndcg(ranked, targets, recipes))
        recommendations.append((alpha, ranked))
    return user, sum(rating >= 4 for rating in targets.values()), metrics, scorable_counts, recommendations


def validate(data: Path = DATA, output: Path = OUTPUT, base: Path = BASE_OUTPUT) -> dict:
    data, output, base = Path(data).resolve(), Path(output).resolve(), Path(base).resolve()
    manifest, recipes, _, _ = load_attributes(data, output, base)
    train, validation = read_reviews(data, include_validation=True)
    require(len(train) == len(validation) == 4000 and sum(map(len, validation.values())) == 16000,
            "Unexpected validation cohort")
    baseline_rows = {}
    with (base / "validation_recommendations_selected.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            baseline_rows[(row["user_id"], int(row["rank"]))] = row["recipe_id"]
    require(len(baseline_rows) == 40000, "Baseline validation list incomplete")
    n = manifest["selected_baseline_n"]
    totals = {alpha: 0.0 for alpha in GRID}
    eligible = 0
    generated = datetime.now(timezone.utc).isoformat()
    grid_path = output / "validation_recommendations_by_grid.csv"
    per_user_path = output / "validation_per_user.csv"
    selected_rows = {alpha: [] for alpha in GRID}
    with grid_path.open("w", encoding="utf-8", newline="") as grid_stream, \
         per_user_path.open("w", encoding="utf-8", newline="") as user_stream:
        grid_writer, user_writer = csv.writer(grid_stream, lineterminator="\n"), csv.writer(user_stream, lineterminator="\n")
        grid_writer.writerow(("generated_utc", "alpha", "neighbor_n", "user_id", "rank", "recipe_id",
                              "score", "score_source", "contributing_neighbors"))
        user_writer.writerow(("user_id", "positive_validation_targets", "scorable_0", "scorable_0.25",
                              "scorable_0.5", "scorable_0.75", "ndcg_0", "ndcg_0.25", "ndcg_0.5", "ndcg_0.75"))
        payloads = ((user, train[user], validation[user]) for user in sorted(train))
        with ProcessPoolExecutor(max_workers=6, initializer=_init_worker,
                                 initargs=(data, output, base)) as pool:
            for user, positives, metrics, scorable_counts, recommendations in pool.map(
                    _score_user, payloads, chunksize=4):
                eligible += positives > 0
                for alpha, ranked in recommendations:
                    metric = metrics[GRID.index(alpha)]
                    if metric is not None:
                        totals[alpha] += metric
                    for rank, (index, score, source, count) in enumerate(ranked, 1):
                        recipe = recipes[index]
                        if alpha == 0:
                            require(recipe == baseline_rows[(user, rank)], "Alpha-zero baseline reproduction failed")
                        row = (generated, alpha, n, user, rank, recipe,
                               "" if score is None else format(score, ".17g"), source, count)
                        grid_writer.writerow(row)
                        selected_rows[alpha].append(row)
                user_writer.writerow((user, positives, *scorable_counts,
                                      *("" if value is None else format(value, ".17g") for value in metrics)))
    require(eligible > 0, "No positive validation users")
    means = {alpha: totals[alpha] / eligible for alpha in GRID}
    selected = max(GRID, key=lambda alpha: (means[alpha], -alpha))
    selected_path = output / "validation_recommendations_selected.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("generated_utc", "alpha", "neighbor_n", "user_id", "rank", "recipe_id",
                         "score", "score_source", "contributing_neighbors"))
        writer.writerows(selected_rows[selected])
    result = {
        "status": "validation_selected_no_test_access", "condition": "2_item_cf_attributes",
        "selected_alpha": selected, "selected_n_from_condition_1": n, "grid": list(GRID),
        "mean_validation_ndcg_at_10": {str(alpha): means[alpha] for alpha in GRID},
        "users_with_positive_validation": eligible, "training_users": len(train),
        "candidate_catalog": len(recipes), "per_user_eligible_candidates": len(recipes) - 28,
        "fit_manifest_sha256": sha(output / "fit_manifest.json"),
        "base_validation_selection_sha256": sha(base / "validation_selection.json"),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "validation_per_user_sha256": sha(per_user_path),
        "validation_grid_recommendations_sha256": sha(grid_path),
        "validation_selected_recommendations_sha256": sha(selected_path),
        "generated_utc": generated,
        "note": "Validation-selected alpha only. Alpha=0 reproduces condition 1; no test rating was read."
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
