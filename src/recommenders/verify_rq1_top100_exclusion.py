"""Independent result checks for the reduced-catalog RQ1 sensitivity experiment."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations

from src.recommenders.contract import DATA, ROOT, ROTATE, sha, verify_lock
from src.recommenders.item_cf import load_recipes
from src.recommenders.rq1_top100_exclusion import MODELS, OUTPUT, selected_recipes
from src.recommenders.rotate import load_factory


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=0, abs_tol=2e-12), (actual, expected)


def independent_gini(counts):
    ordered = sorted(counts)
    total = sum(ordered)
    n = len(ordered)
    cumulative = 0
    for count in ordered:
        cumulative += count
    assert cumulative == total
    running = 0
    cumulative_sum = 0
    for count in ordered:
        running += count
        cumulative_sum += running
    return (n + 1 - 2 * cumulative_sum / total) / n


def verify_rank_only():
    """Test-blind gate: inspect selection, saved rankings, and training IDs only."""
    verify_lock(DATA)
    pre = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    assert pre["status"] == "training_only_selection_passed"
    assert pre["selection_sha256"] == sha(OUTPUT / "excluded_top100_training_mean.csv")
    selected = list(rows(OUTPUT / "excluded_top100_training_mean.csv"))
    assert [(row["recipe_id"], int(row["training_rating_sum"]),
             int(row["training_rating_count"])) for row in selected] == selected_recipes(DATA)
    removed = {row["recipe_id"] for row in selected}
    train = defaultdict(set)
    for row in rows(DATA / "reviews.csv"):
        if row["split"] == "train":
            train[row["user_id"]].add(row["recipe_id"])
    assert len(train) == 4000 and all(len(value) == 28 for value in train.values())
    for model in MODELS:
        manifest = json.loads((OUTPUT / f"{model}_rank_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "reduced_catalog_ranking_complete"
        assert manifest["ranking_sha256"] == sha(OUTPUT / f"{model}_top50.csv")
        assert manifest["preflight_sha256"] == sha(OUTPUT / "preflight.json")
        assert manifest["users"] == 4000 and manifest["rows"] == 200000
        current_user, entries, seen = None, [], set()

        def check(user, user_rows):
            assert user in train and user not in seen and len(user_rows) == 50
            seen.add(user)
            assert [int(row["rank"]) for row in user_rows] == list(range(1, 51))
            recipes = [row["recipe_id"] for row in user_rows]
            assert len(set(recipes)) == 50 and not set(recipes) & (removed | train[user])

        for row in rows(OUTPUT / f"{model}_top50.csv"):
            assert row["condition"] == model
            if current_user is not None and row["user_id"] != current_user:
                check(current_user, entries)
                entries = []
            current_user = row["user_id"]
            entries.append(row)
        check(current_user, entries)
        assert len(seen) == 4000
    # CF and path scores do not depend on the candidate set: removal must preserve
    # their original ordering up to the first new item formerly below rank 50.
    static_prefixes = 0
    for model in MODELS[:2]:
        prior = ROOT / "results/rq1_accuracy_test" / f"{model}_top50.csv"
        old = defaultdict(list)
        new = defaultdict(list)
        for row in rows(prior):
            old[row["user_id"]].append(row["recipe_id"])
        for row in rows(OUTPUT / f"{model}_top50.csv"):
            new[row["user_id"]].append(row["recipe_id"])
        assert len(old) == len(new) == 4000
        for user in train:
            surviving = [recipe for recipe in old[user] if recipe not in removed]
            assert new[user][:len(surviving)] == surviving
            static_prefixes += 1
    # Recreate a few RotatE lists directly from the saved model, ranking the
    # full reduced embedded universe rather than using a top-50 cutoff.
    import numpy as np
    import torch
    from pykeen.losses import MarginRankingLoss
    from pykeen.models import RotatE
    from scipy.stats import rankdata

    sampled_rotate = 0
    config = json.loads(ROTATE.read_text(encoding="utf-8"))
    all_recipes = load_recipes(DATA)
    for variant, weight in (("primary", 0.5), ("hierarchy", 0.75)):
        factory = load_factory(DATA, variant)
        model = RotatE(triples_factory=factory,
                       embedding_dim=config["training"]["embedding_dim"],
                       loss=MarginRankingLoss(margin=config["training"]["margin"]),
                       random_seed=config["training"]["seed"])
        base = ROOT / f"results/rotate_{variant}_full"
        model.load_state(base / "model.pt")
        model.eval()
        embedded = [recipe for recipe in all_recipes if recipe in factory.entity_to_id]
        tail_ids = torch.tensor([factory.entity_to_id[recipe] for recipe in embedded],
                                dtype=torch.long)
        sample_users = sorted(train)[:3]
        saved = defaultdict(list)
        for row in rows(OUTPUT / f"4_rotate_{variant}_top50.csv"):
            if row["user_id"] in sample_users:
                saved[row["user_id"]].append(row)
        with torch.inference_mode():
            for user in sample_users:
                eligible = np.array([i for i, recipe in enumerate(embedded)
                                     if recipe not in removed and recipe not in train[user]])
                scores = []
                for relation in ("RATED_4", "RATED_5"):
                    query = torch.tensor([[factory.entity_to_id[user],
                                           factory.relation_to_id[relation]]], dtype=torch.long)
                    raw = model.predict_t(query, tails=tail_ids).reshape(-1).cpu().numpy()
                    scores.append(rankdata(raw[eligible], method="average") / len(eligible))
                combined = (1 - weight) * scores[0] + weight * scores[1]
                ordering = np.lexsort((eligible, -combined))[:50]
                expected = [embedded[eligible[position]] for position in ordering]
                assert [row["recipe_id"] for row in saved[user]] == expected
                for row, position in zip(saved[user], ordering):
                    close(row["score"], float(combined[position]))
                sampled_rotate += 1
    return {"status": "reduced_catalog_rankings_verified_test_blind",
            "selection_rows": len(selected), "ranking_rows": 800000,
            "users_per_model": 4000, "static_order_prefixes_checked": static_prefixes,
            "rotate_lists_recomputed": sampled_rotate}


def verify():
    assert verify_rank_only()["status"] == "reduced_catalog_rankings_verified_test_blind"
    verify_lock(DATA)
    manifest = json.loads((OUTPUT / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "reduced_catalog_evaluation_complete"
    assert manifest["models"] == list(MODELS) and manifest["k"] == [10, 50]
    assert manifest["selection_sha256"] == sha(OUTPUT / "excluded_top100_training_mean.csv")
    assert manifest["per_user_sha256"] == sha(OUTPUT / "per_user_metrics.csv")
    assert manifest["aggregate_sha256"] == sha(OUTPUT / "aggregate_metrics.csv")
    assert manifest["exposure_sha256"] == sha(OUTPUT / "recipe_exposure.csv")
    selected = list(rows(OUTPUT / "excluded_top100_training_mean.csv"))
    assert len(selected) == 100
    assert [(row["recipe_id"], int(row["training_rating_sum"]),
             int(row["training_rating_count"])) for row in selected] == selected_recipes(DATA)
    removed = {row["recipe_id"] for row in selected}
    catalog = [recipe for recipe in load_recipes(DATA) if recipe not in removed]
    assert len(catalog) == 75872
    ingredients = defaultdict(set)
    for row in rows(DATA / "recipe_ingredients.csv"):
        ingredients[row["recipe_id"]].add(row["ingredient_id"])
    targets = defaultdict(dict)
    train = defaultdict(set)
    removed_targets = removed_positive = 0
    for row in rows(DATA / "reviews.csv"):
        user, recipe = row["user_id"], row["recipe_id"]
        if row["split"] == "train":
            train[user].add(recipe)
        elif row["split"] == "test":
            if recipe in removed:
                removed_targets += 1
                removed_positive += int(row["rating"]) >= 4
            else:
                targets[user][recipe] = int(row["rating"])
    assert len(train) == 4000 and all(len(items) == 28 for items in train.values())
    assert manifest["removed_test_targets"] == removed_targets
    assert manifest["removed_positive_test_targets"] == removed_positive
    assert manifest["positive_test_users"] == sum(
        any(rating >= 4 for rating in targets[user].values()) for user in train)
    saved_user = {(row["model"], row["user_id"]): row
                  for row in rows(OUTPUT / "per_user_metrics.csv")}
    assert len(saved_user) == 16000
    saved_exposure = {(model, k): Counter() for model in MODELS for k in (10, 50)}
    sampled = 0
    for model in MODELS:
        path = OUTPUT / f"{model}_top50.csv"
        rank_manifest = json.loads((OUTPUT / f"{model}_rank_manifest.json").read_text(encoding="utf-8"))
        assert rank_manifest["ranking_sha256"] == manifest["rank_sha256"][model] == sha(path)
        current_user, current = None, []
        seen = set()

        def check_user(user, entries):
            nonlocal sampled
            assert user in train and user not in seen and len(entries) == 50
            seen.add(user)
            assert [int(entry["rank"]) for entry in entries] == list(range(1, 51))
            ranked = [entry["recipe_id"] for entry in entries]
            assert len(set(ranked)) == 50
            assert not set(ranked) & (removed | train[user])
            expected_positives = sum(rating >= 4 for rating in targets[user].values())
            saved = saved_user[model, user]
            assert int(saved["retained_test_targets"]) == len(targets[user])
            assert int(saved["retained_positive_targets"]) == expected_positives
            assert int(saved["scorable_candidates"]) == int(entries[0]["scorable_candidates"])
            assert int(saved["fallback_top50"]) == sum(entry["score_source"] == "fallback"
                                                      for entry in entries)
            ideal_gains = sorted((2 if rating == 5 else 1 for rating in targets[user].values()
                                  if rating >= 4), reverse=True)
            for k in (10, 50):
                top = ranked[:k]
                hits = sum(targets[user].get(recipe, 0) >= 4 for recipe in top)
                actual_dcg = sum((2 if targets[user].get(recipe) == 5 else
                                  1 if targets[user].get(recipe) == 4 else 0) /
                                 math.log2(position + 2)
                                 for position, recipe in enumerate(top))
                ideal_dcg = sum(gain / math.log2(position + 2)
                                for position, gain in enumerate(ideal_gains[:k]))
                assert int(saved[f"hits_at_{k}"]) == hits
                close(saved[f"precision_at_{k}"], hits / k)
                if ideal_dcg:
                    close(saved[f"ndcg_at_{k}"], actual_dcg / ideal_dcg)
                else:
                    assert saved[f"ndcg_at_{k}"] == ""
                saved_exposure[model, k].update(top)
                assert int(saved[f"mapped_positions_at_{k}"]) == sum(
                    bool(ingredients.get(recipe)) for recipe in top)
                n_mapped = int(saved[f"mapped_positions_at_{k}"])
                assert int(saved[f"measurable_pairs_at_{k}"]) == n_mapped * (n_mapped - 1) // 2
                if len(seen) <= 100:
                    pairs = [(ingredients[left], ingredients[right])
                             for left, right in combinations(top, 2)
                             if ingredients.get(left) and ingredients.get(right)]
                    if pairs:
                        expected = math.fsum(len(left ^ right) / len(left | right)
                                             for left, right in pairs) / len(pairs)
                        close(saved[f"ingredient_ild_at_{k}"], expected)
                    else:
                        assert saved[f"ingredient_ild_at_{k}"] == ""
                    sampled += 1

        for entry in rows(path):
            if current_user is not None and entry["user_id"] != current_user:
                check_user(current_user, current)
                current = []
            current_user = entry["user_id"]
            current.append(entry)
        check_user(current_user, current)
        assert len(seen) == 4000 and len(saved_exposure[model, 50]) <= len(catalog)
    saved_exposure_rows = 0
    expected_exposure_order = iter((model, k, recipe) for model in MODELS
                                   for k in (10, 50) for recipe in catalog)
    for row in rows(OUTPUT / "recipe_exposure.csv"):
        model, k, recipe = row["model"], int(row["k"]), row["recipe_id"]
        assert (model, k, recipe) == next(expected_exposure_order)
        assert int(row["exposure_count"]) == saved_exposure[model, k][recipe]
        saved_exposure_rows += 1
    assert saved_exposure_rows == 8 * len(catalog)
    assert next(expected_exposure_order, None) is None
    aggregate = list(rows(OUTPUT / "aggregate_metrics.csv"))
    assert len(aggregate) == 8
    for row in aggregate:
        model, k = row["model"], int(row["k"])
        values = [saved_user[model, user] for user in sorted(train)]
        positive = [value for value in values if int(value["retained_positive_targets"]) > 0]
        counts = [saved_exposure[model, k][recipe] for recipe in catalog]
        assert int(row["users"]) == 4000
        assert int(row["positive_test_users"]) == len(positive)
        assert int(row["retained_test_targets"]) == sum(
            int(value["retained_test_targets"]) for value in values)
        assert int(row["retained_positive_targets"]) == sum(
            int(value["retained_positive_targets"]) for value in values)
        assert int(row["observed_positive_hits"]) == sum(
            int(value[f"hits_at_{k}"]) for value in values)
        close(row["precision"], math.fsum(float(value[f"precision_at_{k}"])
                                           for value in values) / 4000)
        if positive:
            close(row["ndcg_positive_users"], math.fsum(float(value[f"ndcg_at_{k}"])
                                                          for value in positive) / len(positive))
        defined = [float(value[f"ingredient_ild_at_{k}"]) for value in values
                   if value[f"ingredient_ild_at_{k}"] != ""]
        assert int(row["defined_ild_users"]) == len(defined)
        close(row["mean_user_ingredient_ild"], math.fsum(defined) / len(defined))
        measurable = sum(int(value[f"measurable_pairs_at_{k}"]) for value in values)
        mapped = sum(int(value[f"mapped_positions_at_{k}"]) for value in values)
        possible = 4000 * k * (k - 1) // 2
        assert int(row["measurable_pairs"]) == measurable
        assert int(row["possible_pairs"]) == possible
        close(row["pair_coverage"], measurable / possible)
        assert int(row["mapped_positions"]) == mapped
        close(row["ingredient_position_coverage"], mapped / (4000 * k))
        distinct = sum(count > 0 for count in counts)
        assert sum(counts) == 4000 * k
        assert int(row["distinct_exposed_recipes"]) == distinct
        close(row["catalog_coverage"], distinct / len(catalog))
        close(row["exposure_gini"], independent_gini(counts))
        assert int(row["max_recipe_exposure"]) == max(counts)
        top_count = math.ceil(len(catalog) * 0.01)
        close(row["top_1_percent_exposure_share"],
              sum(sorted(counts, reverse=True)[:top_count]) / (4000 * k))
    comparison = json.loads((OUTPUT / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "paired_comparisons_complete"
    assert comparison["new_evaluation_manifest_sha256"] == sha(OUTPUT / "evaluation_manifest.json")
    comparison_files = (("paired_model_sha256", "paired_model_differences.csv", 36),
                        ("paired_full_catalog_sha256", "paired_full_catalog_differences.csv", 24),
                        ("prior_removed_exposure_sha256", "removed_recipes_in_prior_lists.csv", 8))
    for key, filename, expected_count in comparison_files:
        path = OUTPUT / filename
        assert comparison[key] == sha(path) and sum(1 for _ in rows(path)) == expected_count
    for row in rows(OUTPUT / "paired_model_differences.csv"):
        assert int(row["wins_a"]) + int(row["ties"]) + int(row["wins_b"]) == int(row["paired_users"])
    for row in rows(OUTPUT / "paired_full_catalog_differences.csv"):
        assert int(row["improved"]) + int(row["unchanged"]) + int(row["worsened"]) == int(row["paired_users"])
    return {"status": "reduced_catalog_results_verified", "selection_rows": len(selected),
            "ranking_rows": 800000, "per_user_rows": len(saved_user),
            "exposure_rows": saved_exposure_rows,
            "sampled_ild_lists_recomputed": sampled,
            "aggregate_rows": len(aggregate), "comparison_rows": 68}


if __name__ == "__main__":
    import sys
    print(json.dumps(verify_rank_only() if sys.argv[1:] == ["rank-only"] else verify(),
                     indent=2))
