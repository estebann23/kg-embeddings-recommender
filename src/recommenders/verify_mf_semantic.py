"""Audit semantic MF graph inputs, effective factors, rankings, and validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from src.recommenders.contract import DATA, ROOT, sha, verify_lock

CONFIG = ROOT / "config/rq1_mf_semantic.json"
OUTPUT = ROOT / "results/mf_semantic_hummus_4000_v1"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    report = json.loads((output / "validation_selection.json").read_text(encoding="utf-8"))
    check(report["status"] == "validation_selected_no_test_access", "Not validation selected")
    check(report["config_sha256"] == sha(CONFIG) == sha(output / "experiment_config.json"),
          "Configuration snapshot differs")
    check(report["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json"),
          "RQ1 contract lock differs")
    for name, digest in report["source_sha256"].items():
        check(name != "foodon_hierarchy" and sha(data / f"{name}.csv") == digest,
              f"Unexpected source {name}")
    for key, filename in (("model_sha256", "selected_model.npz"),
                          ("test_blind_top50_sha256", "test_blind_top50.csv"),
                          ("validation_per_user_sha256", "validation_per_user.csv")):
        check(report[key] == sha(output / filename), f"Saved {filename} changed")
    trials = report["grid_results"]
    check(len(trials) == 8 and len({(r["content_scale"], r["regularization"], r["epoch"])
                                    for r in trials}) == 8, "Incomplete hyperparameter grid")
    priority = lambda r: (r["validation_mean_ndcg_at_10"], -r["validation_rmse_clipped"],
                          -r["content_scale"], r["regularization"], -r["epoch"])
    check(max(trials, key=priority) == report["selected"], "Wrong selected checkpoint")
    with np.load(output / "selected_model.npz", allow_pickle=False) as model:
        state = {name: model[name].copy() for name in model.files}
    users, recipes = state["user_ids"].tolist(), state["recipe_ids"].tolist()
    ing_ids, con_ids = state["ingredient_ids"].tolist(), state["concept_ids"].tolist()
    check(len(users) == len(set(users)) == 4000 and
          len(recipes) == len(set(recipes)) == 75972, "Saved cohort IDs differ")
    check(len(ing_ids) == len(set(ing_ids)) == report["ingredient_features"] and
          len(con_ids) == len(set(con_ids)) == report["direct_concept_features"],
          "Saved feature vocabulary differs")
    require_shapes = {"user_factors": (4000, 16), "free_item_factors": (75972, 16),
                      "effective_item_factors": (75972, 16),
                      "ingredient_factors": (len(ing_ids), 16),
                      "concept_factors": (len(con_ids), 16),
                      "user_bias": (4000,), "item_bias": (75972,)}
    for name, shape in require_shapes.items():
        check(state[name].shape == shape and np.isfinite(state[name]).all(),
              f"Invalid {name}")
    check(float(state["content_scale"]) == report["selected"]["content_scale"],
          "Saved content scale differs")
    uid = {user: i for i, user in enumerate(users)}
    iid = {recipe: i for i, recipe in enumerate(recipes)}
    ing_index = {ingredient: i for i, ingredient in enumerate(ing_ids)}
    con_index = {concept: i for i, concept in enumerate(con_ids)}
    ingredient_sets = [set() for _ in recipes]
    for row in read_csv(data / "recipe_ingredients.csv"):
        check(row["evidence_state"] == "INFERRED", "Unverified ingredient path")
        ingredient_sets[iid[row["recipe_id"]]].add(row["ingredient_id"])
    concept_for_ingredient = defaultdict(set)
    for row in read_csv(data / "foodon_links.csv"):
        check(row["evidence_state"] == "INFERRED", "Unverified direct concept path")
        concept_for_ingredient[row["ingredient_id"]].add(row["foodon_iri"])
    concept_sets = [set().union(*(concept_for_ingredient[i] for i in ingredients))
                    for ingredients in ingredient_sets]
    check(set(ing_ids) == set().union(*ingredient_sets) and
          set(con_ids) == set().union(*concept_sets), "Feature vocabulary differs from graph")
    ip, ii = state["ingredient_indptr"], state["ingredient_indices"]
    cp, ci = state["concept_indptr"], state["concept_indices"]
    check(len(ip) == len(cp) == len(recipes) + 1 and
          ip[0] == cp[0] == 0 and ip[-1] == len(ii) and cp[-1] == len(ci),
          "Invalid incidence pointers")
    for i in range(len(recipes)):
        check({ing_ids[j] for j in ii[ip[i]:ip[i + 1]]} == ingredient_sets[i] and
              {con_ids[j] for j in ci[cp[i]:cp[i + 1]]} == concept_sets[i],
              f"Graph feature mismatch for {recipes[i]}")
    train = defaultdict(set)
    val = defaultdict(dict)
    counts = np.zeros(len(recipes), dtype=np.int32)
    positive_counts = np.zeros(len(recipes), dtype=np.int32)
    ratings = []
    for row in read_csv(data / "reviews.csv"):
        if row["split"] not in {"train", "validation"}:
            continue  # Test stars never parsed.
        u, i, rating = uid[row["user_id"]], iid[row["recipe_id"]], int(row["rating"])
        if row["split"] == "train":
            check(i not in train[u], "Repeated train pair")
            train[u].add(i)
            counts[i] += 1
            positive_counts[i] += rating >= 4
            ratings.append(rating)
        else:
            check(i not in val[u] and i not in train[u], "Held-out pair crosses training")
            val[u][i] = rating
    check(len(ratings) == 112000 and sum(map(len, val.values())) == 16000 and
          all(len(train[u]) == 28 and len(val[u]) == 4 for u in range(len(users))),
          "Source split counts differ")
    check(np.array_equal(counts, state["train_item_counts"]) and
          np.array_equal(positive_counts, state["positive_item_counts"]),
          "Saved training item counts differ")
    trained = counts > 0
    supervised_ing = np.zeros(len(ing_ids), dtype=bool)
    supervised_con = np.zeros(len(con_ids), dtype=bool)
    for i in np.flatnonzero(trained):
        supervised_ing[ii[ip[i]:ip[i + 1]]] = True
        supervised_con[ci[cp[i]:cp[i + 1]]] = True
    check(np.array_equal(supervised_ing, state["supervised_ingredients"]) and
          np.array_equal(supervised_con, state["supervised_concepts"]),
          "Supervised feature mask differs")
    check(np.all(state["ingredient_factors"][~supervised_ing] == 0) and
          np.all(state["concept_factors"][~supervised_con] == 0) and
          np.all(state["free_item_factors"][~trained] == 0) and
          np.all(state["item_bias"][~trained] == 0),
          "Unsupervised parameter has a value")
    ing_weights = np.repeat(np.divide(1.0, np.diff(ip),
                                      out=np.zeros(len(recipes)), where=np.diff(ip) > 0),
                            np.diff(ip))
    con_weights = np.repeat(np.divide(1.0, np.diff(cp),
                                      out=np.zeros(len(recipes)), where=np.diff(cp) > 0),
                            np.diff(cp))
    ing_matrix = sparse.csr_matrix((ing_weights.astype(np.float32), ii, ip),
                                   shape=(len(recipes), len(ing_ids)))
    con_matrix = sparse.csr_matrix((con_weights.astype(np.float32), ci, cp),
                                   shape=(len(recipes), len(con_ids)))
    effective = (state["free_item_factors"] + float(state["content_scale"]) * 0.5 *
                 (ing_matrix @ state["ingredient_factors"] +
                  con_matrix @ state["concept_factors"]))
    check(np.allclose(effective, state["effective_item_factors"], atol=2e-6, rtol=1e-6),
          "Saved effective factor differs from feature paths")
    semantic_evidence = np.zeros(len(recipes), dtype=bool)
    for i in range(len(recipes)):
        semantic_evidence[i] = bool(np.any(supervised_ing[ii[ip[i]:ip[i + 1]]]) or
                                    np.any(supervised_con[ci[cp[i]:cp[i + 1]]]))
    scorable = trained | semantic_evidence
    check(np.array_equal(scorable, state["scorable"]), "Scorable recipe mask differs")
    check(int((~scorable).sum()) == report["unscorable_recipes"] and
          int(((~trained) & scorable).sum()) == report["semantically_scorable_cold_recipes"],
          "Coverage report differs")
    ranking = read_csv(output / "test_blind_top50.csv")
    per_user = {uid[r["user_id"]]: r for r in read_csv(output / "validation_per_user.csv")}
    by_user = defaultdict(list)
    for row in ranking:
        by_user[uid[row["user_id"]]].append(row)
    check(len(ranking) == 200000 and len(by_user) == len(per_user) == 4000,
          "Saved ranking/metric cardinality differs")
    seen = np.flatnonzero(scorable)
    uf, ef = state["user_factors"], state["effective_item_factors"]
    ub, rb = state["user_bias"], state["item_bias"]
    mean = float(state["global_mean"])
    discounts = 1 / np.log2(np.arange(2, 52))
    ndcg10_total = ndcg50_total = 0.0
    eligible_users = 0
    for start in range(0, len(users), 64):
        end = min(start + 64, len(users))
        block = uf[start:end] @ ef[seen].T + rb[seen][None, :]
        block += ub[start:end, None] + mean
        for offset, user in enumerate(range(start, end)):
            scores = block[offset]
            scores[np.searchsorted(seen, list(train[user]))] = -np.inf
            cutoff = np.partition(scores, -50)[-50]
            eligible = np.flatnonzero(scores >= cutoff)
            expected = seen[eligible[np.lexsort((seen[eligible], -scores[eligible]))[:50]]]
            saved = by_user[user]
            actual = np.asarray([iid[row["recipe_id"]] for row in saved])
            check(len(saved) == 50 and np.array_equal(actual, expected),
                  f"Ranking differs for {users[user]}")
            check(all(int(row["rank"]) == j and row["score_source"] == "score"
                      for j, row in enumerate(saved, 1)), "Rank or score source differs")
            for row, item in zip(saved, actual):
                expected_score = float(mean + ub[user] + rb[item] + uf[user] @ ef[item])
                check(abs(float(row["score"]) - expected_score) < 2e-5,
                      "Saved score differs")
            gains = np.asarray([2 if val[user].get(i) == 5 else
                                1 if val[user].get(i) == 4 else 0 for i in actual])
            ideal = sorted((2 if rating == 5 else 1 for rating in val[user].values()
                            if rating >= 4), reverse=True)
            row = per_user[user]
            check(int(row["positive_validation_targets"]) == len(ideal) and
                  abs(float(row["precision_at_10"]) - sum(gains[:10] > 0) / 10) < 1e-12 and
                  abs(float(row["precision_at_50"]) - sum(gains > 0) / 50) < 1e-12,
                  "Validation positive count/precision differs")
            if ideal:
                eligible_users += 1
                idcg = sum(g / math.log2(j + 2) for j, g in enumerate(ideal))
                n10 = float(np.dot(gains[:10], discounts[:10]) / idcg)
                n50 = float(np.dot(gains, discounts) / idcg)
                check(abs(float(row["ndcg_at_10"]) - n10) < 1e-10 and
                      abs(float(row["ndcg_at_50"]) - n50) < 1e-10,
                      "Validation NDCG differs")
                ndcg10_total += n10
                ndcg50_total += n50
            else:
                check(row["ndcg_at_10"] == row["ndcg_at_50"] == "",
                      "Undefined NDCG not blank")
    check(eligible_users == report["positive_validation_users"] and
          abs(ndcg10_total / eligible_users - report["selected"]["validation_mean_ndcg_at_10"]) < 1e-10 and
          abs(ndcg50_total / eligible_users - report["selected_validation_mean_ndcg_at_50"]) < 1e-10,
          "Aggregate validation outcome differs")
    return {"status": "PASS", "source_graph_paths_checked": len(ii) + len(ci),
            "recipes_checked": len(recipes), "ranking_rows_checked": len(ranking),
            "semantic_cold_recipes_scorable": int(((~trained) & scorable).sum()),
            "remaining_unscorable": int((~scorable).sum()),
            "positive_validation_users": eligible_users,
            "mean_ndcg_at_10": ndcg10_total / eligible_users,
            "mean_ndcg_at_50": ndcg50_total / eligible_users}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.output), indent=2))
