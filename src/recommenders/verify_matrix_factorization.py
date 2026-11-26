"""Recompute saved MF ranking and validation metrics from the selected factors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.recommenders.contract import DATA, ROOT, sha, verify_lock

OUTPUT = ROOT / "results/matrix_factorization_hummus_4000_v1"
CONFIG = ROOT / "config/rq1_matrix_factorization.json"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    report = json.loads((output / "validation_selection.json").read_text(encoding="utf-8"))
    check(report["status"] == "validation_selected_no_test_access", "MF not selected")
    check(report["config_sha256"] == sha(CONFIG) == sha(output / "experiment_config.json"),
          "MF configuration differs")
    check(report["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json"),
          "Frozen RQ1 lock differs")
    for key, filename in (("reviews_sha256", "reviews.csv"),
                          ("recipes_sha256", "recipes.csv")):
        check(report[key] == sha(data / filename), f"Source {filename} differs")
    for key, filename in (("model_sha256", "selected_model.npz"),
                          ("validation_top50_sha256", "validation_top50.csv"),
                          ("test_blind_top50_sha256", "test_blind_top50.csv"),
                          ("validation_per_user_sha256", "validation_per_user.csv")):
        check(report[key] == sha(output / filename), f"Output {filename} differs")
    check((output / "validation_top50.csv").read_bytes() ==
          (output / "test_blind_top50.csv").read_bytes(), "Test-blind list differs")
    trials = report["grid_results"]
    check(len(trials) == 8 and len({(x["factors"], x["regularization"], x["epoch"])
                                    for x in trials}) == 8, "Incomplete or repeated grid")
    key = lambda x: (x["validation_mean_ndcg_at_10"], -x["validation_rmse_clipped"],
                     -x["factors"], x["regularization"], -x["epoch"])
    check(report["selected"] == max(trials, key=key), "Incorrect validation selection")
    with np.load(output / "selected_model.npz", allow_pickle=False) as state:
        users = state["user_ids"].tolist()
        recipes = state["recipe_ids"].tolist()
        uf, rf = state["user_factors"], state["item_factors"]
        ub, rb = state["user_bias"], state["item_bias"]
        mean = float(state["global_mean"])
        item_counts = state["train_item_counts"]
        positive_counts = state["positive_item_counts"]
    check(len(users) == len(set(users)) == 4000 and len(recipes) == len(set(recipes)) == 75972,
          "Saved universe changed")
    check(uf.shape == (4000, report["selected"]["factors"]) and
          rf.shape == (75972, report["selected"]["factors"]) and
          ub.shape == (4000,) and rb.shape == (75972,), "Model shape differs")
    check(all(np.isfinite(x).all() for x in (uf, rf, ub, rb)) and math.isfinite(mean),
          "Non-finite selected model")
    uid = {user: i for i, user in enumerate(users)}
    iid = {recipe: i for i, recipe in enumerate(recipes)}
    train = defaultdict(set)
    validation = defaultdict(dict)
    expected_count = np.zeros(len(recipes), dtype=np.int32)
    expected_positive = np.zeros(len(recipes), dtype=np.int32)
    train_ratings = []
    for row in rows(data / "reviews.csv"):
        split = row["split"]
        if split not in {"train", "validation"}:
            continue  # Never parse test rating values.
        u, i, rating = uid[row["user_id"]], iid[row["recipe_id"]], int(row["rating"])
        if split == "train":
            check(i not in train[u], "Duplicate training pair")
            train[u].add(i)
            expected_count[i] += 1
            expected_positive[i] += rating >= 4
            train_ratings.append(rating)
        else:
            check(i not in validation[u] and i not in train[u], "Duplicate validation pair")
            validation[u][i] = rating
    check(len(train_ratings) == 112000 and sum(map(len, validation.values())) == 16000,
          "Source split counts differ")
    check(all(len(train[u]) == 28 and len(validation[u]) == 4 for u in range(len(users))),
          "Per-user source counts differ")
    check(np.array_equal(item_counts, expected_count) and
          np.array_equal(positive_counts, expected_positive), "Stored item counts differ")
    check(abs(mean - np.mean(train_ratings)) < 1e-5, "Global mean differs")
    cold = item_counts == 0
    check(np.all(rf[cold] == 0) and np.all(rb[cold] == 0),
          "Cold recipes have learned factors")
    ranked_rows = rows(output / "test_blind_top50.csv")
    metric_rows = rows(output / "validation_per_user.csv")
    check(len(ranked_rows) == 200000 and len(metric_rows) == 4000,
          "Recommendation or metric row count differs")
    by_user = defaultdict(list)
    for row in ranked_rows:
        by_user[uid[row["user_id"]]].append(row)
    check(len(by_user) == 4000, "Missing user ranking")
    saved_metrics = {uid[row["user_id"]]: row for row in metric_rows}
    check(len(saved_metrics) == 4000, "Missing user metric")
    seen = np.flatnonzero(~cold)
    discount = 1 / np.log2(np.arange(2, 52))
    ndcg10_sum = 0.0
    ndcg50_sum = 0.0
    positive_users = 0
    for start in range(0, len(users), 64):
        stop = min(start + 64, len(users))
        scores = uf[start:stop] @ rf[seen].T + rb[seen][None, :]
        scores += ub[start:stop, None] + mean
        for offset, u in enumerate(range(start, stop)):
            score = scores[offset]
            score[np.searchsorted(seen, list(train[u]))] = -np.inf
            cutoff = np.partition(score, -50)[-50]
            contenders = np.flatnonzero(score >= cutoff)
            expected = seen[contenders[np.lexsort((seen[contenders], -score[contenders]))[:50]]]
            saved = by_user[u]
            actual = np.asarray([iid[row["recipe_id"]] for row in saved], dtype=np.int64)
            check(len(saved) == 50 and np.array_equal(actual, expected),
                  f"Top-50 order differs for {users[u]}")
            check(all(int(row["rank"]) == j and row["score_source"] == "score"
                      for j, row in enumerate(saved, 1)), "Rank or source differs")
            check(not set(actual) & train[u], "Training item recommended")
            for row, i in zip(saved, actual):
                predicted = float(mean + ub[u] + rb[i] + uf[u] @ rf[i])
                check(abs(float(row["score"]) - predicted) < 2e-5,
                      "Saved recommendation score differs")
            gains = np.array([2 if validation[u].get(i) == 5 else
                              1 if validation[u].get(i) == 4 else 0 for i in actual])
            ideal = sorted((2 if r == 5 else 1 for r in validation[u].values() if r >= 4),
                           reverse=True)
            saved_metric = saved_metrics[u]
            check(int(saved_metric["positive_validation_targets"]) == len(ideal),
                  "Positive target count differs")
            check(abs(float(saved_metric["precision_at_10"]) - sum(gains[:10] > 0) / 10) < 1e-12 and
                  abs(float(saved_metric["precision_at_50"]) - sum(gains > 0) / 50) < 1e-12,
                  "Precision differs")
            if ideal:
                positive_users += 1
                denominator = sum(g / math.log2(j + 2) for j, g in enumerate(ideal))
                n10 = float(np.dot(gains[:10], discount[:10]) / denominator)
                n50 = float(np.dot(gains, discount) / denominator)
                check(abs(float(saved_metric["ndcg_at_10"]) - n10) < 1e-10 and
                      abs(float(saved_metric["ndcg_at_50"]) - n50) < 1e-10,
                      "NDCG differs")
                ndcg10_sum += n10
                ndcg50_sum += n50
            else:
                check(saved_metric["ndcg_at_10"] == saved_metric["ndcg_at_50"] == "",
                      "Undefined NDCG must be blank")
    check(positive_users == report["positive_validation_users"], "Eligible user count differs")
    check(abs(ndcg10_sum / positive_users - report["selected"]["validation_mean_ndcg_at_10"]) < 1e-10,
          "Selected validation NDCG@10 differs")
    check(abs(ndcg50_sum / positive_users - report["selected_validation_mean_ndcg_at_50"]) < 1e-10,
          "Selected validation NDCG@50 differs")
    return {"status": "PASS", "users": len(users), "train_ratings": len(train_ratings),
            "validation_ratings": sum(map(len, validation.values())),
            "rankings_checked": len(ranked_rows), "positive_validation_users": positive_users,
            "mean_ndcg_at_10": ndcg10_sum / positive_users,
            "mean_ndcg_at_50": ndcg50_sum / positive_users,
            "cold_recipes": int(cold.sum())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.output), indent=2))
