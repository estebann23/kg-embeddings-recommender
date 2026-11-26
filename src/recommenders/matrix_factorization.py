"""Train and validation-select a test-blind, interaction-only HUMMUS MF baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.rotate_ranking import recipe_order

CONFIG = ROOT / "config/rq1_matrix_factorization.json"
OUTPUT = ROOT / "results/matrix_factorization_hummus_4000_v1"
GAIN = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def load_inputs(data: Path):
    users = sorted(row["user_id"] for row in read_rows(data / "users.csv"))
    recipes = sorted((row["recipe_id"] for row in read_rows(data / "recipes.csv")),
                     key=recipe_order)
    require(len(users) == len(set(users)) == 4000, "Unexpected user cohort")
    require(len(recipes) == len(set(recipes)) == 75972, "Unexpected recipe catalog")
    uid = {user: i for i, user in enumerate(users)}
    iid = {recipe: i for i, recipe in enumerate(recipes)}
    train, validation = [], []
    per_user = defaultdict(lambda: Counter())
    train_history = defaultdict(set)
    val_labels = defaultdict(dict)
    pairs = set()
    for row in read_rows(data / "reviews.csv"):
        pair = row["user_id"], row["recipe_id"]
        require(pair not in pairs, "Duplicate user-recipe pair")
        pairs.add(pair)
        split = row["split"]
        if split not in {"train", "validation"}:
            continue  # Do not parse test rating values, even for an audit.
        u, i, rating = uid[row["user_id"]], iid[row["recipe_id"]], int(row["rating"])
        require(1 <= rating <= 5, "Rating outside observed 1-5 range")
        per_user[u][split] += 1
        if split == "train":
            train.append((u, i, rating))
            train_history[u].add(i)
        else:
            validation.append((u, i, rating))
            val_labels[u][i] = rating
    require(len(train) == 112000 and len(validation) == 16000,
            "Unexpected training/validation cardinality")
    require(len(per_user) == 4000 and all(c == {"train": 28, "validation": 4}
                                         for c in per_user.values()),
            "Unexpected per-user split")
    require(all(not train_history[u] & set(val_labels[u]) for u in range(len(users))),
            "Validation pair in training")
    return users, recipes, np.asarray(train, dtype=np.int64), \
        np.asarray(validation, dtype=np.int64), train_history, val_labels


class BiasedMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, global_mean: float):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, factors)
        self.item_factors = nn.Embedding(n_items, factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        u, i = self.user_factors(users), self.item_factors(items)
        ub, ib = self.user_bias(users).squeeze(-1), self.item_bias(items).squeeze(-1)
        prediction = self.global_mean + ub + ib + (u * i).sum(-1)
        penalty = (u.square().sum(-1) + i.square().sum(-1) + ub.square() + ib.square()).mean()
        return prediction, penalty


def arrays(model: BiasedMF):
    with torch.no_grad():
        return (model.user_factors.weight.detach().cpu().numpy().copy(),
                model.item_factors.weight.detach().cpu().numpy().copy(),
                model.user_bias.weight.detach().cpu().numpy().reshape(-1).copy(),
                model.item_bias.weight.detach().cpu().numpy().reshape(-1).copy(),
                float(model.global_mean.item()))


def validation_rmse(parameters, validation: np.ndarray, train_item_counts: np.ndarray) -> float:
    uvec, ivec, ub, ib, global_mean = parameters
    users, items, ratings = validation.T
    item_vectors = ivec[items].copy()
    item_vectors[train_item_counts[items] == 0] = 0
    item_bias = ib[items].copy()
    item_bias[train_item_counts[items] == 0] = 0
    prediction = global_mean + ub[users] + item_bias + np.einsum(
        "ij,ij->i", uvec[users], item_vectors)
    return float(np.sqrt(np.mean((np.clip(prediction, 1, 5) - ratings) ** 2)))


def rank_users(parameters, users, train_history, val_labels, train_item_counts,
               positive_item_counts, k=10, collect=False):
    uvec, ivec, ub, ib, global_mean = parameters
    n_items = len(ivec)
    seen = np.flatnonzero(train_item_counts > 0)
    cold = np.flatnonzero(train_item_counts == 0)
    cold = cold[np.lexsort((cold, -positive_item_counts[cold]))]
    require(len(seen) + len(cold) == n_items, "Item partition changed")
    seen_position = np.full(n_items, -1, dtype=np.int32)
    seen_position[seen] = np.arange(len(seen), dtype=np.int32)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    metric_sum = 0.0
    eligible_count = 0
    rows = []
    for start in range(0, len(users), 64):
        end = min(start + 64, len(users))
        block = (uvec[start:end] @ ivec[seen].T).astype(np.float32, copy=False)
        block += ib[seen][None, :]
        block += ub[start:end, None] + global_mean
        require(np.isfinite(block).all(), "Non-finite MF scores")
        for offset, user in enumerate(range(start, end)):
            scores = block[offset]
            excluded = np.fromiter(train_history[user], dtype=np.int64)
            scores[seen_position[excluded]] = -np.inf
            want = min(k, len(seen) - len(excluded))
            cutoff = np.partition(scores, -want)[-want]
            contenders = np.flatnonzero(scores >= cutoff)
            order = contenders[np.lexsort((seen[contenders], -scores[contenders]))[:want]]
            ranked = seen[order].tolist()
            if len(ranked) < k:
                ranked.extend(i for i in cold if i not in train_history[user])
                ranked = ranked[:k]
            require(len(ranked) == k and len(set(ranked)) == k and
                    not set(ranked) & train_history[user], "Invalid MF ranking")
            labels = val_labels[user]
            gains = [GAIN[labels.get(i, 1)] for i in ranked]
            ideal = sorted((GAIN[r] for r in labels.values() if GAIN[r] > 0), reverse=True)
            ndcg = (float(np.dot(gains, discounts)) /
                    float(np.dot(ideal[:k], discounts[:len(ideal)]))) if ideal else None
            if ndcg is not None:
                metric_sum += ndcg
                eligible_count += 1
            if collect:
                rows.append((user, ranked, [float(scores[seen_position[i]]) if
                            seen_position[i] >= 0 else None for i in ranked], ndcg))
    require(eligible_count > 0, "No positive validation users")
    return metric_sum / eligible_count, eligible_count, rows


def train(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(config["dataset_id"] == data.name and config["status"] == "frozen_before_mf_training",
            "MF configuration not frozen for this cohort")
    users, recipes, observed, validation, history, labels = load_inputs(data)
    train_item_counts = np.bincount(observed[:, 1], minlength=len(recipes))
    positive_item_counts = np.bincount(observed[observed[:, 2] >= 4, 1], minlength=len(recipes))
    global_mean = float(observed[:, 2].mean())
    grid = config["grid"]
    require(grid == {"factors": [16, 32], "regularization": [0.01, 0.1],
                     "checkpoint_epochs": [10, 20]}, "Unexpected MF grid")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_bytes(CONFIG.read_bytes())
    train_tensor = torch.tensor(observed, dtype=torch.long)
    trials = []
    best = None
    best_arrays = None
    for factors in grid["factors"]:
        for regularization in grid["regularization"]:
            torch.manual_seed(config["seed"])
            generator = torch.Generator().manual_seed(config["seed"])
            model = BiasedMF(len(users), len(recipes), factors, global_mean)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            for epoch in range(1, max(grid["checkpoint_epochs"]) + 1):
                order = torch.randperm(len(observed), generator=generator)
                loss_total = 0.0
                for selected in order.split(config["batch_size"]):
                    batch = train_tensor[selected]
                    prediction, penalty = model(batch[:, 0], batch[:, 1])
                    loss = (prediction - batch[:, 2].float()).square().mean() + regularization * penalty
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.item()) * len(selected)
                if epoch not in grid["checkpoint_epochs"]:
                    continue
                params = arrays(model)
                mean_ndcg, positive_users, _ = rank_users(
                    params, users, history, labels, train_item_counts, positive_item_counts)
                rmse = validation_rmse(params, validation, train_item_counts)
                record = {"factors": factors, "regularization": regularization,
                          "epoch": epoch, "training_objective": loss_total / len(observed),
                          "validation_rmse_clipped": rmse,
                          "validation_mean_ndcg_at_10": mean_ndcg,
                          "users_with_positive_validation": positive_users}
                trials.append(record)
                print(json.dumps(record), flush=True)
                key = (mean_ndcg, -rmse, -factors, regularization, -epoch)
                if best is None or key > best:
                    best, best_arrays, selected_record = key, params, record
    require(best_arrays is not None and len(trials) == 8, "Incomplete MF search")
    uvec, ivec, ub, ib, selected_mean = best_arrays
    ivec[train_item_counts == 0] = 0
    ib[train_item_counts == 0] = 0
    model_path = output / "selected_model.npz"
    np.savez_compressed(model_path, user_factors=uvec, item_factors=ivec,
                        user_bias=ub, item_bias=ib, global_mean=np.array(selected_mean),
                        train_item_counts=train_item_counts, positive_item_counts=positive_item_counts,
                        user_ids=np.asarray(users), recipe_ids=np.asarray(recipes))
    mean_ndcg, positive_users, selected_rows = rank_users(
        best_arrays, users, history, labels, train_item_counts, positive_item_counts,
        k=config["output_top_k"], collect=True)
    # NDCG@50 above is a different diagnostic from the NDCG@10 tuning objective.
    validation_path = output / "validation_top50.csv"
    with validation_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "rank", "recipe_id", "score", "score_source"))
        for user, ranked, scores, _ in selected_rows:
            for rank, (item, score) in enumerate(zip(ranked, scores), 1):
                writer.writerow((users[user], rank, recipes[item],
                                 "" if score is None else format(score, ".9g"),
                                 "score" if score is not None else "fallback"))
    # The rows contain model scores and IDs only; no held-out labels or metrics.
    test_blind_path = output / "test_blind_top50.csv"
    shutil.copyfile(validation_path, test_blind_path)
    per_user_path = output / "validation_per_user.csv"
    with per_user_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "positive_validation_targets", "ndcg_at_10",
                         "ndcg_at_50", "precision_at_10", "precision_at_50"))
        for user, ranked, _, ndcg50 in selected_rows:
            targets = labels[user]
            n10 = [GAIN[targets.get(item, 1)] for item in ranked[:10]]
            ideal = sorted((GAIN[r] for r in targets.values() if GAIN[r] > 0), reverse=True)
            idcg10 = sum(g / math.log2(j + 2) for j, g in enumerate(ideal[:10]))
            ndcg10 = (sum(g / math.log2(j + 2) for j, g in enumerate(n10)) /
                      idcg10) if idcg10 else None
            writer.writerow((users[user], len(ideal),
                             "" if ndcg10 is None else format(ndcg10, ".17g"),
                             "" if ndcg50 is None else format(ndcg50, ".17g"),
                             format(sum(g > 0 for g in n10) / 10, ".17g"),
                             format(sum(targets.get(item, 0) >= 4 for item in ranked) / 50, ".17g")))
    result = {
        "status": "validation_selected_no_test_access",
        "condition": config["condition"], "dataset_id": data.name,
        "selected": selected_record, "grid_results": trials,
        "train_users": len(users), "train_ratings": len(observed),
        "validation_ratings": len(validation), "catalog_recipes": len(recipes),
        "train_seen_recipes": int((train_item_counts > 0).sum()),
        "unscorable_recipes": int((train_item_counts == 0).sum()),
        "positive_validation_users": positive_users,
        "selected_validation_mean_ndcg_at_50": mean_ndcg,
        "config_sha256": sha(CONFIG),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "reviews_sha256": sha(data / "reviews.csv"),
        "recipes_sha256": sha(data / "recipes.csv"),
        "model_sha256": sha(model_path),
        "validation_top50_sha256": sha(validation_path),
        "test_blind_top50_sha256": sha(test_blind_path),
        "validation_per_user_sha256": sha(per_user_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_policy": "No test rating parsed or used for training, selection, or validation metrics."
    }
    (output / "validation_selection.json").write_text(json.dumps(result, indent=2) + "\n",
                                                      encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))
