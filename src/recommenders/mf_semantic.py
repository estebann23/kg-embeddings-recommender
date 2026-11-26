"""Content-boosted MF using audited ingredient and direct FoodOn links only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.matrix_factorization import (BiasedMF, GAIN, load_inputs,
                                                   rank_users, validation_rmse)

CONFIG = ROOT / "config/rq1_mf_semantic.json"
OUTPUT = ROOT / "results/mf_semantic_hummus_4000_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def csr_from_sets(feature_sets, vocab):
    positions = {feature: j for j, feature in enumerate(vocab)}
    indices = []
    indptr = [0]
    for features in feature_sets:
        indices.extend(sorted((positions[f] for f in features)))
        indptr.append(len(indices))
    return np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64)


def load_features(data, recipes):
    recipe_index = {recipe: i for i, recipe in enumerate(recipes)}
    ingredient_sets = [set() for _ in recipes]
    for row in read_csv(data / "recipe_ingredients.csv"):
        require(row["evidence_state"] == "INFERRED" and
                row["recipe_id"] in recipe_index, "Unexpected recipe-ingredient evidence")
        ingredient_sets[recipe_index[row["recipe_id"]]].add(row["ingredient_id"])
    concept_by_ingredient = defaultdict(set)
    for row in read_csv(data / "foodon_links.csv"):
        require(row["evidence_state"] == "INFERRED" and
                row["foodon_iri"].startswith("http://purl.obolibrary.org/obo/FOODON_"),
                "Unexpected direct FoodOn evidence")
        concept_by_ingredient[row["ingredient_id"]].add(row["foodon_iri"])
    concept_sets = [set().union(*(concept_by_ingredient[i] for i in ingredients))
                    for ingredients in ingredient_sets]
    ingredient_vocab = sorted(set().union(*ingredient_sets))
    concept_vocab = sorted(set().union(*concept_sets))
    ing_ptr, ing_idx = csr_from_sets(ingredient_sets, ingredient_vocab)
    con_ptr, con_idx = csr_from_sets(concept_sets, concept_vocab)
    return (ingredient_vocab, concept_vocab, ing_ptr, ing_idx, con_ptr, con_idx)


def normalized_matrix(ptr, idx, n_features):
    counts = np.diff(ptr)
    data = np.repeat(np.divide(1.0, counts, out=np.zeros(len(counts)),
                               where=counts > 0), counts)
    return sparse.csr_matrix((data.astype(np.float32), idx, ptr),
                             shape=(len(counts), n_features), dtype=np.float32)


def bag_batch(items, ptr, indices):
    starts, ends = ptr[items], ptr[items + 1]
    lengths = ends - starts
    offsets = np.empty(len(items), dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(lengths[:-1])
    flattened = np.concatenate([indices[start:end] for start, end in zip(starts, ends)])
    return torch.from_numpy(flattened), torch.from_numpy(offsets)


class SemanticMF(BiasedMF):
    def __init__(self, n_users, n_items, factors, global_mean, n_ingredients, n_concepts):
        super().__init__(n_users, n_items, factors, global_mean)
        self.ingredient_factors = nn.EmbeddingBag(n_ingredients, factors, mode="mean")
        self.concept_factors = nn.EmbeddingBag(n_concepts, factors, mode="mean")
        nn.init.normal_(self.ingredient_factors.weight, std=0.05)
        nn.init.normal_(self.concept_factors.weight, std=0.05)

    def forward(self, users, items, ing_ids, ing_offsets, con_ids, con_offsets, alpha):
        u, free = self.user_factors(users), self.item_factors(items)
        ing = self.ingredient_factors(ing_ids, ing_offsets)
        con = self.concept_factors(con_ids, con_offsets)
        content = 0.5 * (ing + con)
        item = free + alpha * content
        ub = self.user_bias(users).squeeze(-1)
        ib = self.item_bias(items).squeeze(-1)
        prediction = self.global_mean + ub + ib + (u * item).sum(-1)
        penalty = (u.square().sum(-1) + free.square().sum(-1) +
                   ing.square().sum(-1) + con.square().sum(-1) +
                   ub.square() + ib.square()).mean()
        return prediction, penalty


def effective_arrays(model, alpha, train_counts, ing_ptr, ing_idx, con_ptr, con_idx):
    with torch.no_grad():
        uf = model.user_factors.weight.detach().cpu().numpy().copy()
        free = model.item_factors.weight.detach().cpu().numpy().copy()
        ub = model.user_bias.weight.detach().cpu().numpy().reshape(-1).copy()
        ib = model.item_bias.weight.detach().cpu().numpy().reshape(-1).copy()
        ing = model.ingredient_factors.weight.detach().cpu().numpy().copy()
        con = model.concept_factors.weight.detach().cpu().numpy().copy()
        mean = float(model.global_mean.item())
    ing_matrix = normalized_matrix(ing_ptr, ing_idx, len(ing))
    con_matrix = normalized_matrix(con_ptr, con_idx, len(con))
    trained_items = train_counts > 0
    supervised_ing = np.asarray(ing_matrix[trained_items].getnnz(axis=0) > 0).reshape(-1)
    supervised_con = np.asarray(con_matrix[trained_items].getnnz(axis=0) > 0).reshape(-1)
    ing[~supervised_ing] = 0
    con[~supervised_con] = 0
    free[~trained_items] = 0
    ib[~trained_items] = 0
    effective = free + alpha * 0.5 * (ing_matrix @ ing + con_matrix @ con)
    scorable = trained_items | np.asarray(ing_matrix[:, supervised_ing].getnnz(axis=1) > 0).reshape(-1) | \
        np.asarray(con_matrix[:, supervised_con].getnnz(axis=1) > 0).reshape(-1)
    params = (uf, np.asarray(effective, dtype=np.float32), ub, ib, mean)
    state = {"user_factors": uf, "free_item_factors": free,
             "ingredient_factors": ing, "concept_factors": con,
             "user_bias": ub, "item_bias": ib, "global_mean": np.array(mean),
             "effective_item_factors": params[1], "scorable": scorable,
             "supervised_ingredients": supervised_ing, "supervised_concepts": supervised_con}
    require(np.isfinite(params[1]).all(), "Non-finite semantic item factors")
    return params, state


def run(data=DATA, output=OUTPUT):
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(config["status"] == "frozen_before_semantic_mf_training" and
            config["dataset_id"] == data.name, "Semantic MF config differs")
    require(config["grid"] == {"content_scale": [0.5, 1.0],
                               "regularization": [0.01, 0.1],
                               "checkpoint_epochs": [10, 20]} and config["factors"] == 16,
            "Unexpected semantic MF search")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in ("recipe_ingredients", "foodon_links", "reviews", "recipes"):
        require(sha(data / f"{name}.csv") == manifest["output_sha256"][name],
                f"Source {name} differs from cohort manifest")
    users, recipes, train, validation, history, labels = load_inputs(data)
    ing_vocab, con_vocab, ing_ptr, ing_idx, con_ptr, con_idx = load_features(data, recipes)
    require(len(ing_vocab) > 0 and len(con_vocab) > 0, "Empty semantic vocabulary")
    train_counts = np.bincount(train[:, 1], minlength=len(recipes))
    positive_counts = np.bincount(train[train[:, 2] >= 4, 1], minlength=len(recipes))
    torch.set_num_threads(4)
    train_tensor = torch.tensor(train, dtype=torch.long)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, output / "experiment_config.json")
    trials, best_key, best_state = [], None, None
    for alpha in config["grid"]["content_scale"]:
        for regularization in config["grid"]["regularization"]:
            torch.manual_seed(config["seed"])
            generator = torch.Generator().manual_seed(config["seed"])
            model = SemanticMF(len(users), len(recipes), config["factors"],
                               float(train[:, 2].mean()), len(ing_vocab), len(con_vocab))
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            for epoch in range(1, max(config["grid"]["checkpoint_epochs"]) + 1):
                order = torch.randperm(len(train), generator=generator)
                loss_total = 0.0
                for selected in order.split(config["batch_size"]):
                    batch = train_tensor[selected]
                    items = batch[:, 1].numpy()
                    ing_ids, ing_offsets = bag_batch(items, ing_ptr, ing_idx)
                    con_ids, con_offsets = bag_batch(items, con_ptr, con_idx)
                    prediction, penalty = model(batch[:, 0], batch[:, 1],
                                                ing_ids, ing_offsets, con_ids, con_offsets,
                                                alpha)
                    loss = (prediction - batch[:, 2].float()).square().mean() + regularization * penalty
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.item()) * len(selected)
                if epoch not in config["grid"]["checkpoint_epochs"]:
                    continue
                params, state = effective_arrays(model, alpha, train_counts,
                                                 ing_ptr, ing_idx, con_ptr, con_idx)
                scoreable = state["scorable"].astype(np.int8)
                ndcg, eligible_users, _ = rank_users(
                    params, users, history, labels, scoreable, positive_counts, k=10)
                rmse = validation_rmse(params, validation, scoreable)
                record = {"content_scale": alpha, "regularization": regularization,
                          "epoch": epoch, "factors": config["factors"],
                          "training_objective": loss_total / len(train),
                          "validation_rmse_clipped": rmse,
                          "validation_mean_ndcg_at_10": ndcg,
                          "users_with_positive_validation": eligible_users,
                          "scoreable_recipes": int(state["scorable"].sum())}
                trials.append(record)
                print(json.dumps(record), flush=True)
                key = (ndcg, -rmse, -alpha, regularization, -epoch)
                if best_key is None or key > best_key:
                    best_key, best_state, best_record = key, state, record
    require(best_state is not None and len(trials) == 8, "Incomplete semantic MF search")
    state = best_state
    model_path = output / "selected_model.npz"
    np.savez_compressed(model_path, **state, user_ids=np.asarray(users),
                        recipe_ids=np.asarray(recipes), ingredient_ids=np.asarray(ing_vocab),
                        concept_ids=np.asarray(con_vocab), ingredient_indptr=ing_ptr,
                        ingredient_indices=ing_idx, concept_indptr=con_ptr,
                        concept_indices=con_idx, train_item_counts=train_counts,
                        positive_item_counts=positive_counts,
                        content_scale=np.array(best_record["content_scale"]))
    selected_params = (state["user_factors"], state["effective_item_factors"],
                       state["user_bias"], state["item_bias"], float(state["global_mean"]))
    scoreable = state["scorable"].astype(np.int8)
    ndcg50, eligible_users, selected_rows = rank_users(
        selected_params, users, history, labels, scoreable, positive_counts,
        k=config["output_top_k"], collect=True)
    ranking_path = output / "test_blind_top50.csv"
    with ranking_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "rank", "recipe_id", "score", "score_source"))
        for user, ranked, scores, _ in selected_rows:
            for position, (item, score) in enumerate(zip(ranked, scores), 1):
                writer.writerow((users[user], position, recipes[item],
                                 "" if score is None else format(score, ".9g"),
                                 "score" if score is not None else "fallback"))
    per_user_path = output / "validation_per_user.csv"
    with per_user_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "positive_validation_targets", "ndcg_at_10",
                         "ndcg_at_50", "precision_at_10", "precision_at_50"))
        for user, ranked, _, ndcg_at_50 in selected_rows:
            targets = labels[user]
            gains = [GAIN[targets.get(item, 1)] for item in ranked]
            ideal = sorted((GAIN[r] for r in targets.values() if GAIN[r] > 0), reverse=True)
            idcg = sum(g / math.log2(j + 2) for j, g in enumerate(ideal))
            ndcg_at_10 = (sum(g / math.log2(j + 2) for j, g in enumerate(gains[:10])) /
                          idcg) if idcg else None
            writer.writerow((users[user], len(ideal),
                             "" if ndcg_at_10 is None else format(ndcg_at_10, ".17g"),
                             "" if ndcg_at_50 is None else format(ndcg_at_50, ".17g"),
                             format(sum(g > 0 for g in gains[:10]) / 10, ".17g"),
                             format(sum(g > 0 for g in gains) / 50, ".17g")))
    report = {
        "status": "validation_selected_no_test_access", "condition": config["condition"],
        "dataset_id": data.name, "selected": best_record, "grid_results": trials,
        "train_users": len(users), "train_ratings": len(train),
        "validation_ratings": len(validation), "catalog_recipes": len(recipes),
        "mapped_ingredient_recipes": int(np.sum(np.diff(ing_ptr) > 0)),
        "direct_foodon_recipes": int(np.sum(np.diff(con_ptr) > 0)),
        "ingredient_features": len(ing_vocab), "direct_concept_features": len(con_vocab),
        "supervised_ingredient_features": int(state["supervised_ingredients"].sum()),
        "supervised_concept_features": int(state["supervised_concepts"].sum()),
        "train_seen_recipes": int(np.sum(train_counts > 0)),
        "semantically_scorable_cold_recipes": int(np.sum((train_counts == 0) & state["scorable"])),
        "unscorable_recipes": int(np.sum(~state["scorable"])),
        "positive_validation_users": eligible_users,
        "selected_validation_mean_ndcg_at_50": ndcg50,
        "source_sha256": {name: sha(data / f"{name}.csv") for name in
                          ("reviews", "recipes", "recipe_ingredients", "foodon_links")},
        "config_sha256": sha(CONFIG),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "model_sha256": sha(model_path), "test_blind_top50_sha256": sha(ranking_path),
        "validation_per_user_sha256": sha(per_user_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_policy": "Test rating fields never parsed; only train ratings fit and validation ratings select."
    }
    (output / "validation_selection.json").write_text(json.dumps(report, indent=2) + "\n",
                                                      encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
