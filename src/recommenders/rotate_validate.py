"""Select the frozen RotatE score-fusion weight using validation ratings only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from pykeen.losses import MarginRankingLoss
from pykeen.models import RotatE
from scipy.stats import rankdata

from src.recommenders.contract import DATA, ROTATE, ROOT, sha, verify_lock
from src.recommenders.rotate import load_factory, read_json, require
from src.recommenders.rotate_ranking import WEIGHT_GRID, recipe_order


def validation_rows(data: Path):
    train, validation = defaultdict(set), defaultdict(dict)
    with (data / "reviews.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "train":
                train[row["user_id"]].add(row["recipe_id"])
            elif row["split"] == "validation":
                validation[row["user_id"]][row["recipe_id"]] = int(row["rating"])
    return train, validation


def run(variant: str, data: Path = DATA) -> dict:
    data = Path(data).resolve()
    lock = verify_lock(data)
    config = read_json(ROTATE)
    output = ROOT / "results" / f"rotate_{variant}_full"
    manifest = read_json(output / "training_manifest.json")
    require(manifest["status"] == "trained_not_validated" and manifest["epochs"] == 100,
            "Full training has not completed")
    require(manifest["config_sha256"] == sha(ROTATE) and
            manifest["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json"),
            "Trained model does not match frozen contract")
    factory = load_factory(data, variant)
    require(manifest["triple_sha256"] == read_json(data / f"rotate_{variant}_preflight.json")["triple_sha256"],
            "Trained triple snapshot mismatch")
    model = RotatE(triples_factory=factory, embedding_dim=config["training"]["embedding_dim"],
                   loss=MarginRankingLoss(margin=config["training"]["margin"]),
                   random_seed=config["training"]["seed"])
    model.load_state(output / "model.pt")
    model.eval()

    with (data / "recipes.csv").open(encoding="utf-8", newline="") as stream:
        recipes = sorted((row["recipe_id"] for row in csv.DictReader(stream)), key=recipe_order)
    scorable = [recipe for recipe in recipes if recipe in factory.entity_to_id]
    ids = torch.tensor([factory.entity_to_id[recipe] for recipe in scorable], dtype=torch.long)
    index = {recipe: i for i, recipe in enumerate(scorable)}
    train, validation = validation_rows(data)
    require(len(train) == len(validation) == lock["verified_users"], "User split mismatch")
    totals = np.zeros(len(WEIGHT_GRID), dtype=np.float64)
    per_user = []
    with torch.inference_mode():
        for user_id in sorted(train):
            ratings = validation[user_id]
            if not any(r >= 4 for r in ratings.values()):
                continue
            user_index = factory.entity_to_id[user_id]
            scores = []
            for relation in ("RATED_4", "RATED_5"):
                query = torch.tensor([[user_index, factory.relation_to_id[relation]]], dtype=torch.long)
                values = model.predict_t(query, tails=ids).reshape(-1).cpu().numpy()
                require(np.isfinite(values).all(), "Non-finite validation prediction")
                scores.append(values)
            eligible = np.ones(len(scorable), dtype=bool)
            for recipe in train[user_id]:
                if recipe in index:
                    eligible[index[recipe]] = False
            eligible_index = np.flatnonzero(eligible)
            p4 = rankdata(scores[0][eligible_index], method="average") / len(eligible_index)
            p5 = rankdata(scores[1][eligible_index], method="average") / len(eligible_index)
            gains = np.array([2 if ratings.get(scorable[i]) == 5 else
                              1 if ratings.get(scorable[i]) == 4 else 0 for i in eligible_index])
            ideal_gains = sorted((2 if r == 5 else 1 for r in ratings.values() if r >= 4), reverse=True)
            ideal = sum(g / np.log2(pos + 2) for pos, g in enumerate(ideal_gains[:10]))
            user_metrics = []
            for w in WEIGHT_GRID:
                combined = (1 - w) * p4 + w * p5
                # Full eligible catalog participates; stable source ID breaks exact score ties.
                cutoff = np.partition(combined, -10)[-10]
                contenders = np.flatnonzero(combined >= cutoff)
                top = contenders[np.lexsort((eligible_index[contenders], -combined[contenders]))[:10]]
                dcg = sum(gains[i] / np.log2(pos + 2) for pos, i in enumerate(top))
                user_metrics.append(float(dcg / ideal))
            totals += user_metrics
            per_user.append({"user_id": user_id, "ndcg_at_10": user_metrics})
    require(per_user, "No positive validation users")
    means = totals / len(per_user)
    best = max(range(len(WEIGHT_GRID)),
               key=lambda i: (means[i], -abs(WEIGHT_GRID[i] - 0.5), -WEIGHT_GRID[i]))
    result = {
        "status": "validation_selected_no_test_access",
        "variant": variant,
        "selected_weight_five": WEIGHT_GRID[best],
        "selected_weight_four": 1 - WEIGHT_GRID[best],
        "mean_ndcg_at_10_by_weight": dict(zip(map(str, WEIGHT_GRID), map(float, means))),
        "users_with_positive_validation": len(per_user),
        "training_manifest_sha256": sha(output / "training_manifest.json"),
        "model_sha256": sha(output / "model.pt"),
        "contract_lock_sha256": sha(ROOT / "config/rq1_experiment.lock.json"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "validation_selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("primary", "hierarchy"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.variant), indent=2))


if __name__ == "__main__":
    main()
