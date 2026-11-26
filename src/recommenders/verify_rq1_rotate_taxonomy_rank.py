"""Verify FoodOn hierarchy RotatE top-50 rankings before test metric use."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, ROTATE, sha, verify_lock
from src.recommenders.rotate_ranking import recipe_order
from src.recommenders.rq1_rotate_taxonomy_test import (HIERARCHY, OUTPUT, RANK_HEADER,
                                                       require_preflight)


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def verify(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    require_preflight(data, output)
    path = output / "4_rotate_hierarchy_top50.csv"
    manifest = json.loads((output / "rank_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "hierarchy_test_rankings_complete"
    assert manifest["ranking_sha256"] == sha(path)
    assert manifest["preflight_sha256"] == sha(output / "preflight.json")
    assert manifest["model_sha256"] == sha(HIERARCHY / "model.pt")
    assert manifest["weight_five"] == 0.75
    catalog = {row["recipe_id"] for row in rows(data / "recipes.csv")}
    train = defaultdict(set)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train":
            train[row["user_id"]].add(row["recipe_id"])
    assert len(catalog) == 75972 and len(train) == 4000
    current_user, current_rows = None, []
    user_count = row_count = fallback_count = 0
    sampled_users = {sorted(train)[position] for position in (0, 2000, 3999)}
    sampled = {}

    def check_user(user, entries):
        nonlocal user_count
        assert len(entries) == 50
        assert [int(row["rank"]) for row in entries] == list(range(1, 51))
        assert len({row["recipe_id"] for row in entries}) == 50
        assert all(row["recipe_id"] in catalog and row["recipe_id"] not in train[user] for row in entries)
        assert {int(row["scorable_candidates"]) for row in entries} == {74027}
        assert all(row["score_source"] == "score" and math.isfinite(float(row["score"]))
                   for row in entries)
        assert entries == sorted(entries, key=lambda row:
                                 (-float(row["score"]), recipe_order(row["recipe_id"])))
        if user in sampled_users:
            sampled[user] = [row["recipe_id"] for row in entries]
        user_count += 1

    for row in rows(path):
        assert tuple(row) == RANK_HEADER and row["condition"] == "4_rotate_hierarchy"
        if current_user is None:
            current_user = row["user_id"]
        if row["user_id"] != current_user:
            check_user(current_user, current_rows)
            assert row["user_id"] > current_user
            current_user, current_rows = row["user_id"], []
        current_rows.append(row)
        fallback_count += row["score_source"] == "fallback"
        row_count += 1
    check_user(current_user, current_rows)
    assert user_count == manifest["users"] == 4000
    assert row_count == manifest["rows"] == 200000
    assert fallback_count == manifest["fallback_top50_rows"] == 0

    # Recompute three complete lists directly from the saved PyKEEN state.
    import numpy as np
    import torch
    from pykeen.losses import MarginRankingLoss
    from pykeen.models import RotatE
    from scipy.stats import rankdata
    from src.recommenders.rotate import load_factory

    factory = load_factory(data, "hierarchy")
    config = json.loads(ROTATE.read_text(encoding="utf-8"))
    model = RotatE(triples_factory=factory, embedding_dim=config["training"]["embedding_dim"],
                   loss=MarginRankingLoss(margin=config["training"]["margin"]),
                   random_seed=config["training"]["seed"])
    model.load_state(HIERARCHY / "model.pt")
    model.eval()
    embedded = [recipe for recipe in sorted(catalog, key=recipe_order)
                if recipe in factory.entity_to_id]
    ids = torch.tensor([factory.entity_to_id[recipe] for recipe in embedded], dtype=torch.long)
    with torch.inference_mode():
        for user in sorted(sampled_users):
            eligible = [index for index, recipe in enumerate(embedded) if recipe not in train[user]]
            percentiles = []
            for relation in ("RATED_4", "RATED_5"):
                query = torch.tensor([[factory.entity_to_id[user], factory.relation_to_id[relation]]],
                                     dtype=torch.long)
                raw = model.predict_t(query, tails=ids).reshape(-1).cpu().numpy()
                percentiles.append(rankdata(raw[eligible], method="average") / len(eligible))
            combined = 0.25 * percentiles[0] + 0.75 * percentiles[1]
            expected = [embedded[eligible[index]] for index in
                        np.lexsort((np.asarray(eligible), -combined))[:50]]
            assert sampled[user] == expected
    return {"status": "hierarchy_rankings_verified", "users": user_count,
            "rows": row_count, "fallback_top50_rows": fallback_count,
            "independently_recomputed_user_lists": len(sampled_users)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
