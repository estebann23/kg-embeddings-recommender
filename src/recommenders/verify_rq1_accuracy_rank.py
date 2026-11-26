"""Check complete test-blind top-50 files before any test metric is computed."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf import OUTPUT as CF_OUTPUT
from src.recommenders.explicit_paths import OUTPUT as PATH_OUTPUT
from src.recommenders.rotate_ranking import recipe_order
from src.recommenders.rq1_accuracy_rank import (CONDITIONS, HEADER, OUTPUT,
                                                read_popularity, require_preflight)


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def verify(data: Path = DATA, output: Path = OUTPUT) -> dict:
    data, output = Path(data).resolve(), Path(output).resolve()
    verify_lock(data)
    preflight = require_preflight(data, output)
    catalog = {row["recipe_id"] for row in rows(data / "recipes.csv")}
    assert len(catalog) == 75972
    train = defaultdict(set)
    for row in rows(data / "reviews.csv"):
        if row["split"] == "train":
            train[row["user_id"]].add(row["recipe_id"])
    assert len(train) == 4000 and all(len(history) == 28 for history in train.values())
    popularity = read_popularity(data, CF_OUTPUT)
    validation_top10 = {}
    for condition, path in (("1_item_cf", CF_OUTPUT / "validation_recommendations_selected.csv"),
                            ("3_explicit_paths", PATH_OUTPUT / "validation_recommendations_selected.csv")):
        validation_top10[condition] = {(row["user_id"], int(row["rank"])): row["recipe_id"]
                                       for row in rows(path)}
        assert len(validation_top10[condition]) == 40000
    result = {}
    for condition in CONDITIONS:
        path = output / f"{condition}_top50.csv"
        manifest = json.loads((output / f"{condition}_rank_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "test_blind_rankings_complete"
        assert manifest["ranking_sha256"] == sha(path)
        assert manifest["preflight_sha256"] == sha(output / "preflight.json")
        assert manifest["users"] == 4000 and manifest["rows"] == 200000
        current_user, current_rows = None, []
        users = rows_seen = fallback_rows = 0
        min_scorable, max_scorable = 75972, 0

        def check_user(user, entries):
            nonlocal users, min_scorable, max_scorable
            assert user in train and len(entries) == 50
            assert [int(row["rank"]) for row in entries] == list(range(1, 51))
            assert len({row["recipe_id"] for row in entries}) == 50
            eligible = catalog - train[user]
            assert all(row["recipe_id"] in eligible for row in entries)
            counts = {int(row["scorable_candidates"]) for row in entries}
            assert len(counts) == 1 and 0 <= next(iter(counts)) <= 75944
            min_scorable = min(min_scorable, next(iter(counts)))
            max_scorable = max(max_scorable, next(iter(counts)))
            sources = [row["score_source"] for row in entries]
            assert sources == sorted(sources, key=lambda source: source == "fallback")
            scored = [row for row in entries if row["score_source"] == "score"]
            fallback = [row for row in entries if row["score_source"] == "fallback"]
            assert all(row["score"] != "" and math.isfinite(float(row["score"])) for row in scored)
            assert all(row["score"] == "" for row in fallback)
            assert scored == sorted(scored, key=lambda row: (-float(row["score"]), recipe_order(row["recipe_id"])))
            if condition == "4_rotate_primary":
                assert fallback == sorted(fallback, key=lambda row: recipe_order(row["recipe_id"]))
            else:
                assert fallback == sorted(fallback, key=lambda row:
                                          (-popularity[row["recipe_id"]], recipe_order(row["recipe_id"])))
                for rank in range(1, 11):
                    assert entries[rank - 1]["recipe_id"] == validation_top10[condition][user, rank]
            users += 1

        for row in rows(path):
            assert tuple(row) == HEADER and row["condition"] == condition
            if current_user is None:
                current_user = row["user_id"]
            if row["user_id"] != current_user:
                check_user(current_user, current_rows)
                assert row["user_id"] > current_user
                current_user, current_rows = row["user_id"], []
            current_rows.append(row)
            rows_seen += 1
            fallback_rows += row["score_source"] == "fallback"
        check_user(current_user, current_rows)
        assert users == 4000 and rows_seen == 200000
        assert fallback_rows == manifest["fallback_top50_rows"]
        assert [min_scorable, max_scorable] == [manifest["scorable_candidates_min"],
                                                manifest["scorable_candidates_max"]]
        result[condition] = {"users": users, "rows": rows_seen,
                             "fallback_top50_rows": fallback_rows,
                             "scorable_candidate_range": [min_scorable, max_scorable]}
    # Recompute a small, separated sample of RotatE lists from the saved state.
    import numpy as np
    import torch
    from pykeen.losses import MarginRankingLoss
    from pykeen.models import RotatE
    from scipy.stats import rankdata
    from src.recommenders.rotate import load_factory
    from src.recommenders.contract import ROTATE

    factory = load_factory(data, "primary")
    config = json.loads(ROTATE.read_text(encoding="utf-8"))
    model = RotatE(triples_factory=factory, embedding_dim=config["training"]["embedding_dim"],
                   loss=MarginRankingLoss(margin=config["training"]["margin"]),
                   random_seed=config["training"]["seed"])
    model.load_state(ROOT / "results/rotate_primary_full/model.pt")
    model.eval()
    ordered_catalog = sorted(catalog, key=recipe_order)
    embedded = [recipe for recipe in ordered_catalog if recipe in factory.entity_to_id]
    ids = torch.tensor([factory.entity_to_id[recipe] for recipe in embedded], dtype=torch.long)
    selected_users = {sorted(train)[position] for position in (0, 2000, 3999)}
    saved = defaultdict(list)
    for row in rows(output / "4_rotate_primary_top50.csv"):
        if row["user_id"] in selected_users:
            saved[row["user_id"]].append(row["recipe_id"])
    with torch.inference_mode():
        for user in sorted(selected_users):
            eligible = [index for index, recipe in enumerate(embedded) if recipe not in train[user]]
            query_user = factory.entity_to_id[user]
            percentiles = []
            for relation in ("RATED_4", "RATED_5"):
                query = torch.tensor([[query_user, factory.relation_to_id[relation]]], dtype=torch.long)
                raw = model.predict_t(query, tails=ids).reshape(-1).cpu().numpy()
                percentiles.append(rankdata(raw[eligible], method="average") / len(eligible))
            scores = 0.5 * (percentiles[0] + percentiles[1])
            expected = [embedded[eligible[index]] for index in
                        np.lexsort((np.asarray(eligible), -scores))[:50]]
            assert saved[user] == expected
    result["4_rotate_primary"]["independently_recomputed_user_lists"] = len(selected_users)
    return {"status": "test_blind_rankings_verified", "preflight_sha256": sha(output / "preflight.json"),
            "conditions": result}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
