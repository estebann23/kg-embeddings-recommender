"""Freeze and verify the RQ1 research contract without fitting a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/hummus_4000_v1"
CONTRACT = ROOT / "config/rq1_experiment.json"
ROTATE = ROOT / "config/rq1_rotate.json"
LOCK = ROOT / "config/rq1_experiment.lock.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def snapshot(data: Path = DATA) -> dict:
    data = Path(data).resolve()
    contract, rotate = read(CONTRACT), read(ROTATE)
    manifest = read(data / "manifest.json")
    check(contract["status"] == rotate["status"] == "frozen", "Both contracts must be frozen")
    check(contract["dataset_id"] == rotate["dataset_id"] == manifest["dataset_id"] == data.name,
          "Dataset ID mismatch")
    check(sha(data / "manifest.json") == contract["source_manifest_sha256"], "Source manifest changed")
    check(contract["cohort"]["users"] == manifest["counts"]["output_users"] and
          contract["cohort"]["recipes"] == manifest["counts"]["output_recipes"], "Cohort size mismatch")
    for split in ("train", "validation", "test"):
        check(contract["cohort"][f"{split}_reviews"] == manifest["counts"][f"{split}_reviews"],
              f"{split} count mismatch")
    check(manifest["parameters"]["split_strategy"] == "chronological" and
          manifest["parameters"]["max_events"] == 40, "Unexpected split protocol")
    check(rotate["ranking"]["weight_five_grid"] ==
          contract["conditions"]["4_rotate"]["validation_grid"]["weight_five"],
          "RotatE weight grid mismatch")
    check(rotate["ranking"]["selected_weight_five"] is None,
          "A validation weight cannot be selected before training")
    check(rotate["training"]["negative_sampling_protocol"] == "typed_tail_training_only_v1" and
          rotate["training"]["negative_samples_per_positive"] == 1, "Negative-sampling protocol changed")
    check(contract["rating_policy"]["ndcg_gain"] ==
          {"1": 0, "2": 0, "3": 0, "4": 1, "5": 2}, "NDCG gains changed")
    check(contract["evaluation"]["k"] == [10, 5], "Evaluation K changed")

    for name in ("users", "recipes", "reviews", "graph_nodes", "graph_edges",
                 "recipe_ingredients", "foodon_links", "foodon_hierarchy"):
        check(sha(data / f"{name}.csv") == manifest["output_sha256"][name], f"Changed {name} file")
    rating_report = read(data / "kge_rating_reconciliation.json")
    check(sha(data / "kge_rating_triples.csv") == rating_report["projection_sha256"],
          "Rating projection changed")
    check(rating_report["held_out_in_projection"] == 0, "Held-out rating leakage")

    recipes = set()
    with (data / "recipes.csv").open(encoding="utf-8", newline="") as stream:
        recipes = {row["recipe_id"] for row in csv.DictReader(stream)}
    check(len(recipes) == contract["cohort"]["recipes"], "Duplicate or missing recipe IDs")
    per_user = defaultdict(Counter)
    pairs = set()
    split_counts = Counter()
    with (data / "reviews.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in {"train", "validation", "test"}:
                continue
            check(row["recipe_id"] in recipes, "Review recipe outside candidate universe")
            pair = (row["user_id"], row["recipe_id"])
            check(pair not in pairs, "User-recipe pair crosses splits or is duplicated")
            pairs.add(pair)
            per_user[row["user_id"]][row["split"]] += 1
            split_counts[row["split"]] += 1
    check(len(per_user) == contract["cohort"]["users"], "Evaluated user count mismatch")
    check(all(counts == {"train": 28, "validation": 4, "test": 8} for counts in per_user.values()),
          "Per-user chronological split sizes changed")
    check(all(split_counts[split] == contract["cohort"][f"{split}_reviews"]
              for split in ("train", "validation", "test")), "Review split count mismatch")

    return {
        "schema_version": 1,
        "dataset_id": data.name,
        "status": "frozen_before_training",
        "contract_sha256": sha(CONTRACT),
        "rotate_config_sha256": sha(ROTATE),
        "description_sha256": sha(ROOT / "description.md"),
        "manual_sha256": sha(ROOT / "MANUAL.md"),
        "source_manifest_sha256": sha(data / "manifest.json"),
        "source_archive_sha256": {name: entry["sha256"] for name, entry in manifest["sources"].items()},
        "rating_projection_sha256": sha(data / "kge_rating_triples.csv"),
        "verified_users": len(per_user),
        "verified_recipes": len(recipes),
        "verified_split_counts": dict(split_counts),
        "note": "Contract frozen and checked before any model or embedding training."
    }


def verify_lock(data: Path = DATA) -> dict:
    expected = snapshot(data)
    actual = read(LOCK)
    check(actual == expected, "Experiment lock differs from current contracts or data")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.command == "freeze":
        value = snapshot(args.data)
        LOCK.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    else:
        value = verify_lock(args.data)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
