"""Check a completed RotatE run against its frozen pretraining contract."""

from __future__ import annotations

import argparse
import json
import math

import torch

from src.recommenders.contract import DATA, ROOT, ROTATE, sha, verify_lock
from src.recommenders.rotate import load_factory, read_json, require


def verify(variant: str) -> dict:
    require(variant in {"primary", "hierarchy"}, "Unknown variant")
    verify_lock()
    config = read_json(ROTATE)
    preflight = read_json(DATA / f"rotate_{variant}_preflight.json")
    output = ROOT / "results" / f"rotate_{variant}_full"
    manifest = read_json(output / "training_manifest.json")
    factory = load_factory(DATA, variant)
    require(manifest["status"] == "trained_not_validated", "Run has not completed fitting")
    require(manifest["variant"] == variant and
            manifest["epochs"] == config["training"]["epochs"] == 100,
            "Variant or epoch count mismatch")
    require(len(manifest["losses"]) == 100 and
            all(math.isfinite(loss) for loss in manifest["losses"]),
            "Missing or non-finite epoch loss")
    require(manifest["config_sha256"] == sha(ROTATE) == preflight["config_sha256"] and
            manifest["triple_sha256"] == preflight["triple_sha256"] and
            manifest["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json"),
            "Frozen input hash mismatch")
    entity_ids = read_json(output / "entity_to_id.json")
    relation_ids = read_json(output / "relation_to_id.json")
    require(entity_ids == factory.entity_to_id and relation_ids == factory.relation_to_id,
            "Saved ID mapping differs from training factory")
    # PyKEEN writes protocol-5 pickle; these are locally generated, trusted files.
    state = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    require(state and all(torch.isfinite(tensor).all().item() for tensor in state.values()
                          if isinstance(tensor, torch.Tensor)), "Non-finite model parameter")
    checkpoint = torch.load(output / "checkpoints/training.pt", map_location="cpu", weights_only=False)
    require(checkpoint["epoch"] == 100 and checkpoint["random_seed"] == config["training"]["seed"],
            "Final checkpoint epoch or seed mismatch")
    require(checkpoint["entity_to_id_dict"] == factory.entity_to_id and
            checkpoint["relation_to_id_dict"] == factory.relation_to_id,
            "Checkpoint ID mappings differ from audited triples")
    require(all(group["lr"] == config["training"]["learning_rate"]
                for group in checkpoint["optimizer_state_dict"]["param_groups"]),
            "Checkpoint optimizer learning rate differs from frozen config")
    require(checkpoint["loss"] == manifest["losses"],
            "Final checkpoint losses differ from training manifest")
    saved_state = checkpoint["model_state_dict"]
    require(state.keys() == saved_state.keys() and
            all(torch.equal(tensor, saved_state[name]) for name, tensor in state.items()),
            "Final model weights differ from epoch-100 checkpoint")
    return {"variant": variant, "verified_epochs": 100,
            "first_loss": manifest["losses"][0], "last_loss": manifest["losses"][-1],
            "entities": factory.num_entities, "relations": factory.num_relations,
            "model_sha256": sha(output / "model.pt"), "status": "training_verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("primary", "hierarchy"), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.variant), indent=2))


if __name__ == "__main__":
    main()
