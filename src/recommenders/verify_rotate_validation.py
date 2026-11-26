"""Independently verify saved RotatE validation-only weight selections."""

from __future__ import annotations

import argparse
import json
import math

from src.recommenders.contract import ROOT, sha, verify_lock
from src.recommenders.rotate_ranking import WEIGHT_GRID
from src.recommenders.verify_rotate_training import verify as verify_training


def verify(variant: str) -> dict:
    verify_lock()
    trained = verify_training(variant)
    output = ROOT / "results" / f"rotate_{variant}_full"
    selection = json.loads((output / "validation_selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "validation_selected_no_test_access"
    assert selection["variant"] == variant
    assert selection["model_sha256"] == trained["model_sha256"] == sha(output / "model.pt")
    assert selection["training_manifest_sha256"] == sha(output / "training_manifest.json")
    assert selection["contract_lock_sha256"] == sha(ROOT / "config/rq1_experiment.lock.json")
    assert selection["users_with_positive_validation"] == 3995
    values = selection["mean_ndcg_at_10_by_weight"]
    assert set(values) == {str(weight) for weight in WEIGHT_GRID}
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in values.values())
    winner = max(WEIGHT_GRID, key=lambda weight:
                 (values[str(weight)], -abs(weight - 0.5), -weight))
    assert selection["selected_weight_five"] == winner
    assert selection["selected_weight_four"] == 1 - winner
    return {"variant": variant, "weight_five": winner,
            "validation_mean_ndcg_at_10": values[str(winner)],
            "users_with_positive_validation": selection["users_with_positive_validation"],
            "status": "validation_selection_verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("primary", "hierarchy"), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.variant), indent=2))


if __name__ == "__main__":
    main()
