"""Descriptive paired-user bootstrap intervals for the fixed taxonomy ablation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

from src.recommenders.contract import sha
from src.recommenders.rq1_rotate_taxonomy_test import OUTPUT
from src.recommenders.verify_rq1_rotate_taxonomy_results import verify


METRICS = ("precision_at_10", "precision_at_50", "ndcg_at_10", "ndcg_at_50")
SEED = 2026
REPLICATES = 5000


def run() -> dict:
    assert verify()["status"] == "taxonomy_accuracy_verified"
    per_user = {}
    with (OUTPUT / "per_user_accuracy.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            per_user[row["user_id"], row["condition"]] = row
    users = sorted({user for user, _ in per_user})
    assert len(users) == 4000 and len(per_user) == 8000
    differences = np.asarray([[float(per_user[user, "4_rotate_hierarchy"][metric]) -
                               float(per_user[user, "4_rotate_primary"][metric])
                               for metric in METRICS] for user in users], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    estimates = np.empty((REPLICATES, len(METRICS)), dtype=np.float64)
    for start in range(0, REPLICATES, 250):
        stop = min(start + 250, REPLICATES)
        sample = rng.integers(0, len(users), size=(stop - start, len(users)))
        estimates[start:stop] = differences[sample].mean(axis=1)
    path = OUTPUT / "paired_bootstrap_intervals.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("metric", "users", "mean_hierarchy_minus_primary", "ci_2.5_percent",
                         "ci_97.5_percent", "bootstrap_replicates", "seed"))
        for position, metric in enumerate(METRICS):
            lower, upper = np.quantile(estimates[:, position], [0.025, 0.975])
            writer.writerow((metric, len(users), format(differences[:, position].mean(), ".17g"),
                             format(lower, ".17g"), format(upper, ".17g"), REPLICATES, SEED))
    result = {"status": "paired_bootstrap_complete", "users": len(users),
              "replicates": REPLICATES, "seed": SEED,
              "per_user_accuracy_sha256": sha(OUTPUT / "per_user_accuracy.csv"),
              "intervals_sha256": sha(path),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "Descriptive user-resampling intervals for a fixed post-test comparison; no model selection."}
    (OUTPUT / "uncertainty_manifest.json").write_text(json.dumps(result, indent=2) + "\n",
                                                       encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
