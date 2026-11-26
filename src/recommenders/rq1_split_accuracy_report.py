"""Render a descriptive report from verified five-model split accuracy tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.recommenders.contract import ROOT, sha
from src.recommenders.verify_rq1_split_accuracy_5models import verify

OUTPUT = ROOT / "results/rq1_split_accuracy_5models"
NAMES = {
    "interaction_only_cf": "Interaction-only CF",
    "weighted_graph_paths": "Weighted graph paths",
    "semantic_mf": "Semantic MF",
    "rotate_primary": "RotatE primary",
    "rotate_foodon_taxonomy": "RotatE + FoodOn taxonomy",
}
POPULATIONS = {
    "random_50": "Random half",
    "semantic_50": "Semantic half",
    "exact_ingredient": "Exact ingredient",
    "direct_foodon_only": "Direct FoodOn only",
    "foodon_parent_only": "FoodOn parent only",
    "no_observed_overlap": "No observed overlap",
    "insufficient_ingredient_mapping": "Insufficient mapping",
}
METRICS = ("precision_at_10", "precision_at_50", "recall_at_10", "recall_at_50",
           "ndcg_at_10", "ndcg_at_50")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def render(output=OUTPUT):
    output = Path(output).resolve()
    audit = verify(output=output)
    assert audit["status"] == "PASS"
    data = read_csv(output / "aggregate_accuracy.csv")
    grid = read_csv(output / "validation_grid.csv")
    seal = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    by_key = {(row["population"], row["model"]): row for row in data}
    selected = {row["model"]: row for row in grid if row["selected"] == "1"}
    assert len(selected) == 5 and len(data) == 35
    lines = [
        "# Five-model RQ1 accuracy on the random and semantic test split",
        "",
        "## Scope and execution order",
        "",
        "The existing 4,000-user HUMMUS chronological split remains 28 training, 4 validation, and 8 test ratings per user. The saved test subdivision assigns exactly four test ratings per user to a fixed-seed random half and four to a semantic half. The semantic half is partitioned into five mutually exclusive graph-evidence bins. All **five** named models were evaluated, as confirmed by the user; the split is **50%/50%**. No model was retrained or retuned from test labels for this evaluation.",
        "",
        "Before calculating these split metrics, the run verified the frozen dataset and split manifests, the five existing validation-selected settings, and all five test-blind top-50 ranking hashes. The earlier validation grids were reused rather than changed after prior test work. Because the semantic MF comparator was added after the original four-condition test work, this five-model comparison is retrospective. The full candidate universe is 75,972 cohort recipes, with each user's 28 training-rated recipes excluded. Validation and test targets remain eligible. No RQ2 dietary filter is applied.",
        "",
        "## Models, tuning formulas, and selected values",
        "",
        "| Model | Prediction/ranking operation | Validation-tuned parameter(s) and tried values | Selected setting | Selected validation mean NDCG@10 |",
        "| --- | --- | --- | --- | ---: |",
        f"| Interaction-only CF | Positive part of cosine similarity between signed 1–5 training-rating item vectors; weighted mean over the top \(N\) rated neighbors. | \(N\) = 5, 10, 20. | \(N=5\) (all three tied; smaller wins). | {float(selected['interaction_only_cf']['validation_mean_ndcg_at_10']):.9f} |",
        f"| Weighted graph paths | Training 4–5-star recipe profile; rating-gain-weighted typed-path similarity \(0.5J_{{ingredient}}+0.5[(1-\\beta)J_{{direct}}+\\beta J_{{parent}}]\). | \(\\beta\) = 0, 0.25, 0.5. | \(\\beta=0.5\). | {float(selected['weighted_graph_paths']['validation_mean_ndcg_at_10']):.9f} |",
        f"| Semantic MF | Global/user/item biases plus user vector dotted with free recipe factor and scaled mean ingredient/direct-FoodOn factors. | Content scale 0.5, 1.0; regularization 0.01, 0.1; epochs 10, 20; factors fixed at 16. | Scale 0.5; reg. 0.01; epoch 10. | {float(selected['semantic_mf']['validation_mean_ndcg_at_10']):.9f} |",
        f"| RotatE primary | Mean of within-user percentile ranks of `RATED_4` and `RATED_5` triple scores with weights \(1-w_5\) and \(w_5\). | \(w_5\) = 0, 0.25, 0.5, 0.75, 1. | \(w_5=0.5\). | {float(selected['rotate_primary']['validation_mean_ndcg_at_10']):.9f} |",
        f"| RotatE + FoodOn taxonomy | Same score rule, trained with the verified FoodOn `SUBCLASS_OF` triples added. | \(w_5\) = 0, 0.25, 0.5, 0.75, 1. | \(w_5=0.75\). | {float(selected['rotate_foodon_taxonomy']['validation_mean_ndcg_at_10']):.9f} |",
        "",
        "For CF, each observed training star is mapped to a signed weight (1→−1, 2→−0.5, 3→0, 4→0.5, 5→1); candidate score is the similarity-weighted mean over its top \(N\) matching history recipes. The graph-path \(J\) terms are recipe-frequency-weighted Jaccard similarities of audited ingredient, direct-concept, and bounded one-step parent sets; only training 4–5-star recipes start positive paths. The semantic MF equation and loss are recorded in the [semantic MF validation report](../mf_semantic_hummus_4000_v1/report.md). RotatE uses \(S=(1-w_5)p_4+w_5p_5\), where \(p_4,p_5\) are within-user percentile ranks of the two learned positive-relation scores. The taxonomy variant adds only verified `SUBCLASS_OF` training triples.",
        "",
        "The RotatE embedding architecture and training budget were fixed in their existing contracts; the validation grid selected the two positive-rating score weights, not embedding dimensions or epochs. For semantic MF, the factor size, optimizer, learning rate, and batch size were fixed to the prior ratings-only MF baseline. All selected values above maximize mean validation NDCG@10 under their declared tie rules. The complete tried settings and values are in [validation_grid.csv](validation_grid.csv).",
        "",
        "## Test populations and denominators",
        "",
        "| Population | Users with targets | Users with positive targets | Test ratings | Positive ratings (4–5) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    first = "interaction_only_cf"
    for population, name in POPULATIONS.items():
        row = by_key[population, first]
        lines.append(f"| {name} | {row['users_with_targets']} | {row['users_with_positive_targets']} | {row['targets']} | {row['positive_targets']} |")
    lines += [
        "",
        "The random and semantic halves each contain 16,000 ratings, with four per user. The five semantic-bin rating counts sum to 16,000. Each test recipe is compared with **all 28 of that user's training-rated recipes**, regardless of star value. Bin precedence is: shared mapped ingredient; no exact ingredient match but shared direct FoodOn concept; neither preceding match but shared source-asserted one-step FoodOn parent (parents on more than 5% of mapped recipes are omitted); mapped ingredient evidence but no overlap in those recorded link families; insufficient mapped ingredient evidence for the test recipe or all training recipes. Bin users may appear in more than one bin because their four semantic ratings can have different graph profiles. Bin metrics use only users with at least one target in that bin.",
        "",
        "## Metric definitions",
        "",
        "For a user and one population, an observed positive is a held-out rating of 4 or 5 **inside that population**. `hits@K` counts such recipes in the model's unchanged top-K list. **Precision@K** is `hits@K / K`, averaged over users with at least one target in the population. **Recall@K** is `hits@K / number of observed positives in that population`, averaged only over users with at least one positive. **NDCG@K** is discounted gain divided by the ideal discounted gain for that user's positive targets in that population, also averaged only over positive-target users. Ratings 1–3 have gain 0, rating 4 has gain 1, and rating 5 has gain 2. The discount at rank \(r\) is \(1/\\log_2(r+1)\).",
        "",
        "The fixed top-50 ranking is reused for every population. A recipe labelled in another half or bin remains a candidate but has zero **observed gain for the current population**; this does not designate it as a confirmed dislike. A user with no positive target contributes zero to Precision and has undefined Recall/NDCG, so that user is excluded from those two macro means. Values below are fractions, not percentages. The CSV files preserve full numeric precision.",
        "",
        "## Accuracy results: two test halves",
        "",
    ]

    def append_metrics(populations):
        lines.append("| Population | Model | Hits@10 | Hits@50 | Precision@10 | Precision@50 | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for population in populations:
            for model in NAMES:
                row = by_key[population, model]
                metrics = " | ".join(f"{float(row[key]):.8f}" for key in METRICS)
                # CSV order is P10,P50,R10,R50,N10,N50.
                lines.append(f"| {POPULATIONS[population]} | {NAMES[model]} | {row['hits_at_10']} | {row['hits_at_50']} | {metrics} |")

    append_metrics(("random_50", "semantic_50"))
    lines += ["", "## Accuracy results: five semantic bins", ""]
    append_metrics(tuple(POPULATIONS)[2:])
    lines += [
        "",
        "## Files and checks",
        "",
        "- [aggregate_accuracy.csv](aggregate_accuracy.csv): one row for each model and population (35 rows), with user/target denominators, hit counts, and the six requested metrics.",
        "- [per_user_accuracy.csv](per_user_accuracy.csv): one row for each model, population, and user with a target (72,805 rows). Undefined Recall/NDCG cells are blank.",
        "- [validation_grid.csv](validation_grid.csv): all previously tried validation values and the selected setting for each model.",
        "- [experiment_config.json](experiment_config.json), [preflight.json](preflight.json), and [evaluation_manifest.json](evaluation_manifest.json): population rules, source/model/split hashes, selections, output hashes, and counts.",
        "",
        "The evaluation implementation is [`src/recommenders/rq1_split_accuracy_5models.py`](../../src/recommenders/rq1_split_accuracy_5models.py). The independent [`verify_rq1_split_accuracy_5models.py`](../../src/recommenders/verify_rq1_split_accuracy_5models.py) re-read the source test reviews and fixed rank files, recalculated all **72,805** per-user rows and **35** aggregate rows, and checked the random/semantic hit-count sum against **16,000** earlier complete-test user/model records for the four previously evaluated models. All checks passed. This report lists descriptive measurements and implementation choices only; it makes no result inference.",
        "",
        f"Report input fingerprints: preflight SHA-256 `{sha(output / 'preflight.json')}`; aggregate CSV SHA-256 `{sha(output / 'aggregate_accuracy.csv')}`.",
        "",
    ]
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "report_written", "path": str(path), "sha256": sha(path),
            "data_rows": len(data), "metric_columns": len(METRICS)}


if __name__ == "__main__":
    print(json.dumps(render(), indent=2))
