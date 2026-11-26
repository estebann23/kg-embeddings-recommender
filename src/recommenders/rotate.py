"""Audited RotatE triple preparation and a guarded PyKEEN training entry point.

Preparation uses only the Python standard library. It never fits embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/processed/hummus_4000_v1"
DEFAULT_CONFIG = ROOT / "config/rq1_rotate.json"
RATING_RELATIONS = {f"RATED_{rating}" for rating in range(1, 6)}
GRAPH_RELATIONS = {"CONTAINS_MAPPED_INGREDIENT", "MAPS_TO_FOODON", "SUBCLASS_OF"}
ENDPOINT_LABELS = {
    **{relation: ("ObservedUser", "Recipe") for relation in RATING_RELATIONS},
    "CONTAINS_MAPPED_INGREDIENT": ("Recipe", "Ingredient"),
    "MAPS_TO_FOODON": ("Ingredient", "FoodOnConcept"),
    "SUBCLASS_OF": ("FoodOnConcept", "FoodOnConcept"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def selected_relations(config: dict, hierarchy: bool) -> list[str]:
    relations = config["primary_relations"].copy()
    expected = RATING_RELATIONS | {"CONTAINS_MAPPED_INGREDIENT", "MAPS_TO_FOODON"}
    require(len(relations) == len(set(relations)) and set(relations) == expected,
            "Primary relations must be the five ratings, ingredient, and direct FoodOn links")
    require(config["nutrition_relations"] == [], "Nutrition needs a separate frozen encoding")
    if hierarchy:
        require(config["hierarchy_ablation_relation"] == "SUBCLASS_OF", "Unknown hierarchy relation")
        relations.append("SUBCLASS_OF")
    return relations


def verify_sources(data: Path, config: dict) -> tuple[dict, dict]:
    manifest = read_json(data / "manifest.json")
    report = read_json(data / "kge_rating_reconciliation.json")
    require(manifest["dataset_id"] == config["dataset_id"] == data.name, "Dataset mismatch")
    require(sha256(data / "manifest.json") == report["manifest_sha256"], "Rating report manifest hash mismatch")
    for name in ("graph_nodes", "graph_edges", "reviews"):
        actual = sha256(data / f"{name}.csv")
        require(actual == manifest["output_sha256"][name], f"Changed source: {name}")
        if name in report["source_sha256"]:
            require(actual == report["source_sha256"][name], f"Rating report source mismatch: {name}")
    require(sha256(data / "kge_rating_triples.csv") == report["projection_sha256"],
            "Changed rating projection")
    require(report["held_out_in_projection"] == 0, "Rating projection contains held-out data")
    require(report["projection_triples"] == manifest["counts"]["train_reviews"],
            "Training rating count mismatch")
    return manifest, report


def prepare(data: Path = DEFAULT_DATA, config_path: Path = DEFAULT_CONFIG,
            hierarchy: bool = False) -> dict:
    data, config_path = Path(data).resolve(), Path(config_path).resolve()
    config = read_json(config_path)
    relations = selected_relations(config, hierarchy)
    manifest, report = verify_sources(data, config)

    node_labels = {}
    with (data / "graph_nodes.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["id"] not in node_labels, f"Duplicate node: {row['id']}")
            node_labels[row["id"]] = row["label"]
    require(len(node_labels) == manifest["counts"]["output_graph_nodes"], "Node count mismatch")

    review_by_id = {}
    held_out_pairs = set()
    with (data / "reviews.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "train":
                require(row["review_id"] not in review_by_id, "Duplicate training review ID")
                review_by_id[row["review_id"]] = row
            elif row["split"] in {"validation", "test"}:
                held_out_pairs.add((row["user_id"], row["recipe_id"]))
    require(len(review_by_id) == report["training_reviews"], "Training review count mismatch")

    counts = Counter()
    seen = set()
    rated_pairs = set()
    entities = set()
    users = set()
    recipes = set()
    variant = "hierarchy" if hierarchy else "primary"
    output = data / f"rotate_{variant}_train.tsv"
    # A temporary file prevents a partially checked export from looking complete.
    temp = output.with_suffix(".tsv.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink, delimiter="\t", lineterminator="\n")
            with (data / "kge_rating_triples.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    head, relation, tail = row["source_id"], row["relation"], row["target_id"]
                    require(relation in RATING_RELATIONS, "Unknown rating relation")
                    review = review_by_id.get(row["review_id"])
                    require(review is not None and review["split"] == "train", "Rating lacks training review")
                    require((head, tail) == (review["user_id"], review["recipe_id"]), "Rating endpoint mismatch")
                    require(row["rating"] == review["rating"] and relation == f"RATED_{review['rating']}",
                            "Rating value or relation mismatch")
                    require((head, tail) not in held_out_pairs, "Held-out pair in training")
                    require((head, tail) not in rated_pairs, "Duplicate user-recipe training pair")
                    require(node_labels.get(head) == "ObservedUser" and node_labels.get(tail) == "Recipe",
                            "Rating endpoint type mismatch")
                    rated_pairs.add((head, tail))
                    writer.writerow((head, relation, tail))
                    counts[relation] += 1
                    users.add(head)
                    recipes.add(tail)
                    entities.update((head, tail))
            require(len(rated_pairs) == report["projection_triples"], "Projection count mismatch")
            for rating in range(1, 6):
                require(counts[f"RATED_{rating}"] == report["ratings_by_value"][str(rating)],
                        f"Rating {rating} count mismatch")

            with (data / "graph_edges.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    relation = row["relation"]
                    if relation not in GRAPH_RELATIONS or relation not in relations:
                        continue
                    head, tail = row["source_id"], row["target_id"]
                    expected_head, expected_tail = ENDPOINT_LABELS[relation]
                    require(node_labels.get(head) == expected_head and node_labels.get(tail) == expected_tail,
                            f"Endpoint type mismatch for {relation}")
                    require(not row["review_id"] and not row["rating"], "Semantic edge carries rating evidence")
                    triple = (head, relation, tail)
                    require(triple not in seen, f"Duplicate semantic triple: {triple}")
                    seen.add(triple)
                    writer.writerow(triple)
                    counts[relation] += 1
                    entities.update((head, tail))
        require(counts["CONTAINS_MAPPED_INGREDIENT"] == manifest["counts"]["mapped_ingredient_relations"],
                "Ingredient relation count mismatch")
        require(counts["MAPS_TO_FOODON"] == manifest["counts"]["foodkg_core_foodon_links"],
                "FoodOn relation count mismatch")
        if hierarchy:
            require(counts["SUBCLASS_OF"] == manifest["counts"]["output_foodon_hierarchy"],
                    "Hierarchy relation count mismatch")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)

    result = {
        "schema_version": 1,
        "status": "prepared_not_trained",
        "dataset_id": config["dataset_id"],
        "variant": variant,
        "config_sha256": sha256(config_path),
        "source_sha256": {name: sha256(data / name) for name in
                          ("manifest.json", "kge_rating_reconciliation.json", "kge_rating_triples.csv")},
        "triple_file": output.name,
        "triple_sha256": sha256(output),
        "relation_counts": dict(sorted(counts.items())),
        "triple_count": sum(counts.values()),
        "entity_count": len(entities),
        "training_users": len(users),
        "training_recipes": len(recipes),
        "scorable_recipes": sum(node_labels.get(entity) == "Recipe" for entity in entities),
        "cohort_recipes": manifest["counts"]["output_recipes"],
        "note": "Only training ratings and selected recipe facts; no PyKEEN model has been fitted."
    }
    result["unscorable_recipes"] = result["cohort_recipes"] - result["scorable_recipes"]
    report_path = data / f"rotate_{variant}_preflight.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_factory(data: Path = DEFAULT_DATA, variant: str = "primary"):
    """Load the prepared training triples without splitting or training them."""
    from pykeen.triples import TriplesFactory

    data = Path(data).resolve()
    report = read_json(data / f"rotate_{variant}_preflight.json")
    path = data / report["triple_file"]
    require(sha256(path) == report["triple_sha256"], "Prepared triple hash mismatch")
    factory = TriplesFactory.from_path(path, create_inverse_triples=False)
    require(factory.num_triples == report["triple_count"], "PyKEEN triple count mismatch")
    require(set(report["relation_counts"]) == set(factory.relation_to_id), "PyKEEN relation mapping mismatch")
    return factory


def score_positive_relations(model, factory, user_id: str, recipe_ids: list[str],
                             batch_size: int = 4096) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Return raw RotatE scores, keeping 4- and 5-star relations separate.

    Ranking, candidate exclusions, score calibration, and fallback must be
    applied by the frozen experiment protocol. Higher PyKEEN scores are better.
    """
    import torch

    require(batch_size > 0, "Batch size must be positive")
    require(user_id in factory.entity_to_id, "User has no training entity")
    available = [recipe for recipe in recipe_ids if recipe in factory.entity_to_id]
    missing = [recipe for recipe in recipe_ids if recipe not in factory.entity_to_id]
    require(len(recipe_ids) == len(set(recipe_ids)), "Duplicate candidate recipe")
    scores = {relation: {} for relation in ("RATED_4", "RATED_5")}
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for relation in scores:
            require(relation in factory.relation_to_id, f"Missing relation: {relation}")
            for start in range(0, len(available), batch_size):
                batch = available[start:start + batch_size]
                triples = torch.tensor(
                    [(factory.entity_to_id[user_id], factory.relation_to_id[relation],
                      factory.entity_to_id[recipe]) for recipe in batch],
                    dtype=torch.long, device=device,
                )
                values = model.predict_hrt(triples).reshape(-1).detach().cpu().tolist()
                scores[relation].update(zip(batch, values))
    return scores, missing


def train(data: Path = DEFAULT_DATA, config_path: Path = DEFAULT_CONFIG,
          variant: str = "primary", output: Path | None = None):
    """Train only on the locked training triple factory; save recoverable outputs."""
    config = read_json(Path(config_path))
    require(config["status"] == "frozen", "Training requires a frozen experiment contract")
    training = config["training"]
    ranking = config["ranking"]
    require(training["negative_sampling_protocol"] is not None and
            training["negative_samples_per_positive"] is not None and
            ranking["combination"] is not None and ranking["weight_five_grid"] is not None and
            ranking["unscorable_policy"] is not None,
            "Freeze negative sampling and ranking choices before training")
    require(training["negative_sampling_protocol"] == "typed_tail_training_only_v1",
            "Unsupported negative sampling protocol")
    require(training["loss"] == "MarginRankingLoss" and training["margin"] == 1.0,
            "Unsupported loss protocol")
    require(ranking["combination"] == "within_user_relation_percentile_weighted_sum" and
            ranking["weight_five_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0],
            "Unsupported ranking protocol")
    require(variant in {"primary", "hierarchy"}, "Unknown variant")
    require(output is not None, "Specify a model output directory")
    data = Path(data).resolve()
    config_path = Path(config_path).resolve()
    preflight = read_json(data / f"rotate_{variant}_preflight.json")
    require(preflight["config_sha256"] == sha256(config_path),
            "Configuration changed after preflight; prepare again")
    verify_sources(data, config)
    from src.recommenders.contract import ROTATE, verify_lock
    require(config_path == ROTATE.resolve(), "Training must use the locked RotatE configuration")
    verify_lock(data)

    from pykeen.models import RotatE
    from pykeen.losses import MarginRankingLoss
    from pykeen.training import SLCWATrainingLoop
    from pykeen.utils import set_random_seed
    from src.recommenders.rotate_sampling import TypedTailNegativeSampler, build_sampling_inputs

    factory = load_factory(data, variant)
    sampling_inputs = build_sampling_inputs(factory, data / "graph_nodes.csv")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_random_seed(training["seed"])
    model = RotatE(triples_factory=factory, embedding_dim=training["embedding_dim"],
                   loss=MarginRankingLoss(margin=training["margin"]),
                   random_seed=training["seed"])
    loop = SLCWATrainingLoop(
        model=model, triples_factory=factory,
        optimizer="Adam", optimizer_kwargs={"lr": training["learning_rate"]},
        negative_sampler=TypedTailNegativeSampler,
        negative_sampler_kwargs={"num_negs_per_pos": training["negative_samples_per_positive"],
                                 **sampling_inputs},
    )
    losses = loop.train(
        triples_factory=factory, num_epochs=training["epochs"], batch_size=training["batch_size"],
        checkpoint_directory=output / "checkpoints", checkpoint_name="training.pt",
        checkpoint_frequency=1, checkpoint_on_failure=True, pin_memory=False,
        use_tqdm=False, use_tqdm_batch=False,
    )
    model.save_state(output / "model.pt")
    (output / "entity_to_id.json").write_text(json.dumps(factory.entity_to_id, sort_keys=True), encoding="utf-8")
    (output / "relation_to_id.json").write_text(json.dumps(factory.relation_to_id, sort_keys=True), encoding="utf-8")
    result = {"variant": variant, "epochs": training["epochs"], "losses": losses,
              "config_sha256": sha256(config_path), "triple_sha256": preflight["triple_sha256"],
              "contract_lock_sha256": sha256(ROOT / "config/rq1_experiment.lock.json"),
              "status": "trained_not_validated"}
    (output / "training_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check-factory", "train"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=("primary", "hierarchy"), default="primary")
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.data, args.config, args.variant == "hierarchy"), indent=2))
    elif args.command == "check-factory":
        factory = load_factory(args.data, args.variant)
        print(json.dumps({"num_triples": factory.num_triples, "num_entities": factory.num_entities,
                          "relations": sorted(factory.relation_to_id)}, indent=2))
    else:
        train(args.data, args.config, args.variant, args.model_output)


if __name__ == "__main__":
    main()
