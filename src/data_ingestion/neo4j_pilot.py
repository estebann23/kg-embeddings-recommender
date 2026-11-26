# Load a HUMMUS export into a dataset / Neo4j namespace

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.data_ingestion.validate import validate
from src.kg.connection import Neo4jConnection


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/hummus_pilot_v1"
DATASET_LABELS = {"hummus_pilot_v1": "HummusPilot", "hummus_4000_v1": "HummusCohort"}
NODE_LABELS = {"ObservedUser", "Recipe", "Ingredient", "FoodOnConcept"}
RELATIONS = {"RATED", "CONTAINS_MAPPED_INGREDIENT", "MAPS_TO_FOODON", "SUBCLASS_OF"}


def scoped_id(dataset_id: str, entity_id: str) -> str:
    """Avoid global label-ID constraints colliding across dataset snapshots."""
    return entity_id if dataset_id == "hummus_pilot_v1" else f"{dataset_id}::{entity_id}"


def batches(path: Path, size: int):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def review_dates(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["review_id"]: row["submit_date"]
                for row in csv.DictReader(stream) if row["split"] == "train"}


def load(output: Path = DEFAULT_OUTPUT, batch_size: int = 500) -> dict:
    output = output.resolve()
    if batch_size < 1 or batch_size > 5000:
        raise ValueError("batch_size must be 1..5000")
    validate(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    dataset_id = manifest["dataset_id"]
    if dataset_id not in DATASET_LABELS:
        raise ValueError("Unsupported HUMMUS dataset ID")
    namespace = DATASET_LABELS[dataset_id]
    manager = "hummus-pilot-ingestion" if dataset_id == "hummus_pilot_v1" else "hummus-cohort-ingestion"
    manifest_hash = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    dates = review_dates(output / "reviews.csv")
    with Neo4jConnection() as db:
        db.write(f"CREATE CONSTRAINT {namespace.lower()}_identity IF NOT EXISTS "
                 f"FOR (n:{namespace}) REQUIRE (n.dataset_id, n.id) IS UNIQUE")
        stale = db.read(f"MATCH (n:{namespace} {{dataset_id: $dataset_id}}) "
                        "WHERE n.manifest_sha256 IS NULL OR n.manifest_sha256 <> $manifest_hash "
                        "RETURN count(n) AS count", dataset_id=dataset_id, manifest_hash=manifest_hash)[0]["count"]
        if stale:
            raise ValueError(f"Existing {dataset_id} nodes belong to a different manifest; use a new versioned dataset ID")
        for batch in batches(output / "graph_nodes.csv", batch_size):
            groups = {}
            for row in batch:
                label = row["label"]
                if label not in NODE_LABELS:
                    raise ValueError(f"Unexpected node label {label}")
                row["entity_id"] = row["id"]
                row["id"] = scoped_id(dataset_id, row["id"])
                groups.setdefault(label, []).append(row)
            for label, rows in groups.items():
                db.write(
                    f"UNWIND $rows AS row MERGE (n:{namespace} {{dataset_id: $dataset_id, id: row.id}}) "
                    f"SET n:{label} SET n.name = row.name, n.entity_id = row.entity_id, "
                    "n.source_id = row.source_id, n.external_iri = row.external_iri, "
                    "n.managed_by = $manager, n.manifest_sha256 = $manifest_hash",
                    rows=rows, dataset_id=dataset_id, manifest_hash=manifest_hash, manager=manager,
                )
        if "recipe_nutrients" in manifest["output_sha256"]:
            numeric_fields = ("serving_size_g", "calories", "total_fat_g", "saturated_fat_g",
                              "cholesterol_mg", "sodium_mg", "total_carbohydrate_g",
                              "dietary_fiber_g", "sugars_g", "protein_g")
            assignments = ", ".join(f"n.{field} = CASE WHEN row.{field} = '' THEN null ELSE toFloat(row.{field}) END"
                                    for field in numeric_fields)
            for batch in batches(output / "recipe_nutrients.csv", batch_size):
                for row in batch:
                    row["recipe_id"] = scoped_id(dataset_id, row["recipe_id"])
                result = db.write(
                    f"UNWIND $rows AS row MATCH (n:Recipe:{namespace} {{dataset_id: $dataset_id, id: row.recipe_id}}) "
                    f"SET {assignments}, n.nutrition_parse_status = row.parse_status, "
                    "n.nutrition_missing_fields = row.missing_fields, n.nutrition_source_file = row.source_file "
                    "RETURN count(n) AS count", rows=batch, dataset_id=dataset_id)
                if result[0]["count"] != len(batch):
                    raise ValueError("A nutrition batch did not match all recipe nodes")
        for batch in batches(output / "graph_edges.csv", batch_size):
            groups = {}
            for row in batch:
                relation = row["relation"]
                if relation not in RELATIONS:
                    raise ValueError(f"Unexpected relation {relation}")
                row["submit_date"] = dates[row["review_id"]] if relation == "RATED" else ""
                row["entity_id"] = row["id"]
                row["id"] = scoped_id(dataset_id, row["id"])
                row["source_id"] = scoped_id(dataset_id, row["source_id"])
                row["target_id"] = scoped_id(dataset_id, row["target_id"])
                groups.setdefault(relation, []).append(row)
            for relation, rows in groups.items():
                result = db.write(
                    f"UNWIND $rows AS row MATCH (s:{namespace} {{dataset_id: $dataset_id, id: row.source_id}}) "
                    f"MATCH (t:{namespace} {{dataset_id: $dataset_id, id: row.target_id}}) "
                    f"MERGE (s)-[r:{relation} {{id: row.id}}]->(t) "
                    "SET r.dataset_id = $dataset_id, r.entity_id = row.entity_id, r.review_id = row.review_id, "
                    "r.rating = CASE WHEN row.rating = '' THEN null ELSE toInteger(row.rating) END, "
                    "r.submit_date = row.submit_date, r.source_file = row.source_file, "
                    "r.evidence_state = row.evidence_state, r.managed_by = $manager, "
                    "r.manifest_sha256 = $manifest_hash RETURN count(r) AS count",
                    rows=rows, dataset_id=dataset_id, manifest_hash=manifest_hash, manager=manager,
                )
                if result[0]["count"] != len(rows):
                    raise ValueError("An edge batch did not match all endpoints")
        node_count = db.read(f"MATCH (n:{namespace} {{dataset_id: $id}}) RETURN count(n) AS count", id=dataset_id)[0]["count"]
        edge_count = db.read(f"MATCH (s:{namespace} {{dataset_id: $id}})-[r]->(t:{namespace} {{dataset_id: $id}}) "
                             "WHERE r.dataset_id = $id RETURN count(r) AS count", id=dataset_id)[0]["count"]
    expected_nodes = manifest["counts"]["output_graph_nodes"]
    expected_edges = manifest["counts"]["output_graph_edges"]
    if node_count != expected_nodes or edge_count != expected_edges:
        raise ValueError(f"Neo4j counts {node_count}/{edge_count} differ from export {expected_nodes}/{expected_edges}")
    return {"dataset_id": dataset_id, "nodes": node_count, "edges": edge_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(load(output=args.output, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
