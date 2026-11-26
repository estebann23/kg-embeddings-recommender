# Verify the HUMMUS extract dataset against the Neo4j export

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.data_ingestion.neo4j_pilot import DATASET_LABELS, batches, scoped_id
from src.data_ingestion.validate import validate
from src.kg.connection import Neo4jConnection


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/hummus_4000_v1"
RELATIONS = {"RATED", "CONTAINS_MAPPED_INGREDIENT", "MAPS_TO_FOODON", "SUBCLASS_OF"}


def verify(output: Path = DEFAULT_OUTPUT, batch_size: int = 1000) -> dict:
    output = output.resolve()
    if not 1 <= batch_size <= 5000:
        raise ValueError("batch_size must be 1..5000")
    local = validate(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = manifest["dataset_id"]
    namespace = DATASET_LABELS[dataset_id]
    manager = "hummus-pilot-ingestion" if dataset_id == "hummus_pilot_v1" else "hummus-cohort-ingestion"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with (output / "reviews.csv").open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    train_dates = {row["review_id"]: row["submit_date"] for row in reviews if row["split"] == "train"}
    train_rating_counts = Counter(int(row["rating"]) for row in reviews if row["split"] == "train")
    held_out_ids = [row["review_id"] for row in reviews if row["split"] in {"validation", "test"}]
    node_labels, edge_relations = Counter(), Counter()
    with Neo4jConnection() as db:
        node_count = db.read(f"MATCH (n:{namespace} {{dataset_id: $dataset_id}}) RETURN count(n) AS count",
                             dataset_id=dataset_id)[0]["count"]
        edge_count = db.read(f"MATCH (s:{namespace} {{dataset_id: $dataset_id}})-[r]->"
                             f"(t:{namespace} {{dataset_id: $dataset_id}}) "
                             "WHERE r.dataset_id = $dataset_id RETURN count(r) AS count",
                             dataset_id=dataset_id)[0]["count"]
        if node_count != manifest["counts"]["output_graph_nodes"] or edge_count != manifest["counts"]["output_graph_edges"]:
            raise ValueError("Neo4j node or edge count differs from the export")
        for batch in batches(output / "graph_nodes.csv", batch_size):
            ids = [scoped_id(dataset_id, row["id"]) for row in batch]
            actual = db.read(f"UNWIND $ids AS id MATCH (n:{namespace} {{dataset_id: $dataset_id, id: id}}) "
                             "RETURN n.id AS id, n.entity_id AS entity_id, labels(n) AS labels, n.name AS name, "
                             "n.source_id AS source_id, n.external_iri AS external_iri, "
                             "n.managed_by AS managed_by, n.manifest_sha256 AS manifest_sha256",
                             ids=ids, dataset_id=dataset_id)
            by_id = {row["id"]: row for row in actual}
            if len(by_id) != len(batch):
                raise ValueError("Neo4j node batch does not match export")
            for expected in batch:
                row = by_id[scoped_id(dataset_id, expected["id"])]
                if set(row["labels"]) != {namespace, expected["label"]} or any(
                    row[key] != expected[key] for key in ("name", "source_id", "external_iri")
                ) or row["entity_id"] != expected["id"] or row["managed_by"] != manager or row["manifest_sha256"] != manifest_hash:
                    raise ValueError(f"Node differs: {expected['id']}")
                node_labels[expected["label"]] += 1
        if "recipe_nutrients" in manifest["output_sha256"]:
            numeric_fields = ("serving_size_g", "calories", "total_fat_g", "saturated_fat_g",
                              "cholesterol_mg", "sodium_mg", "total_carbohydrate_g",
                              "dietary_fiber_g", "sugars_g", "protein_g")
            properties = ", ".join(f"n.{field} AS {field}" for field in numeric_fields)
            for batch in batches(output / "recipe_nutrients.csv", batch_size):
                actual = db.read(
                    f"UNWIND $ids AS id MATCH (n:Recipe:{namespace} {{dataset_id: $dataset_id, id: id}}) "
                    f"RETURN n.id AS recipe_id, {properties}, n.nutrition_parse_status AS parse_status, "
                    "n.nutrition_missing_fields AS missing_fields, n.nutrition_source_file AS source_file",
                    ids=[scoped_id(dataset_id, row["recipe_id"]) for row in batch], dataset_id=dataset_id)
                by_id = {row["recipe_id"]: row for row in actual}
                if len(by_id) != len(batch):
                    raise ValueError("Neo4j nutrient batch does not match export")
                for expected in batch:
                    row = by_id[scoped_id(dataset_id, expected["recipe_id"])]
                    if (row["parse_status"] != expected["parse_status"] or
                        row["missing_fields"] != expected["missing_fields"] or
                        row["source_file"] != expected["source_file"] or
                        any(row[field] != (float(expected[field]) if expected[field] else None)
                            for field in numeric_fields)):
                        raise ValueError(f"Nutrition differs: {expected['recipe_id']}")
        for batch in batches(output / "graph_edges.csv", batch_size):
            groups = {}
            for row in batch:
                if row["relation"] not in RELATIONS:
                    raise ValueError("Unexpected relationship type")
                groups.setdefault(row["relation"], []).append(row)
            for relation, rows in groups.items():
                lookup_rows = [{"id": scoped_id(dataset_id, row["id"]),
                                "source_id": scoped_id(dataset_id, row["source_id"]),
                                "target_id": scoped_id(dataset_id, row["target_id"])} for row in rows]
                actual = db.read(
                    f"UNWIND $rows AS row MATCH (s:{namespace} {{dataset_id: $dataset_id, id: row.source_id}})"
                    f"-[r:{relation} {{id: row.id}}]->(t:{namespace} {{dataset_id: $dataset_id, id: row.target_id}}) "
                    "RETURN r.id AS id, r.entity_id AS entity_id, r.review_id AS review_id, r.rating AS rating, "
                    "r.submit_date AS submit_date, r.source_file AS source_file, "
                    "r.evidence_state AS evidence_state, r.dataset_id AS dataset_id, "
                    "r.managed_by AS managed_by, r.manifest_sha256 AS manifest_sha256",
                    rows=lookup_rows, dataset_id=dataset_id)
                by_id = {row["id"]: row for row in actual}
                if len(by_id) != len(rows):
                    raise ValueError("Neo4j relationship batch does not match export")
                for expected in rows:
                    row = by_id[scoped_id(dataset_id, expected["id"])]
                    rating = int(expected["rating"]) if expected["rating"] else None
                    date = train_dates[expected["review_id"]] if relation == "RATED" else ""
                    if (row["entity_id"] != expected["id"] or row["review_id"] != expected["review_id"] or row["rating"] != rating or
                        row["submit_date"] != date or row["source_file"] != expected["source_file"] or
                        row["evidence_state"] != expected["evidence_state"] or
                        row["dataset_id"] != dataset_id or row["managed_by"] != manager or
                        row["manifest_sha256"] != manifest_hash):
                        raise ValueError(f"Relationship differs: {expected['id']}")
                    edge_relations[relation] += 1
        # The graph's rated edges are exactly the local train IDs, so no held-out ID can be present.
        if edge_relations["RATED"] != len(train_dates) or any(review in train_dates for review in held_out_ids):
            raise ValueError("Held-out interaction leakage")
        rating_rows = db.read(
            f"MATCH (u:ObservedUser:{namespace} {{dataset_id: $dataset_id}})-"
            f"[r:RATED {{dataset_id: $dataset_id}}]->(v:Recipe:{namespace} {{dataset_id: $dataset_id}}) "
            "RETURN r.rating AS rating, count(r) AS count ORDER BY rating",
            dataset_id=dataset_id)
        graph_rating_counts = Counter({row["rating"]: row["count"] for row in rating_rows})
        if graph_rating_counts != train_rating_counts or set(graph_rating_counts) - {1, 2, 3, 4, 5}:
            raise ValueError("Neo4j ratings by value differ from training reviews")
        sample = db.read(
            f"MATCH (u:ObservedUser:{namespace} {{dataset_id: $dataset_id}})-[:RATED]->"
            f"(r:Recipe:{namespace})-[:CONTAINS_MAPPED_INGREDIENT]->"
            f"(i:Ingredient:{namespace})-[:MAPS_TO_FOODON]->(f:FoodOnConcept:{namespace}) "
            "RETURN u.entity_id AS user, r.entity_id AS recipe, i.external_iri AS ingredient, "
            "f.external_iri AS foodon LIMIT 1", dataset_id=dataset_id)
        if not sample:
            raise ValueError("No user-to-FoodOn path in the imported cohort")
    return {"dataset_id": dataset_id, "manifest_sha256": manifest_hash,
            "node_count": node_count, "nodes_by_label": dict(sorted(node_labels.items())),
            "edge_count": edge_count, "edges_by_relation": dict(sorted(edge_relations.items())),
            "training_ratings_by_value": {str(value): train_rating_counts[value] for value in range(1, 6)},
            "neo4j_ratings_by_value": {str(value): graph_rating_counts[value] for value in range(1, 6)},
            "held_out_rating_edges": 0, "local_validation": local, "sample_path": sample[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    result = verify(args.output, args.batch_size)
    (args.output / "neo4j_reconciliation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
