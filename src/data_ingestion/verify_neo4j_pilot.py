# Verify every node and edge in Neo4j against the CSV export

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.data_ingestion.neo4j_pilot import DEFAULT_OUTPUT
from src.data_ingestion.validate import validate
from src.kg.connection import Neo4jConnection


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(output: Path = DEFAULT_OUTPUT) -> dict:
    output = output.resolve()
    check(output == DEFAULT_OUTPUT.resolve(), "Unexpected pilot directory")
    validate()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    dataset_id = manifest["dataset_id"]
    expected_nodes = {row["id"]: row for row in csv_rows(output / "graph_nodes.csv")}
    expected_edges = {row["id"]: row for row in csv_rows(output / "graph_edges.csv")}
    reviews = csv_rows(output / "reviews.csv")
    train_dates = {row["review_id"]: row["submit_date"] for row in reviews if row["split"] == "train"}
    held_out_ids = {row["review_id"] for row in reviews if row["split"] in {"validation", "test"}}
    check(len(expected_nodes) == manifest["counts"]["output_graph_nodes"], "Node CSV IDs are not unique")
    check(len(expected_edges) == manifest["counts"]["output_graph_edges"], "Edge CSV IDs are not unique")
    with Neo4jConnection() as db:
        nodes = db.read(
            "MATCH (n:HummusPilot {dataset_id: $dataset_id}) "
            "RETURN n.id AS id, labels(n) AS labels, n.name AS name, "
            "n.source_id AS source_id, n.external_iri AS external_iri, "
            "n.managed_by AS managed_by, n.manifest_sha256 AS manifest_sha256",
            dataset_id=dataset_id,
        )
        edges = db.read(
            "MATCH (s:HummusPilot {dataset_id: $dataset_id})-[r]->"
            "(t:HummusPilot {dataset_id: $dataset_id}) "
            "RETURN r.id AS id, s.id AS source_id, type(r) AS relation, "
            "t.id AS target_id, r.review_id AS review_id, r.rating AS rating, "
            "r.submit_date AS submit_date, r.source_file AS source_file, "
            "r.evidence_state AS evidence_state, r.managed_by AS managed_by, "
            "r.manifest_sha256 AS manifest_sha256, r.dataset_id AS dataset_id",
            dataset_id=dataset_id,
        )
        touching_edges = db.read(
            "MATCH (s)-[r]->(t) "
            "WHERE (s:HummusPilot AND s.dataset_id = $dataset_id) "
            "OR (t:HummusPilot AND t.dataset_id = $dataset_id) "
            "RETURN count(r) AS count",
            dataset_id=dataset_id,
        )[0]["count"]
        path = db.read(
            "MATCH (u:ObservedUser:HummusPilot {dataset_id: $dataset_id})-[:RATED]->"
            "(r:Recipe:HummusPilot)-[:CONTAINS_MAPPED_INGREDIENT]->"
            "(i:Ingredient:HummusPilot)-[:MAPS_TO_FOODON]->(f:FoodOnConcept:HummusPilot) "
            "RETURN u.id AS user, r.id AS recipe, i.external_iri AS ingredient, "
            "f.external_iri AS foodon ORDER BY user, recipe, ingredient, foodon LIMIT 1",
            dataset_id=dataset_id,
        )
    actual_nodes = {row["id"]: row for row in nodes}
    actual_edges = {row["id"]: row for row in edges}
    check(len(actual_nodes) == len(nodes) and set(actual_nodes) == set(expected_nodes), "Node IDs differ from export")
    check(len(actual_edges) == len(edges) and set(actual_edges) == set(expected_edges), "Edge IDs differ from export")
    check(touching_edges == len(edges), "Unexpected relationship touches a pilot node")
    leaked = {row["review_id"] for row in edges if row["relation"] == "RATED"} & held_out_ids
    check(not leaked, "Held-out rating entered Neo4j")
    for node_id, expected in expected_nodes.items():
        actual = actual_nodes[node_id]
        check(set(actual["labels"]) == {"HummusPilot", expected["label"]}, f"Wrong labels: {node_id}")
        for key in ("name", "source_id", "external_iri"):
            check(actual[key] == expected[key], f"Wrong {key}: {node_id}")
        check(actual["managed_by"] == "hummus-pilot-ingestion" and
              actual["manifest_sha256"] == manifest_hash, f"Wrong provenance: {node_id}")
    for edge_id, expected in expected_edges.items():
        actual = actual_edges[edge_id]
        for key in ("source_id", "relation", "target_id", "review_id", "source_file", "evidence_state"):
            check(actual[key] == expected[key], f"Wrong {key}: {edge_id}")
        rating = int(expected["rating"]) if expected["rating"] else None
        date = train_dates[expected["review_id"]] if expected["relation"] == "RATED" else ""
        check(actual["rating"] == rating and actual["submit_date"] == date,
              f"Wrong rating or date: {edge_id}")
        check(actual["dataset_id"] == dataset_id and
              actual["managed_by"] == "hummus-pilot-ingestion" and
              actual["manifest_sha256"] == manifest_hash, f"Wrong provenance: {edge_id}")
    check(bool(path), "No observed-user to FoodOn path found")
    node_labels = Counter(row["label"] for row in expected_nodes.values())
    edge_relations = Counter(row["relation"] for row in expected_edges.values())
    report = {
        "dataset_id": dataset_id,
        "manifest_sha256": manifest_hash,
        "node_count": len(nodes),
        "nodes_by_label": dict(sorted(node_labels.items())),
        "edge_count": len(edges),
        "edges_by_relation": dict(sorted(edge_relations.items())),
        "held_out_rating_edges": len(leaked),
        "sample_path": path[0],
    }
    return report


if __name__ == "__main__":
    result = verify()
    report_path = DEFAULT_OUTPUT / "neo4j_reconciliation.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
