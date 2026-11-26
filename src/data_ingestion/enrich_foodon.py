# Add FoodOn superclass closure for mapped ingredients

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from src.data_ingestion.hummus import FIELDS, csv_rows, iri_id, sha256_file, write_csv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/hummus_4000_v1"
FOODON_PREFIX = "http://purl.obolibrary.org/obo/FOODON_"
SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
HIERARCHY_FIELDS = ["child_iri", "parent_iri", "source_file", "evidence_state"]


def enrich(output: Path = DEFAULT_OUTPUT, root: Path = ROOT / "data/raw") -> dict:
    output, root = output.resolve(), root.resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_id"] != output.name or "foodon_hierarchy" in manifest["output_sha256"]:
        raise ValueError("Unexpected dataset ID or FoodOn hierarchy already installed")
    direct = set()
    with (output / "foodon_links.csv").open(encoding="utf-8", newline="") as stream:
        direct.update(row["foodon_iri"] for row in csv.DictReader(stream))
    parents = defaultdict(set)
    labels = {}
    counters = Counter()
    for _, row in csv_rows(root, "core"):
        child = row.get("subject", "").strip()
        if not child.startswith(FOODON_PREFIX):
            continue
        predicate, target = row.get("predicate", "").strip(), row.get("object", "").strip()
        if predicate == SUBCLASS:
            if target.startswith(FOODON_PREFIX):
                parents[child].add(target)
            else:
                counters["non_named_foodon_superclass_rows_skipped"] += 1
        elif predicate == LABEL and child not in labels:
            labels[child] = target
    reached = set(direct)
    queue = deque(sorted(direct))
    hierarchy = set()
    while queue:
        child = queue.popleft()
        for parent in sorted(parents.get(child, ())):
            if parent == child:
                counters["self_superclass_rows_skipped"] += 1
                continue
            hierarchy.add((child, parent))
            if parent not in reached:
                reached.add(parent)
                queue.append(parent)
    rows = [{"child_iri": child, "parent_iri": parent, "source_file": "FoodKG_simplified.csv",
             "evidence_state": "SOURCE_ASSERTED"} for child, parent in sorted(hierarchy)]
    write_csv(output / "foodon_hierarchy.csv", HIERARCHY_FIELDS, rows)

    # Rewrite node names from source labels where present and add only newly reached ancestors.
    nodes_path = output / "graph_nodes.csv"
    nodes_tmp = output / "graph_nodes.foodon.tmp"
    existing_iris = set()
    with nodes_path.open(encoding="utf-8", newline="") as source, nodes_tmp.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=FIELDS["graph_nodes"])
        writer.writeheader()
        for row in reader:
            if row["label"] == "FoodOnConcept":
                existing_iris.add(row["external_iri"])
                row["name"] = labels.get(row["external_iri"], row["name"])
            writer.writerow(row)
        for iri in sorted(reached - existing_iris):
            writer.writerow({"id": iri_id("foodon", iri), "label": "FoodOnConcept",
                             "name": labels.get(iri, iri.rsplit("/", 1)[-1]),
                             "source_id": "", "external_iri": iri})
    nodes_tmp.replace(nodes_path)

    edges_path = output / "graph_edges.csv"
    edges_tmp = output / "graph_edges.foodon.tmp"
    with edges_path.open(encoding="utf-8", newline="") as source, edges_tmp.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=FIELDS["graph_edges"])
        writer.writeheader()
        writer.writerows(reader)
        for child, parent in sorted(hierarchy):
            writer.writerow({"id": iri_id("subclass", child + parent),
                             "source_id": iri_id("foodon", child), "relation": "SUBCLASS_OF",
                             "target_id": iri_id("foodon", parent), "review_id": "", "rating": "",
                             "source_file": "FoodKG_simplified.csv", "evidence_state": "SOURCE_ASSERTED"})
    edges_tmp.replace(edges_path)

    enrichment_path = output / "enrichment_edges.csv"
    enrichment_tmp = output / "enrichment_edges.foodon.tmp"
    with enrichment_path.open(encoding="utf-8", newline="") as source, enrichment_tmp.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=FIELDS["enrichment_edges"])
        writer.writeheader()
        writer.writerows(reader)
        for child, parent in sorted(hierarchy):
            writer.writerow({"source_id": iri_id("foodon", child), "relation": "SUBCLASS_OF",
                             "target_id": iri_id("foodon", parent), "source_file": "FoodKG_simplified.csv",
                             "evidence_state": "SOURCE_ASSERTED"})
    enrichment_tmp.replace(enrichment_path)

    manifest["counts"]["output_foodon_hierarchy"] = len(rows)
    manifest["counts"]["output_graph_nodes"] += len(reached - existing_iris)
    manifest["counts"]["output_graph_edges"] += len(rows)
    manifest["counts"]["output_enrichment_edges"] += len(rows)
    manifest["counts"]["foodon_named_concepts_reached"] = len(reached)
    manifest["counts"].update(counters)
    for table in ("foodon_hierarchy", "graph_nodes", "graph_edges", "enrichment_edges"):
        manifest["output_sha256"][table] = sha256_file(output / f"{table}.csv")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"direct_concepts": len(direct), "named_concepts_reached": len(reached),
            "subclass_edges": len(rows), "ancestor_nodes_added": len(reached - existing_iris),
            "skipped": dict(counters)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    print(json.dumps(enrich(args.output, args.raw_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
