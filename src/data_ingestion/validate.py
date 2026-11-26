# Validate the normalized HUMMUS pilot and the (training only) graph export

import csv
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/processed/hummus_pilot_v1"


def rows(name, output=OUTPUT):
    with (output / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(output=OUTPUT):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == output.name, "Dataset ID differs from output directory"
    for table, expected in manifest["output_sha256"].items():
        assert sha(output / f"{table}.csv") == expected, f"Changed output: {table}"
    users, recipes, reviews = rows("users", output), rows("recipes", output), rows("reviews", output)
    raw_lines, ingredients = rows("raw_ingredient_lines", output), rows("recipe_ingredients", output)
    foodon_links = rows("foodon_links", output)
    crosswalk, graph_nodes, graph_edges = rows("entity_crosswalk", output), rows("graph_nodes", output), rows("graph_edges", output)
    user_ids = {row["user_id"] for row in users}
    recipe_ids = {row["recipe_id"] for row in recipes}
    node_ids = {row["id"] for row in graph_nodes}
    assert len(users) == manifest["parameters"]["users_limit"]
    assert len(users) == manifest["counts"]["output_users"]
    assert len(recipes) == manifest["counts"]["output_recipes"]
    assert len(reviews) == manifest["counts"]["output_reviews"]
    assert len(graph_nodes) == manifest["counts"]["output_graph_nodes"]
    assert len(graph_edges) == manifest["counts"]["output_graph_edges"]
    assert len(user_ids) == len(users) and len(recipe_ids) == len(recipes)
    assert len(node_ids) == len(graph_nodes)
    assert len({row["review_id"] for row in reviews}) == len(reviews)
    assert len({row["id"] for row in graph_edges}) == len(graph_edges)
    assert all(row["user_id"] in user_ids and row["recipe_id"] in recipe_ids for row in reviews)
    assert all(row["recipe_id"] in recipe_ids and row["ingredient_id"] in node_ids for row in ingredients)
    assert all(row["recipe_id"] in recipe_ids for row in raw_lines)
    assert all(row["source_id"] in node_ids and row["target_id"] in node_ids for row in graph_edges)
    assert all(row["entity_type"] in {"Recipe", "Ingredient"} and row["foodkg_iri"].startswith("http") for row in crosswalk)
    assert len({(row["entity_type"], row["hummus_id"]) for row in crosswalk}) == len(crosswalk)
    assert all(row["label"] != "ScenarioProfile" for row in graph_nodes)
    assert all(row["relation"] in {"RATED", "CONTAINS_MAPPED_INGREDIENT", "MAPS_TO_FOODON", "SUBCLASS_OF"} for row in graph_edges)
    assert all(row["foodon_iri"].startswith("http://purl.obolibrary.org/obo/FOODON_") for row in foodon_links)
    if "foodon_hierarchy" in manifest["output_sha256"]:
        hierarchy = rows("foodon_hierarchy", output)
        assert len(hierarchy) == manifest["counts"]["output_foodon_hierarchy"]
        assert all(row["child_iri"].startswith("http://purl.obolibrary.org/obo/FOODON_") and
                   row["parent_iri"].startswith("http://purl.obolibrary.org/obo/FOODON_") and
                   row["evidence_state"] == "SOURCE_ASSERTED" for row in hierarchy)
        assert sum(row["relation"] == "SUBCLASS_OF" for row in graph_edges) == len(hierarchy)
    nutrition_count = 0
    if "recipe_nutrients" in manifest["output_sha256"]:
        nutrient_ids = set()
        numeric_fields = ("serving_size_g", "calories", "total_fat_g", "saturated_fat_g",
                          "cholesterol_mg", "sodium_mg", "total_carbohydrate_g",
                          "dietary_fiber_g", "sugars_g", "protein_g")
        with (output / "recipe_nutrients.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                assert row["recipe_id"] in recipe_ids and row["recipe_id"] not in nutrient_ids
                nutrient_ids.add(row["recipe_id"])
                assert row["parse_status"] in {"COMPLETE", "PARTIAL", "UNPARSEABLE"}
                assert all(not row[field] or float(row[field]) >= 0 for field in numeric_fields)
                nutrition_count += 1
        assert nutrient_ids == recipe_ids and nutrition_count == manifest["counts"]["output_recipe_nutrients"]
    by_user = defaultdict(lambda: defaultdict(set))
    dates_by_user = defaultdict(lambda: defaultdict(list))
    for row in reviews:
        assert row["rating"] in {"0", "1", "2", "3", "4", "5"}
        if row["split"] == "unrated":
            assert row["rating"] == "0" and row["is_rated"] == "False"
        else:
            assert row["rating"] != "0" and row["is_rated"] == "True"
            by_user[row["user_id"]][row["split"]].add(row["recipe_id"])
            dates_by_user[row["user_id"]][row["split"]].append(row["submit_date"])
    for user, split_pairs in by_user.items():
        assert all(split_pairs[split] for split in ("train", "validation", "test")), user
        assert not (split_pairs["train"] & split_pairs["validation"] | split_pairs["train"] & split_pairs["test"] | split_pairs["validation"] & split_pairs["test"]), user
        if manifest["parameters"].get("split_strategy") == "chronological":
            dated = dates_by_user[user]
            assert all(dated.values()) and all(all(value for value in dates) for dates in dated.values()), user
            assert max(dated["train"]) <= min(dated["validation"]) <= max(dated["validation"]) <= min(dated["test"]), user
            assert len(dated["train"]) >= 2, user
    train_rows = {row["review_id"]: row for row in reviews if row["split"] == "train"}
    train_ids = set(train_rows)
    held_out_ids = {row["review_id"] for row in reviews if row["split"] in {"validation", "test"}}
    rated_edges = [row for row in graph_edges if row["relation"] == "RATED"]
    graph_ratings = {row["review_id"] for row in rated_edges}
    assert len(rated_edges) == len(graph_ratings) == len(train_rows)
    assert graph_ratings == train_ids and not (graph_ratings & held_out_ids)
    for edge in rated_edges:
        review = train_rows[edge["review_id"]]
        assert (edge["id"] == edge["review_id"] and
                edge["source_id"] == review["user_id"] and
                edge["target_id"] == review["recipe_id"] and
                edge["rating"] == review["rating"] and
                edge["rating"] in {"1", "2", "3", "4", "5"} and
                edge["evidence_state"] == "CONFIRMED"), edge["review_id"]
    rating_counts = Counter(row["rating"] for row in rated_edges)
    assert rating_counts == Counter(row["rating"] for row in train_rows.values())
    assert all(row["evidence_state"] == "INFERRED" for row in graph_edges if row["relation"] == "CONTAINS_MAPPED_INGREDIENT")
    assert all(row["evidence_state"] == "INFERRED" for row in graph_edges if row["relation"] == "MAPS_TO_FOODON")
    counts = Counter(row["split"] for row in reviews)
    assert len(users) > 0 and len(recipes) > 0 and len(ingredients) > 0 and len(raw_lines) > 0
    return {"users": len(users), "recipes": len(recipes), "mapped_ingredient_edges": len(ingredients),
            "foodon_links": len(foodon_links),
            "recipe_nutrient_rows": nutrition_count,
            "raw_ingredient_fragments": len(raw_lines), "reviews_by_split": dict(counts),
            "graph_nodes": len(graph_nodes), "graph_edges": len(graph_edges),
            "training_ratings_by_value": {str(value): rating_counts[str(value)] for value in range(1, 6)},
            "held_out_rating_edges_in_graph": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
