# Ingest the HUMMUS source data files into a mid-scale pilot graph

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile


SOURCES = {
    "members": ("hummus_data/raw/members.zip", "members.csv"),
    "recipes": ("hummus_data/raw/recipes.zip", "recipes.csv"),
    "reviews": ("hummus_data/raw/reviews.zip", "reviews.csv"),
    "mapping": ("food_kg/graph_ingredient_mapping.zip", "graph_ingredient_mapping.csv"),
    "labels": ("food_kg/graph_ingredient_labels.zip", "graph_ingredient_labels.csv"),
    "core": ("food_kg/FoodKG_simplified_extended.zip", "FoodKG_simplified.csv"),
}
FIELDS = {
    "users": ["user_id", "source_member_id", "source_file"],
    "recipes": ["recipe_id", "source_recipe_id", "title", "foodkg_recipe_iri", "recipe_url", "ingredient_source_present", "raw_ingredients", "nutrition_json", "source_file"],
    "raw_ingredient_lines": ["recipe_id", "raw_fragment", "parse_status", "source_file"],
    "reviews": ["review_id", "user_id", "recipe_id", "rating", "is_rated", "submit_date", "split", "source_file"],
    "recipe_ingredients": ["recipe_id", "ingredient_id", "ingredient_name", "foodkg_ingredient_iri", "source_ingredient_iri", "evidence_state", "source_file"],
    "ingredient_labels": ["ingredient_id", "foodkg_ingredient_iri", "ingredient_name", "label", "source_file", "evidence_state"],
    "foodon_links": ["ingredient_id", "foodkg_ingredient_iri", "foodon_iri", "source_file", "evidence_state"],
    "entity_crosswalk": ["entity_type", "hummus_id", "foodkg_iri", "mapping_method", "evidence_state", "source_file"],
    "enrichment_edges": ["source_id", "relation", "target_id", "source_file", "evidence_state"],
    "rejected_records": ["source_file", "source_key", "stage", "reason"],
    "graph_nodes": ["id", "label", "name", "source_id", "external_iri"],
    "graph_edges": ["id", "source_id", "relation", "target_id", "review_id", "rating", "source_file", "evidence_state"],
}


def csv_rows(root: Path, key: str):
    relative, member = SOURCES[key]
    archive = root / relative
    with ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if Path(name).name == member]
        if len(names) != 1:
            raise ValueError(f"Expected one {member} in {archive}, found {names}")
        with zf.open(names[0]) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8-sig", errors="strict", newline="")
            reader = csv.DictReader(text)
            if not reader.fieldnames:
                raise ValueError(f"Missing CSV header: {archive}")
            for number, row in enumerate(reader, start=2):
                yield number, row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_inventory(path: Path, member: str) -> dict:
    with ZipFile(path) as zf:
        candidates = [info for info in zf.infolist() if Path(info.filename).name == member]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {member} in {path}")
        info = candidates[0]
        with zf.open(info) as stream:
            reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
            header = next(reader)
    return {"member": info.filename, "uncompressed_bytes": info.file_size, "columns": header}


def stable_id(prefix: str, value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid source ID: {value!r}")
    return f"{prefix}_{value}"


def iri_id(prefix: str, iri: str) -> str:
    return f"{prefix}_{hashlib.sha256(iri.encode('utf-8')).hexdigest()[:20]}"


def valid_iri(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not any(c.isspace() for c in value)


def rating_value(raw: str) -> int | None:
    try:
        number = float(raw)
    except (ValueError, TypeError):
        return None
    return int(number) if number.is_integer() and 0 <= number <= 5 else None


def event_date(raw: str) -> str:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_users(root: Path, limit: int, min_ratings: int) -> tuple[list[str], Counter]:
    counts: Counter = Counter()
    audit: Counter = Counter()
    for _, row in csv_rows(root, "reviews"):
        audit["raw_review_rows"] += 1
        user, recipe, review = (row.get(key, "").strip() for key in ("member_id", "recipe_id", "review_id"))
        rating = rating_value(row.get("rating", ""))
        if not user or not recipe or not review or rating is None:
            audit["invalid_review_identity_or_rating"] += 1
        elif rating >= 1:
            counts[user] += 1
        elif rating == 0:
            audit["unrated_comments"] += 1
    selected = sorted((user for user, count in counts.items() if count >= min_ratings), key=lambda user: (-counts[user], int(user) if user.isdigit() else user))[:limit]
    audit["eligible_users"] = sum(count >= min_ratings for count in counts.values())
    return selected, audit


def collect_reviews(root: Path, selected: set[str], max_events: int, rejected: list[dict], audit: Counter,
                    split_strategy: str = "pilot") -> tuple[list[dict], set[str]]:
    events: dict[str, dict[str, dict]] = defaultdict(dict)
    unrated_examples = {}
    for line, row in csv_rows(root, "reviews"):
        user = row.get("member_id", "").strip()
        if user not in selected:
            continue
        recipe = row.get("recipe_id", "").strip()
        review = row.get("review_id", "").strip()
        rating = rating_value(row.get("rating", ""))
        if not user or not recipe or not review or rating is None:
            rejected.append({"source_file": "reviews.csv", "source_key": str(line), "stage": "parse", "reason": "invalid identity or rating"})
            continue
        date = event_date(row.get("submit_date", ""))
        if rating == 0:
            unrated_examples.setdefault(user, (user, recipe, review, date))
            continue
        if split_strategy == "chronological" and not date:
            rejected.append({"source_file": "reviews.csv", "source_key": review, "stage": "date", "reason": "invalid rated event date"})
            audit["rated_events_invalid_date"] += 1
            continue
        # One event per user-recipe pair prevents a target pair crossing splits.
        current = events[user].get(recipe)
        if current is not None:
            audit["duplicate_user_recipe_ratings_collapsed"] += 1
        candidate = {"review_id": stable_id("hummus_review", review), "user_id": stable_id("hummus_user", user),
                     "recipe_id": stable_id("hummus_recipe", recipe), "rating": rating, "is_rated": True,
                     "submit_date": date, "split": "", "source_file": "reviews.csv"}
        if current is None or (candidate["submit_date"], candidate["review_id"]) > (current["submit_date"], current["review_id"]):
            events[user][recipe] = candidate
    output = []
    retained_users = set()
    for user in sorted(selected, key=lambda item: int(item) if item.isdigit() else item):
        candidates = sorted(events[user].values(), key=lambda item: (item["submit_date"], item["review_id"]))
        if len(candidates) < 3:
            audit["users_without_three_unique_positive_recipes"] += 1
            continue
        retained_users.add(user)
        retained = candidates[-max_events:] if max_events else candidates
        start = len(output)
        for event in retained:
            event["split"] = "train"
            output.append(event)
        user_events = output[start:]
        if split_strategy == "chronological":
            from math import ceil
            n_test = max(1, ceil(0.20 * len(user_events)))
            n_validation = max(1, ceil(0.10 * len(user_events)))
            if len(user_events) - n_test - n_validation < 2:
                del output[start:]
                retained_users.remove(user)
                audit["users_without_two_training_pairs"] += 1
                continue
            for event in user_events[-(n_test + n_validation):-n_test]:
                event["split"] = "validation"
            for event in user_events[-n_test:]:
                event["split"] = "test"
        else:
            user_events[-2]["split"] = "validation"
            user_events[-1]["split"] = "test"
    eligible_unrated = [unrated_examples[user] for user in sorted(retained_users) if user in unrated_examples]
    if eligible_unrated:
        user, recipe, review, date = eligible_unrated[0]
        output.append({"review_id": stable_id("hummus_review", review), "user_id": stable_id("hummus_user", user),
                       "recipe_id": stable_id("hummus_recipe", recipe), "rating": 0, "is_rated": False,
                       "submit_date": date, "split": "unrated", "source_file": "reviews.csv"})
    output.sort(key=lambda item: item["review_id"])
    return output, retained_users


def collect_recipes(root: Path, wanted: set[str], rejected: list[dict], audit: Counter) -> tuple[dict[str, dict], list[dict]]:
    found = {}
    raw_lines = []
    for line, row in csv_rows(root, "recipes"):
        audit["raw_recipe_rows"] += 1
        if not row.get("food_kg_locator", "").strip():
            audit["raw_recipes_missing_foodkg_locator"] += 1
        if not row.get("ingredients", "").strip() or row["ingredients"].strip() in {"{}", "set()"}:
            audit["raw_recipes_missing_ingredient_source"] += 1
        rid = row.get("recipe_id", "").strip()
        if rid not in wanted:
            continue
        if rid in found:
            rejected.append({"source_file": "recipes.csv", "source_key": rid, "stage": "identity", "reason": "duplicate recipe_id"})
            continue
        title = row.get("title", "").strip()
        if not title:
            rejected.append({"source_file": "recipes.csv", "source_key": rid, "stage": "parse", "reason": "blank title"})
            continue
        iri = row.get("food_kg_locator", "").strip()
        if not iri:
            audit["selected_recipes_missing_foodkg_locator"] += 1
        if iri and not valid_iri(iri):
            rejected.append({"source_file": "recipes.csv", "source_key": rid, "stage": "mapping", "reason": "invalid FoodKG locator"})
            iri = ""
        nutrition_raw = row.get("nutrition", "")
        nutrition = ""
        if nutrition_raw:
            try:
                parsed = json.loads(nutrition_raw)
                if isinstance(parsed, dict):
                    nutrition = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
                else:
                    audit["nutrition_not_object"] += 1
            except json.JSONDecodeError:
                audit["nutrition_parse_errors"] += 1
        ingredient_raw = row.get("ingredients", "").strip()
        if not ingredient_raw or ingredient_raw in {"{}", "set()"}:
            audit["selected_recipes_missing_ingredient_source"] += 1
        if ingredient_raw:
            try:
                fragments = ast.literal_eval(ingredient_raw)
                if not isinstance(fragments, (set, list, tuple)) or not all(isinstance(value, str) for value in fragments):
                    raise ValueError("ingredient collection is not strings")
                raw_lines.extend({"recipe_id": stable_id("hummus_recipe", rid), "raw_fragment": fragment,
                                  "parse_status": "SOURCE_FRAGMENT", "source_file": "recipes.csv"} for fragment in sorted(fragments))
            except (SyntaxError, ValueError):
                audit["raw_ingredient_parse_errors"] += 1
                raw_lines.append({"recipe_id": stable_id("hummus_recipe", rid), "raw_fragment": ingredient_raw,
                                  "parse_status": "UNPARSED", "source_file": "recipes.csv"})
        found[rid] = {"recipe_id": stable_id("hummus_recipe", rid), "source_recipe_id": rid, "title": title,
                      "foodkg_recipe_iri": iri, "recipe_url": row.get("recipe_url", "").strip(),
                      "ingredient_source_present": bool(ingredient_raw and ingredient_raw not in {"{}", "set()"}), "raw_ingredients": ingredient_raw,
                      "nutrition_json": nutrition, "source_file": "recipes.csv"}
    return found, raw_lines


def collect_users(root: Path, wanted: set[str], rejected: list[dict], audit: Counter) -> set[str]:
    found = set()
    for _, row in csv_rows(root, "members"):
        audit["raw_member_rows"] += 1
        uid = row.get("member_id", "").strip()
        if uid in wanted:
            if uid in found:
                rejected.append({"source_file": "members.csv", "source_key": uid, "stage": "identity", "reason": "duplicate member_id"})
            found.add(uid)
    return found


def collect_foodkg(root: Path, recipes: dict[str, dict], audit: Counter, rejected: list[dict]):
    iri_to_recipes = defaultdict(set)
    for row in recipes.values():
        iri = row["foodkg_recipe_iri"]
        if iri:
            iri_to_recipes[iri].add(row["recipe_id"])
    ambiguous = {iri: ids for iri, ids in iri_to_recipes.items() if len(ids) > 1}
    audit["ambiguous_foodkg_recipe_locators"] = len(ambiguous)
    audit["recipes_with_ambiguous_foodkg_locator"] = sum(len(ids) for ids in ambiguous.values())
    for iri, ids in sorted(ambiguous.items()):
        for recipe_id in sorted(ids):
            rejected.append({"source_file": "recipes.csv", "source_key": recipe_id,
                             "stage": "mapping", "reason": "FoodKG locator shared by multiple HUMMUS recipe IDs: " + iri})
    iri_to_recipe = {iri: next(iter(ids)) for iri, ids in iri_to_recipes.items() if len(ids) == 1}
    mapping = {}
    for line, row in csv_rows(root, "mapping"):
        recipe_iri = row.get("recipe", "").strip()
        if recipe_iri not in iri_to_recipe:
            continue
        ingredient_iri = row.get("foodKG_mapped_ingredient", "").strip()
        source_iri = row.get("foodKG_ingredient", "").strip()
        name = row.get("ing_name", "").strip()
        if not name or not valid_iri(ingredient_iri):
            rejected.append({"source_file": "graph_ingredient_mapping.csv", "source_key": str(line), "stage": "mapping", "reason": "missing name or mapped IRI"})
            continue
        key = (iri_to_recipe[recipe_iri], ingredient_iri)
        mapping.setdefault(key, {"recipe_id": key[0], "ingredient_id": iri_id("foodkg_ingredient", ingredient_iri),
                                 "ingredient_name": name, "foodkg_ingredient_iri": ingredient_iri,
                                 "source_ingredient_iri": source_iri if valid_iri(source_iri) else "",
                                 "evidence_state": "INFERRED", "source_file": "graph_ingredient_mapping.csv"})
    relevant = {row["foodkg_ingredient_iri"] for row in mapping.values()}
    labels = defaultdict(set)
    canonical_names = {}
    for _, row in csv_rows(root, "labels"):
        iri = row.get("ingredient_id", "").strip()
        if iri in relevant and row.get("label", "").strip():
            labels[iri].add(row["label"].strip())
            canonical_names.setdefault(iri, row.get("ingredient_name", "").strip())
    audit["mapped_ingredient_relations"] = len(mapping)
    audit["mapped_ingredient_iris_with_label"] = sum(bool(labels[iri]) for iri in relevant)
    return mapping, labels, canonical_names


def collect_core_links(root: Path, recipes: dict[str, dict], mapping: dict, audit: Counter) -> list[dict]:
    recipe_iris = {row["foodkg_recipe_iri"] for row in recipes.values() if row["foodkg_recipe_iri"]}
    ingredient_iris = {row["foodkg_ingredient_iri"] for row in mapping.values()}
    verified_recipes = set()
    verified_ingredients = set()
    links = set()
    equivalent = "http://www.w3.org/2002/07/owl#equivalentClass"
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
    for _, row in csv_rows(root, "core"):
        audit["foodkg_core_triples_scanned"] += 1
        subject, predicate, target = (row.get(key, "").strip() for key in ("subject", "predicate", "object"))
        if subject in recipe_iris and predicate in {rdf_type, rdfs_label}:
            verified_recipes.add(subject)
        if subject in ingredient_iris:
            verified_ingredients.add(subject)
            if predicate == equivalent and target.startswith("http://purl.obolibrary.org/obo/FOODON_"):
                links.add((subject, target))
    audit["foodkg_core_recipe_iris_verified"] = len(verified_recipes)
    audit["foodkg_core_ingredient_iris_verified"] = len(verified_ingredients)
    audit["foodkg_core_foodon_links"] = len(links)
    return [{"ingredient_id": iri_id("foodkg_ingredient", ingredient), "foodkg_ingredient_iri": ingredient,
             "foodon_iri": foodon, "source_file": "FoodKG_simplified.csv", "evidence_state": "INFERRED"}
            for ingredient, foodon in sorted(links)]


def ingest(root: Path, output: Path, users_limit: int = 40, min_ratings: int = 8, max_events: int = 8,
           split_strategy: str = "pilot", selection_buffer: int = 0) -> dict:
    if users_limit < 1 or min_ratings < 3 or max_events < 0 or selection_buffer < 0 or (split_strategy == "pilot" and max_events < 3):
        raise ValueError("Need positive user limits and enough distinct events per user.")
    if split_strategy not in {"pilot", "chronological"}:
        raise ValueError("split_strategy must be pilot or chronological")
    root, output = root.resolve(), output.resolve()
    if not root.is_dir() or output == root or root in output.parents:
        raise ValueError("Output must be outside the immutable raw source directory.")
    output.mkdir(parents=True, exist_ok=True)
    audit = Counter()
    rejected = []
    selected, counts = select_users(root, users_limit + selection_buffer, min_ratings)
    audit.update(counts)
    reviews, retained = collect_reviews(root, set(selected), max_events, rejected, audit, split_strategy)
    wanted_recipes = {row["recipe_id"].removeprefix("hummus_recipe_") for row in reviews}
    recipes, raw_lines = collect_recipes(root, wanted_recipes, rejected, audit)
    found_users = collect_users(root, retained, rejected, audit)
    valid_reviews = []
    for row in reviews:
        uid = row["user_id"].removeprefix("hummus_user_")
        rid = row["recipe_id"].removeprefix("hummus_recipe_")
        if uid not in found_users or rid not in recipes:
            rejected.append({"source_file": "reviews.csv", "source_key": row["review_id"], "stage": "join", "reason": "missing member or recipe"})
        else:
            valid_reviews.append(row)
    # A dropped pair can invalidate a user's split; exclude that user entirely.
    split_counts = defaultdict(Counter)
    for row in valid_reviews:
        split_counts[row["user_id"]][row["split"]] += 1
    keep_users = {user for user, c in split_counts.items() if c["train"] and c["validation"] and c["test"]}
    ordered_keep = [stable_id("hummus_user", uid) for uid in selected
                    if stable_id("hummus_user", uid) in keep_users]
    if len(ordered_keep) < users_limit:
        raise ValueError(f"Only {len(ordered_keep)} users passed source joins and split; increase --selection-buffer")
    keep_users = set(ordered_keep[:users_limit])
    audit["selected_user_pool"] = len(selected)
    audit["selected_users_removed_after_join"] = len(selected) - len(ordered_keep)
    valid_reviews = [row for row in valid_reviews if row["user_id"] in keep_users]
    used_recipes = {row["recipe_id"].removeprefix("hummus_recipe_") for row in valid_reviews}
    recipes = {rid: row for rid, row in recipes.items() if rid in used_recipes}
    raw_lines = [row for row in raw_lines if row["recipe_id"].removeprefix("hummus_recipe_") in used_recipes]
    mapping, labels, canonical_names = collect_foodkg(root, recipes, audit, rejected)
    foodon_links = collect_core_links(root, recipes, mapping, audit)
    users = [{"user_id": stable_id("hummus_user", uid), "source_member_id": uid, "source_file": "members.csv"}
             for uid in sorted(found_users) if stable_id("hummus_user", uid) in keep_users]
    recipe_rows = sorted(recipes.values(), key=lambda row: row["recipe_id"])
    ingredients = sorted(mapping.values(), key=lambda row: (row["recipe_id"], row["ingredient_id"]))
    ingredient_labels = [{"ingredient_id": iri_id("foodkg_ingredient", iri), "foodkg_ingredient_iri": iri,
                          "ingredient_name": canonical_names.get(iri, ""), "label": label,
                          "source_file": "graph_ingredient_labels.csv", "evidence_state": "INFERRED"}
                         for iri, categories in sorted(labels.items()) for label in sorted(categories)]
    crosswalk = [{"entity_type": "Recipe", "hummus_id": row["recipe_id"], "foodkg_iri": row["foodkg_recipe_iri"],
                  "mapping_method": "HUMMUS food_kg_locator", "evidence_state": "INFERRED", "source_file": "recipes.csv"}
                 for row in recipe_rows if row["foodkg_recipe_iri"]]
    ingredient_crosswalk = {row["ingredient_id"]: row["foodkg_ingredient_iri"] for row in ingredients}
    crosswalk += [{"entity_type": "Ingredient", "hummus_id": ingredient_id, "foodkg_iri": ingredient_crosswalk[ingredient_id],
                   "mapping_method": "HUMMUS FoodKG mapping", "evidence_state": "INFERRED", "source_file": "graph_ingredient_mapping.csv"}
                  for ingredient_id in sorted(ingredient_crosswalk)]
    enrichment = [{"source_id": row["recipe_id"], "relation": "CONTAINS_MAPPED_INGREDIENT", "target_id": row["ingredient_id"],
                   "source_file": row["source_file"], "evidence_state": row["evidence_state"]} for row in ingredients]
    enrichment += [{"source_id": row["ingredient_id"], "relation": "MAPS_TO_FOODON", "target_id": iri_id("foodon", row["foodon_iri"]),
                    "source_file": row["source_file"], "evidence_state": row["evidence_state"]} for row in foodon_links]
    nodes = ([{"id": row["user_id"], "label": "ObservedUser", "name": row["source_member_id"], "source_id": row["source_member_id"], "external_iri": ""} for row in users]
             + [{"id": row["recipe_id"], "label": "Recipe", "name": row["title"], "source_id": row["source_recipe_id"], "external_iri": row["foodkg_recipe_iri"]} for row in recipe_rows]
             + [{"id": row["ingredient_id"], "label": "Ingredient", "name": canonical_names.get(row["foodkg_ingredient_iri"]) or row["ingredient_name"], "source_id": "", "external_iri": row["foodkg_ingredient_iri"]}
                for row in {r["ingredient_id"]: r for r in ingredients}.values()]
             + [{"id": iri_id("foodon", iri), "label": "FoodOnConcept", "name": iri.rsplit("/", 1)[-1], "source_id": "", "external_iri": iri}
                for iri in sorted({row["foodon_iri"] for row in foodon_links})])
    edges = [{"id": row["review_id"], "source_id": row["user_id"], "relation": "RATED", "target_id": row["recipe_id"],
              "review_id": row["review_id"], "rating": row["rating"], "source_file": "reviews.csv", "evidence_state": "CONFIRMED"}
             for row in valid_reviews if row["split"] == "train"]
    edges += [{"id": iri_id("contains", row["recipe_id"] + row["ingredient_id"]), "source_id": row["recipe_id"],
               "relation": "CONTAINS_MAPPED_INGREDIENT", "target_id": row["ingredient_id"], "review_id": "", "rating": "",
               "source_file": row["source_file"], "evidence_state": row["evidence_state"]} for row in ingredients]
    edges += [{"id": iri_id("foodonlink", row["ingredient_id"] + row["foodon_iri"]), "source_id": row["ingredient_id"],
               "relation": "MAPS_TO_FOODON", "target_id": iri_id("foodon", row["foodon_iri"]), "review_id": "", "rating": "",
               "source_file": row["source_file"], "evidence_state": row["evidence_state"]} for row in foodon_links]
    nodes.sort(key=lambda row: row["id"])
    edges.sort(key=lambda row: row["id"])
    tables = {"users": users, "recipes": recipe_rows, "reviews": valid_reviews, "recipe_ingredients": ingredients,
              "ingredient_labels": ingredient_labels, "foodon_links": foodon_links,
              "raw_ingredient_lines": raw_lines,
              "entity_crosswalk": crosswalk, "enrichment_edges": enrichment, "rejected_records": rejected,
              "graph_nodes": nodes, "graph_edges": edges}
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", FIELDS[name], rows)
        audit[f"output_{name}"] = len(rows)
    audit["foodkg_recipe_locator_coverage"] = sum(bool(row["foodkg_recipe_iri"]) for row in recipe_rows)
    audit["recipes_with_mapped_ingredients"] = len({row["recipe_id"] for row in ingredients})
    audit["train_reviews"] = sum(row["split"] == "train" for row in valid_reviews)
    audit["validation_reviews"] = sum(row["split"] == "validation" for row in valid_reviews)
    audit["test_reviews"] = sum(row["split"] == "test" for row in valid_reviews)
    for user in keep_users:
        pairs = [(row["user_id"], row["recipe_id"]) for row in valid_reviews if row["user_id"] == user and row["split"] in {"train", "validation", "test"}]
        if len(pairs) != len(set(pairs)):
            raise AssertionError("User-recipe pair crosses split boundary")
    held_out = {row["review_id"] for row in valid_reviews if row["split"] != "train"}
    if held_out & {row["review_id"] for row in edges if row["review_id"]}:
        raise AssertionError("Held-out review leaked into training graph")
    source_hashes = {}
    dataset_id = output.name
    if not re.fullmatch(r"hummus_[a-z0-9_]+", dataset_id):
        raise ValueError("Output directory name must be a stable hummus dataset ID")
    manifest = {"dataset_id": dataset_id, "source_root": str(root), "parameters": {"users_limit": users_limit,
                "min_ratings": min_ratings, "max_events": max_events, "split_strategy": split_strategy,
                "selection_buffer": selection_buffer,
                "selection": "most observed 1-5 raw ratings; tie by source member ID; latest distinct user-recipe review by date and review ID; chronological split when requested"},
                "sources": {key: {"path": relative, "sha256": source_hashes.setdefault(relative, sha256_file(root / relative)), "bytes": (root / relative).stat().st_size,
                                    "archive": archive_inventory(root / relative, member)}
                            for key, (relative, member) in SOURCES.items()},
                "counts": dict(sorted(audit.items())),
                "output_sha256": {name: sha256_file(output / f"{name}.csv") for name in tables}}
    readme = root / "README.md"
    if readme.is_file():
        manifest["source_readme"] = {"path": "README.md", "sha256": sha256_file(readme)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hummus_pilot_v1"))
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--min-ratings", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--split-strategy", choices=("pilot", "chronological"), default="pilot")
    parser.add_argument("--selection-buffer", type=int, default=0)
    args = parser.parse_args()
    result = ingest(args.raw_root, args.output, args.users, args.min_ratings, args.max_events,
                    args.split_strategy, args.selection_buffer)
    print(json.dumps(result["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
