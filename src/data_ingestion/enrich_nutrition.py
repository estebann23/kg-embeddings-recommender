# Parse recipe nutrition values from source source into explicit attributes with types/units

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.data_ingestion.hummus import sha256_file, write_csv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/hummus_4000_v1"
SPECS = {
    "calories": ("calories", ""),
    "total_fat_g": ("totalFat", "g"),
    "saturated_fat_g": ("saturatedFat", "g"),
    "cholesterol_mg": ("cholesterol", "mg"),
    "sodium_mg": ("sodium", "mg"),
    "total_carbohydrate_g": ("totalCarbohydrate", "g"),
    "dietary_fiber_g": ("dietaryFiber", "g"),
    "sugars_g": ("sugars", "g"),
    "protein_g": ("protein", "g"),
}
FIELDS = ["recipe_id", "serving_size_g", *SPECS, "parse_status", "missing_fields", "source_file"]


def parse_number(value: object, unit: str) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*" + re.escape(unit), raw, re.IGNORECASE)
    return match.group(1) if match else ""


def parse_serving_size(value: object) -> str:
    if value is None:
        return ""
    match = re.fullmatch(r"\s*\d+(?:\.\d+)?\s*\((\d+(?:\.\d+)?)\)\s*g\s*", str(value), re.IGNORECASE)
    return match.group(1) if match else ""


def enrich(output: Path = DEFAULT_OUTPUT) -> dict:
    output = output.resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_id"] != output.name:
        raise ValueError("Dataset ID and output directory differ")
    rows = []
    statuses = Counter()
    missing = Counter()
    with (output / "recipes.csv").open(encoding="utf-8", newline="") as stream:
        for recipe in csv.DictReader(stream):
            nutrition = json.loads(recipe["nutrition_json"]) if recipe["nutrition_json"] else {}
            values = {"recipe_id": recipe["recipe_id"],
                      "serving_size_g": parse_serving_size(nutrition.get("servingSize"))}
            for output_name, (source_name, unit) in SPECS.items():
                values[output_name] = parse_number(nutrition.get(source_name), unit)
            absent = [key for key in ("serving_size_g", *SPECS) if not values[key]]
            status = "COMPLETE" if not absent else ("UNPARSEABLE" if len(absent) == 10 else "PARTIAL")
            values.update(parse_status=status, missing_fields=";".join(absent), source_file="recipes.csv")
            rows.append(values)
            statuses[status] += 1
            missing.update(absent)
    if len(rows) != manifest["counts"]["output_recipes"]:
        raise ValueError("Nutrition row count differs from recipe count")
    path = output / "recipe_nutrients.csv"
    write_csv(path, FIELDS, rows)
    manifest["counts"]["output_recipe_nutrients"] = len(rows)
    manifest["counts"]["nutrition_parse_complete"] = statuses["COMPLETE"]
    manifest["counts"]["nutrition_parse_partial"] = statuses["PARTIAL"]
    manifest["counts"]["nutrition_parse_unparseable"] = statuses["UNPARSEABLE"]
    manifest["nutrition_missing_by_field"] = dict(sorted(missing.items()))
    manifest["output_sha256"]["recipe_nutrients"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"recipes": len(rows), "parse_status": dict(statuses), "missing_by_field": dict(sorted(missing.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(enrich(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
