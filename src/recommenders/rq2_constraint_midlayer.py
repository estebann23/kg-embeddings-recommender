"""KG ingredient compatibility mid-layer for 40 fictional RQ2 user scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.recommenders.contract import DATA, ROOT, sha, verify_lock
from src.recommenders.item_cf import load_recipes, read_reviews, require


OUTPUT = ROOT / "results/rq2_primary_rotate_40"
PILOT = ROOT / "data/processed/hummus_pilot_v1"
CONFIG = ROOT / "config/rq2_primary_rotate_40.json"
RULES = ROOT / "config/rq2_ingredient_rules.json"


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def assign_profiles():
    original = list(rows(PILOT / "rq2_pilot_user_classification.csv"))
    require(len(original) == 40 and len({row["user_id"] for row in original}) == 40,
            "Pilot user population changed")
    assigned = {row["user_id"]: int(row["profile_number"]) for row in original
                if row["profile_number"]}
    remaining = sorted((row["user_id"] for row in original if not row["profile_number"]),
                       key=lambda user: int(user.removeprefix("hummus_user_")))
    require(len(assigned) == 10 and len(remaining) == 30 and
            Counter(assigned.values()) == {i: 2 for i in range(1, 6)},
            "Original RQ2 allocation changed")
    for index, user in enumerate(remaining):
        assigned[user] = index % 5 + 1
    require(len(assigned) == 40 and Counter(assigned.values()) == {i: 8 for i in range(1, 6)},
            "Extended profile assignment incorrect")
    return assigned, {row["user_id"] for row in original if row["profile_number"]}


def preflight():
    verify_lock(DATA)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    require(config["status"] == rules["status"] == "frozen_before_rq2_classification" and
            set(rules["profiles"]) == set(map(str, range(1, 6))), "RQ2 protocol changed")
    assigned, original = assign_profiles()
    train, _ = read_reviews(DATA)
    require(set(assigned) <= set(train) and all(len(train[user]) == 28 for user in assigned),
            "Pilot users absent from RotatE training cohort")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "profile_assignments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "profile_number", "profile_label", "assignment_source"))
        for user in sorted(assigned, key=lambda value: int(value.removeprefix("hummus_user_"))):
            profile = assigned[user]
            writer.writerow((user, profile, rules["profiles"][str(profile)]["label"],
                             "original_pilot_10" if user in original else "balanced_extension_30"))
    result = {"status": "rq2_midlayer_preflight_passed", "users": 40,
              "users_per_profile": {str(i): 8 for i in range(1, 6)},
              "catalog": len(load_recipes(DATA)), "original_assignments": len(original),
              "new_fictional_assignments": 30,
              "config_sha256": sha(CONFIG), "rules_sha256": sha(RULES),
              "assignment_sha256": sha(OUTPUT / "profile_assignments.csv"),
              "pilot_assignment_sha256": sha(PILOT / "rq2_pilot_user_classification.csv"),
              "graph_edges_sha256": sha(DATA / "graph_edges.csv"),
              "graph_nodes_sha256": sha(DATA / "graph_nodes.csv"),
              "raw_lines_sha256": sha(DATA / "raw_ingredient_lines.csv"),
              "recipes_sha256": sha(DATA / "recipes.csv"),
              "reviews_sha256": sha(DATA / "reviews.csv")}
    write_json(OUTPUT / "midlayer_preflight.json", result)
    return result


def require_preflight():
    pre = json.loads((OUTPUT / "midlayer_preflight.json").read_text(encoding="utf-8"))
    require(pre["status"] == "rq2_midlayer_preflight_passed" and
            pre["config_sha256"] == sha(CONFIG) and pre["rules_sha256"] == sha(RULES) and
            pre["assignment_sha256"] == sha(OUTPUT / "profile_assignments.csv") and
            pre["pilot_assignment_sha256"] == sha(PILOT / "rq2_pilot_user_classification.csv") and
            pre["graph_edges_sha256"] == sha(DATA / "graph_edges.csv") and
            pre["graph_nodes_sha256"] == sha(DATA / "graph_nodes.csv") and
            pre["raw_lines_sha256"] == sha(DATA / "raw_ingredient_lines.csv") and
            pre["recipes_sha256"] == sha(DATA / "recipes.csv") and
            pre["reviews_sha256"] == sha(DATA / "reviews.csv"), "RQ2 source seal changed")
    return pre


def classify():
    require_preflight()
    recipes = load_recipes(DATA)
    rule_data = json.loads(RULES.read_text(encoding="utf-8"))
    graph_names = {row["id"]: row["name"] for row in rows(DATA / "graph_nodes.csv")
                   if row["label"] == "Ingredient"}
    graph_ingredients = defaultdict(list)
    edge_count = 0
    for row in rows(DATA / "graph_edges.csv"):
        if row["relation"] == "CONTAINS_MAPPED_INGREDIENT":
            require(row["target_id"] in graph_names and row["evidence_state"] == "INFERRED",
                    "Unexpected graph ingredient evidence")
            graph_ingredients[row["source_id"]].append((row["target_id"], graph_names[row["target_id"]]))
            edge_count += 1
    require(edge_count == 586292 and len(graph_ingredients) == 65984,
            "Graph ingredient projection changed")
    raw_counts = Counter(row["recipe_id"] for row in rows(DATA / "raw_ingredient_lines.csv"))
    raw_text = {row["recipe_id"]: normalize(row["raw_ingredients"])
                for row in rows(DATA / "recipes.csv")}
    require(len(raw_text) == len(raw_counts) == len(recipes) == 75972,
            "Raw ingredient catalog changed")
    compiled = {int(number): [(rule["id"], re.compile(rule["pattern"], re.I),
                               re.compile(rule["exclude"], re.I) if rule["exclude"] else None)
                              for rule in profile["rules"]]
                for number, profile in rule_data["profiles"].items()}
    uncertain_patterns = {int(number): re.compile(profile["uncertain_pattern"], re.I)
                          for number, profile in rule_data["profiles"].items()}
    global_uncertain = re.compile(rule_data["global_uncertain_pattern"], re.I)
    path = OUTPUT / "recipe_profile_compatibility.csv"
    counts = {number: Counter() for number in compiled}
    reconciled_count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("profile_number", "recipe_id", "status", "raw_fragment_count",
                         "graph_ingredient_count", "all_graph_names_in_raw", "reconciled_list",
                         "conflict_rule_id", "conflict_ingredient_id", "conflict_ingredient_name",
                         "ambiguous_ingredient_name"))
        for recipe in recipes:
            mapped = graph_ingredients.get(recipe, [])
            raw = raw_text[recipe]
            supported = [(identifier, name) for identifier, name in mapped
                         if normalize(name) and normalize(name) in raw]
            reconciled = bool(mapped) and len(mapped) == raw_counts[recipe] and len(supported) == len(mapped)
            reconciled_count += reconciled
            for number, rules in compiled.items():
                conflicts = [(rule_id, identifier, name)
                             for identifier, name in supported
                             for rule_id, pattern, exclude in rules
                             if pattern.search(name) and not (exclude and exclude.search(name))]
                conflict = min(conflicts) if conflicts else None
                ambiguous = sorted(name for _, name in supported
                                   if uncertain_patterns[number].search(name) or
                                   global_uncertain.search(name))
                status = ("INCOMPATIBLE" if conflict else
                          "COMPATIBLE" if reconciled and not ambiguous else "UNKNOWN")
                counts[number][status] += 1
                writer.writerow((number, recipe, status, raw_counts[recipe], len(mapped),
                                 str(len(supported) == len(mapped)), str(reconciled),
                                 *(conflict if conflict else ("", "", "")),
                                 ambiguous[0] if ambiguous else ""))
    assignments = {row["user_id"]: int(row["profile_number"])
                   for row in rows(OUTPUT / "profile_assignments.csv")}
    statuses = {number: {} for number in compiled}
    for row in rows(path):
        statuses[int(row["profile_number"])][row["recipe_id"]] = row["status"]
    train, _ = read_reviews(DATA)
    with (OUTPUT / "candidate_coverage_by_user.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("user_id", "profile_number", "base_candidates", "compatible_candidates",
                         "incompatible_candidates", "unknown_candidates", "compatible_fraction",
                         "allow_unknown_candidates", "allow_unknown_fraction"))
        for user in sorted(assignments):
            profile = assignments[user]
            excluded = {recipe for recipe, _ in train[user]}
            tally = Counter(status for recipe, status in statuses[profile].items() if recipe not in excluded)
            require(sum(tally.values()) == 75944, "Candidate partition incomplete")
            writer.writerow((user, profile, 75944, tally["COMPATIBLE"],
                             tally["INCOMPATIBLE"], tally["UNKNOWN"],
                             format(tally["COMPATIBLE"] / 75944, ".17g"),
                             tally["COMPATIBLE"] + tally["UNKNOWN"],
                             format((tally["COMPATIBLE"] + tally["UNKNOWN"]) / 75944, ".17g")))
    result = {"status": "rq2_kg_compatibility_complete", "catalog_recipes": len(recipes),
              "graph_ingredient_edges": edge_count, "recipes_with_graph_ingredients": len(graph_ingredients),
              "recipes_with_reconciled_declared_lists": reconciled_count,
              "profile_status_counts": {str(key): dict(value) for key, value in counts.items()},
              "classification_sha256": sha(path),
              "candidate_coverage_sha256": sha(OUTPUT / "candidate_coverage_by_user.csv"),
              "preflight_sha256": sha(OUTPUT / "midlayer_preflight.json")}
    write_json(OUTPUT / "midlayer_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "classify"))
    print(json.dumps((preflight if parser.parse_args().command == "preflight" else classify)(), indent=2))
