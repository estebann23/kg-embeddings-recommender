"""One evidence-aware constraint policy shared by future recommenders."""
from dataclasses import asdict

from src.kg.models import CompatibilityResult, GraphData, SemanticRelationship
from src.kg.semantics import relation_definitions


def confirmed(edge: SemanticRelationship) -> bool:
    return any(record.evidence_state == "CONFIRMED" for record in edge.observations)


def uncertain(edge: SemanticRelationship) -> bool:
    # Records here all describe the same positive RDF assertion. A confirmed
    # observation establishes it even if an older source only inferred it.
    # Contradictory positive/negative diet assertions are handled separately.
    return not confirmed(edge)


class ConstraintEngine:
    """Evaluate a consistent graph snapshot obtained through GraphRepository."""

    def __init__(self, data: GraphData):
        self.data = data
        self.entities = {entity.semantic_uri: entity for entity in data.entities}
        self.by_id = {(entity.label, entity.id): entity for entity in data.entities}
        self.definitions = relation_definitions()
        self.outgoing = {}
        for edge in data.relationships:
            self.outgoing.setdefault(edge.source_uri, []).append(edge)

    def entity(self, label: str, identifier: str):
        try:
            return self.by_id[label, identifier]
        except KeyError:
            raise ValueError(f"Unknown {label} ID: {identifier}") from None

    def edges(self, source_uri: str, relationship_type: str | None = None, target_uri: str | None = None):
        return [edge for edge in self.outgoing.get(source_uri, [])
                if (relationship_type is None or edge.relationship_type == relationship_type)
                and (target_uri is None or edge.target_uri == target_uri)]

    def path(self, edges) -> list[dict]:
        return [{"relationship_id": edge.id, "relationship_type": edge.relationship_type,
                 "source_id": self.entities[edge.source_uri].id,
                 "target_id": self.entities[edge.target_uri].id,
                 "observations": [asdict(record) for record in edge.observations]} for edge in edges]

    def evaluate(self, profile_id: str, dish_id: str) -> CompatibilityResult:
        profile = self.entity("UserProfile", profile_id)
        dish = self.entity("Dish", dish_id)
        ingredients = self.edges(dish.semantic_uri, "CONTAINS")
        reasons = []

        def record(outcome, constraint_type, message, path, **values):
            reasons.append({"outcome": outcome, "constraint_type": constraint_type,
                            "profile_relation": path[0].relationship_type if path else None,
                            "dish_id": dish.id, "reason": message, "path": self.path(path), **values})

        def clearance(profile_edge, rule_edge=None):
            path = [profile_edge] + ([rule_edge] if rule_edge else [])
            if dish.properties.get("composition_complete") is not True:
                record("UNKNOWN", "COMPOSITION", "ingredient composition incomplete or not asserted", path,
                       composition_complete=dish.properties.get("composition_complete"))
            elif any(uncertain(edge) for edge in ingredients):
                record("UNKNOWN", "COMPOSITION", "ingredient composition contains uncertain observations", path + ingredients)

        def avoid(profile_edge, ingredient_uri, rule_edge=None):
            prefix = [profile_edge] + ([rule_edge] if rule_edge else [])
            known_requirement = confirmed(profile_edge) and (rule_edge is None or confirmed(rule_edge))
            matches = [edge for edge in ingredients if edge.target_uri == ingredient_uri]
            for match in matches:
                status = "INCOMPATIBLE" if known_requirement and confirmed(match) else "UNKNOWN"
                record(status, "INGREDIENT_EXCLUSION", "excluded ingredient evidence", prefix + [match],
                       ingredient_id=self.entities[ingredient_uri].id)
            if not matches:
                clearance(profile_edge, rule_edge)

        def allergy(profile_edge, allergen_uri, rule_edge=None):
            prefix = [profile_edge] + ([rule_edge] if rule_edge else [])
            known_requirement = confirmed(profile_edge) and (rule_edge is None or confirmed(rule_edge))
            conflicts = []
            for ingredient_edge in ingredients:
                for allergen_edge in self.edges(ingredient_edge.target_uri, "HAS_ALLERGEN", allergen_uri):
                    conflicts.append((ingredient_edge, allergen_edge))
                    status = "INCOMPATIBLE" if known_requirement and confirmed(ingredient_edge) and confirmed(allergen_edge) else "UNKNOWN"
                    record(status, "ALLERGY", "ingredient-to-allergen evidence", prefix + [ingredient_edge, allergen_edge],
                           ingredient_id=self.entities[ingredient_edge.target_uri].id,
                           allergen_id=self.entities[allergen_uri].id)
            if not conflicts:
                record("UNKNOWN", "ALLERGY", "no explicit evidence establishing allergen absence", prefix,
                       allergen_id=self.entities[allergen_uri].id)
            clearance(profile_edge, rule_edge)

        hard_edges = [edge for edge in self.edges(profile.semantic_uri) if self.definitions[edge.relationship_type].kind == "hard"]
        for profile_edge in hard_edges:
            relationship_type = profile_edge.relationship_type
            if uncertain(profile_edge):
                record("UNKNOWN", "REQUIREMENT", "mandatory profile relationship has uncertain evidence", [profile_edge])
            if relationship_type == "MUST_AVOID":
                avoid(profile_edge, profile_edge.target_uri)
            elif relationship_type == "ALLERGIC_TO":
                allergy(profile_edge, profile_edge.target_uri)
            else:
                if relationship_type == "REQUIRES_DIET":
                    positive = self.edges(dish.semantic_uri, "COMPATIBLE_WITH_DIET", profile_edge.target_uri)
                    negative = self.edges(dish.semantic_uri, "INCOMPATIBLE_WITH_DIET", profile_edge.target_uri)
                    if positive and negative:
                        record("UNKNOWN", "DIET", "contradictory dietary assessments require review", [profile_edge] + positive + negative)
                    for edge in negative:
                        outcome = "INCOMPATIBLE" if confirmed(profile_edge) and confirmed(edge) else "UNKNOWN"
                        record(outcome, "DIET", "negative dietary assessment", [profile_edge, edge], diet_id=self.entities[profile_edge.target_uri].id)
                    if not positive or any(uncertain(edge) for edge in positive):
                        record("UNKNOWN", "DIET", "confirmed positive dietary assessment missing or uncertain", [profile_edge] + positive,
                               diet_id=self.entities[profile_edge.target_uri].id)
                rules = [edge for edge in self.edges(profile_edge.target_uri) if edge.relationship_type in {"EXCLUDES_INGREDIENT", "EXCLUDES_ALLERGEN"}]
                if not rules:
                    record("UNKNOWN", "RULE", "mandatory pattern, restriction, or goal has no documented exclusion rules", [profile_edge])
                for rule in rules:
                    if uncertain(rule):
                        record("UNKNOWN", "RULE", "mandatory exclusion rule has uncertain evidence", [profile_edge, rule])
                    if rule.relationship_type == "EXCLUDES_INGREDIENT":
                        avoid(profile_edge, rule.target_uri, rule)
                    else:
                        allergy(profile_edge, rule.target_uri, rule)
        outcomes = {reason["outcome"] for reason in reasons}
        status = "INCOMPATIBLE" if "INCOMPATIBLE" in outcomes else "UNKNOWN" if "UNKNOWN" in outcomes else "COMPATIBLE"
        return CompatibilityResult(profile_id, dish_id, status, tuple(reasons))

    def soft_preferences(self, profile_id: str) -> list[dict]:
        profile = self.entity("UserProfile", profile_id)
        return [{"profile_id": profile.id, "value_id": self.entities[edge.target_uri].id,
                 "relationship_type": edge.relationship_type, "path": self.path([edge]),
                 "weights": [record.weight for record in edge.observations]}
                for edge in self.edges(profile.semantic_uri) if self.definitions[edge.relationship_type].kind == "soft"]

    def soft_matches(self, profile_id: str, dish_id: str) -> list[dict]:
        profile = self.entity("UserProfile", profile_id)
        dish = self.entity("Dish", dish_id)
        result = []
        target_relationships = {"LIKES_INGREDIENT": "CONTAINS", "DISLIKES_INGREDIENT": "CONTAINS",
                                "PREFERS_CUISINE": "HAS_CUISINE", "PREFERS_TASTE": "HAS_TASTE", "FOLLOWS_DIET": "COMPATIBLE_WITH_DIET"}
        for preference in self.edges(profile.semantic_uri):
            if self.definitions[preference.relationship_type].kind != "soft" or uncertain(preference):
                continue
            paths = []
            effect = "penalty" if preference.relationship_type == "DISLIKES_INGREDIENT" else "positive"
            if preference.relationship_type in target_relationships:
                matches = self.edges(dish.semantic_uri, target_relationships[preference.relationship_type], preference.target_uri)
                paths = [[preference, match] for match in matches if not uncertain(match)]
            elif preference.relationship_type == "PREFERS_HEALTH_GOAL":
                rules = self.edges(preference.target_uri, "EXCLUDES_INGREDIENT")
                allergen_rules = self.edges(preference.target_uri, "EXCLUDES_ALLERGEN")
                for rule in allergen_rules:
                    if uncertain(rule):
                        continue
                    for ingredient in self.edges(dish.semantic_uri, "CONTAINS"):
                        for allergen in self.edges(ingredient.target_uri, "HAS_ALLERGEN", rule.target_uri):
                            if not uncertain(ingredient) and not uncertain(allergen):
                                paths.append([preference, rule, ingredient, allergen])
                                effect = "penalty"
                if rules and all(not uncertain(rule) for rule in rules):
                    conflicts = [(rule, match) for rule in rules for match in self.edges(dish.semantic_uri, "CONTAINS", rule.target_uri) if confirmed(match)]
                    if conflicts:
                        paths.extend([[preference, rule, match] for rule, match in conflicts])
                        effect = "penalty"
                    elif dish.properties.get("composition_complete") is True and all(not uncertain(edge) for edge in self.edges(dish.semantic_uri, "CONTAINS")):
                        # Only ingredient exclusions can be cleared from explicit complete composition.
                        if not allergen_rules:
                            paths = [[preference] + rules + self.edges(dish.semantic_uri, "CONTAINS")]
            for path in paths:
                result.append({"profile_id": profile.id, "dish_id": dish.id, "value_id": self.entities[preference.target_uri].id,
                               "preference_type": preference.relationship_type, "effect": effect,
                               "weights": [record.weight for record in preference.observations], "path": self.path(path),
                               "composition_complete": dish.properties.get("composition_complete"),
                               "node_evidence": [{"entity_id": dish.id, "property": "composition_complete",
                                                  "value": dish.properties.get("composition_complete"),
                                                  "source": dish.properties.get("source", [])}]})
        return result

    def candidate_groups(self, profile_id: str) -> dict[str, list[dict]]:
        self.entity("UserProfile", profile_id)
        groups = {status: [] for status in ("COMPATIBLE", "INCOMPATIBLE", "UNKNOWN")}
        for restaurant in sorted((entity for entity in self.data.entities if entity.label == "Restaurant"), key=lambda entity: entity.id):
            for offer in self.edges(restaurant.semantic_uri, "OFFERS"):
                dish = self.entities[offer.target_uri]
                result = self.evaluate(profile_id, dish.id)
                status = result.status
                reasons = list(result.reasons)
                if status == "COMPATIBLE" and uncertain(offer):
                    status = "UNKNOWN"
                    reasons.append({"outcome": "UNKNOWN", "constraint_type": "ASSOCIATION", "reason": "restaurant association uncertain", "path": self.path([offer])})
                matches = self.soft_matches(profile_id, dish.id) if status == "COMPATIBLE" else []
                for match in matches:
                    match["restaurant_id"] = restaurant.id
                    match["path"].extend(self.path([offer]))
                groups[status].append({"profile_id": profile_id, "restaurant_id": restaurant.id, "restaurant_name": restaurant.name,
                    "dish_id": dish.id, "dish_name": dish.name, "compatibility_status": status, "reasons": reasons,
                    "association_evidence": self.path([offer]), "soft_matches": matches})
        return groups

    def eligible_restaurants(self, profile_id: str) -> list[str]:
        return sorted({row["restaurant_id"] for row in self.candidate_groups(profile_id)["COMPATIBLE"]})


class CompatibilityService:
    def __init__(self, repository):
        self.repository = repository

    def evaluate(self, profile_id: str, dish_id: str) -> CompatibilityResult:
        return ConstraintEngine(self.repository.graph_data()).evaluate(profile_id, dish_id)

    def candidate_groups(self, profile_id: str) -> dict[str, list[dict]]:
        return ConstraintEngine(self.repository.graph_data()).candidate_groups(profile_id)
