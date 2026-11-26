# constraint policy checkers (Not in use anymore, but useful for reference)

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

    def evaluate(self, profile_id: str, recipe_id: str) -> CompatibilityResult:
        profile = self.entity("ScenarioProfile", profile_id)
        recipe = self.entity("Recipe", recipe_id)
        ingredients = self.edges(recipe.semantic_uri, "CONTAINS")
        reasons = []

        def record(outcome, constraint_type, message, path, **values):
            reasons.append({"outcome": outcome, "constraint_type": constraint_type,
                            "profile_relation": path[0].relationship_type if path else None,
                            "recipe_id": recipe.id, "reason": message, "path": self.path(path), **values})

        def clearance(profile_edge, rule_edge=None):
            path = [profile_edge] + ([rule_edge] if rule_edge else [])
            if recipe.properties.get("composition_complete") is not True:
                record("UNKNOWN", "COMPOSITION", "ingredient composition incomplete or not asserted", path,
                       composition_complete=recipe.properties.get("composition_complete"))
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
                    positive = self.edges(recipe.semantic_uri, "COMPATIBLE_WITH_DIET", profile_edge.target_uri)
                    negative = self.edges(recipe.semantic_uri, "INCOMPATIBLE_WITH_DIET", profile_edge.target_uri)
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
                    record("UNKNOWN", "RULE", "mandatory dietary pattern has no documented exclusion rules", [profile_edge])
                for rule in rules:
                    if uncertain(rule):
                        record("UNKNOWN", "RULE", "mandatory exclusion rule has uncertain evidence", [profile_edge, rule])
                    if rule.relationship_type == "EXCLUDES_INGREDIENT":
                        avoid(profile_edge, rule.target_uri, rule)
                    else:
                        allergy(profile_edge, rule.target_uri, rule)
        outcomes = {reason["outcome"] for reason in reasons}
        status = "INCOMPATIBLE" if "INCOMPATIBLE" in outcomes else "UNKNOWN" if "UNKNOWN" in outcomes else "COMPATIBLE"
        return CompatibilityResult(profile_id, recipe_id, status, tuple(reasons))

    def candidate_groups(self, profile_id: str) -> dict[str, list[dict]]:
        self.entity("ScenarioProfile", profile_id)
        groups = {status: [] for status in ("COMPATIBLE", "INCOMPATIBLE", "UNKNOWN")}
        for recipe in sorted((entity for entity in self.data.entities if entity.label == "Recipe"), key=lambda entity: entity.id):
            result = self.evaluate(profile_id, recipe.id)
            groups[result.status].append({
                "profile_id": profile_id, "recipe_id": recipe.id, "recipe_name": recipe.name,
                "compatibility_status": result.status, "reasons": list(result.reasons),
            })
        return groups

    def eligible_recipes(self, profile_id: str) -> list[str]:
        return sorted(row["recipe_id"] for row in self.candidate_groups(profile_id)["COMPATIBLE"])


class CompatibilityService:
    def __init__(self, repository):
        self.repository = repository

    def evaluate(self, profile_id: str, recipe_id: str) -> CompatibilityResult:
        return ConstraintEngine(self.repository.graph_data()).evaluate(profile_id, recipe_id)

    def candidate_groups(self, profile_id: str) -> dict[str, list[dict]]:
        return ConstraintEngine(self.repository.graph_data()).candidate_groups(profile_id)
