"""Executable CQ1-CQ10 over the Neo4j-derived graph snapshot."""
from dataclasses import asdict

from src.kg.compatibility import ConstraintEngine, confirmed, uncertain


def run_competency_queries(engine: ConstraintEngine, profile_id: str, diet_id: str = "diet_vegan") -> dict:
    diet = engine.entity("DietaryPattern", diet_id)
    dishes = sorted((entity for entity in engine.data.entities if entity.label == "Dish"), key=lambda entity: entity.id)
    decisions = [engine.evaluate(profile_id, dish.id) for dish in dishes]
    groups = engine.candidate_groups(profile_id)
    confirmed_diet_dishes = [dish.id for dish in dishes if any(not uncertain(edge) for edge in engine.edges(dish.semantic_uri, "COMPATIBLE_WITH_DIET", diet.semantic_uri))]
    allergy_reasons = [{"dish_id": result.dish_id, **reason} for result in decisions for reason in result.reasons
                       if reason["constraint_type"] == "ALLERGY" and reason["outcome"] == "INCOMPATIBLE"]
    avoidance_reasons = [{"dish_id": result.dish_id, **reason} for result in decisions for reason in result.reasons
                         if reason["profile_relation"] == "MUST_AVOID"]
    rule_reasons = [{"dish_id": result.dish_id, **reason} for result in decisions for reason in result.reasons
                    if reason["profile_relation"] == "REQUIRES_DIET" and any(step["relationship_type"] in {"EXCLUDES_INGREDIENT", "EXCLUDES_ALLERGEN"} for step in reason["path"])]
    conflicts = []
    profile = engine.entity("UserProfile", profile_id)
    liked = engine.edges(profile.semantic_uri, "LIKES_INGREDIENT")
    for result in decisions:
        for reason in result.reasons:
            for preference in liked:
                if reason["outcome"] == "INCOMPATIBLE" and reason.get("ingredient_id") == engine.entities[preference.target_uri].id:
                    conflicts.append({"dish_id": result.dish_id, "status": result.status, "reason": reason,
                                      "preference_evidence": engine.path([preference])})
    return {"CQ1": confirmed_diet_dishes, "CQ2": allergy_reasons, "CQ3": avoidance_reasons,
            "CQ4": rule_reasons, "CQ5": engine.eligible_restaurants(profile_id),
            "CQ6": engine.soft_preferences(profile_id), "CQ7": conflicts,
            "CQ8": [asdict(result) for result in decisions if result.status == "INCOMPATIBLE"],
            "CQ9": [asdict(result) for result in decisions if result.status == "UNKNOWN"],
            "CQ10": [match for row in groups["COMPATIBLE"] for match in row["soft_matches"]]}
