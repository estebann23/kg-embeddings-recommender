# Competency queries over the graph

from dataclasses import asdict

from src.kg.compatibility import ConstraintEngine


def run_competency_queries(engine: ConstraintEngine, profile_id: str) -> dict:
    """Return recipe decisions and their ingredient/rule evidence paths."""
    engine.entity("ScenarioProfile", profile_id)
    recipes = sorted(
        (entity for entity in engine.data.entities if entity.label == "Recipe"),
        key=lambda entity: entity.id,
    )
    decisions = [engine.evaluate(profile_id, recipe.id) for recipe in recipes]
    return {
        "compatible_recipes": [result.recipe_id for result in decisions if result.status == "COMPATIBLE"],
        "incompatible_recipes": [asdict(result) for result in decisions if result.status == "INCOMPATIBLE"],
        "unknown_recipes": [asdict(result) for result in decisions if result.status == "UNKNOWN"],
    }
