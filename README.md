# Knowledge Graph Embeddings-based Restaurant Recommendation Engine

The RDF/OWL vocabulary and SHACL shapes define the semantic model. Neo4j is
the operational graph. The validated RDF evaluates hard constraints, and returns candidates grouped
as `COMPATIBLE`, `INCOMPATIBLE`, or `UNKNOWN`, with graph-backed preference
evidence.

## Inspect the graph in Neo4j

```cypher
MATCH (n:FoodPreferenceKG {dataset_id: 'synthetic_v1'})
UNWIND [label IN labels(n) WHERE label <> 'FoodPreferenceKG'] AS label
RETURN label, count(*) AS count ORDER BY label;
```

```cypher
MATCH (p:UserProfile {id: 'seed_profile_004'})-[r]->(x)
RETURN p, r, x;
```

```cypher
MATCH (p:UserProfile {id: 'seed_profile_004'})-[:PREFERS_TASTE]->(t:Taste)
      <-[:HAS_TASTE]-(d:Dish)<-[:OFFERS]-(r:Restaurant)
RETURN p, t, d, r;
```

Inspect all observations:

```cypher
MATCH (:Dish {id: 'seed_dish_002'})-[r:CONTAINS]->(:Ingredient {id: 'ingredient_cashew'})
MATCH (e:EvidenceRecord) WHERE e.relationship_id = r.id
RETURN r, e ORDER BY e.semantic_uri;
```

The architecture is:

```text
Existing RDF/OWL vocabulary + source RDF
    -> SHACL validation -> RDF/Neo4j mapping -> atomic Neo4j ingestion
    -> evidence-aware constraints -> COMPATIBLE / INCOMPATIBLE / UNKNOWN
    -> candidate restaurants/dishes + graph-backed soft-preference evidence
```
