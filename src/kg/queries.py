"""Parameterized graph reads and evidence-summary maintenance."""
NODES = """
MATCH (n:FoodPreferenceKG {dataset_id: $dataset_id})
WHERE NOT n:EvidenceRecord
RETURN labels(n) AS labels, properties(n) AS properties ORDER BY n.semantic_uri
"""
RELATIONSHIPS = """
MATCH (s:FoodPreferenceKG {dataset_id: $dataset_id})-[r]->(t:FoodPreferenceKG {dataset_id: $dataset_id})
WHERE r.dataset_id = $dataset_id
RETURN s.semantic_uri AS source_uri, t.semantic_uri AS target_uri,
       type(r) AS relationship_type, properties(r) AS properties ORDER BY r.id
"""
EVIDENCE = """
MATCH (e:EvidenceRecord {dataset_id: $dataset_id})
RETURN properties(e) AS properties ORDER BY e.semantic_uri
"""
SUMMARIZE_EVIDENCE = """
MATCH ()-[r {dataset_id: $dataset_id}]->()
MATCH (e:EvidenceRecord {dataset_id: $dataset_id}) WHERE e.relationship_id = r.id
WITH r, e ORDER BY CASE WHEN e.evidence_state = 'CONFIRMED' THEN 0 ELSE 1 END, e.semantic_uri
WITH r, collect(e) AS records
WITH r, records,
     reduce(states = [], e IN records | CASE WHEN e.evidence_state IN states THEN states ELSE states + e.evidence_state END) AS states,
     reduce(weights = [], e IN records | CASE WHEN e.weight IS NULL OR e.weight IN weights THEN weights ELSE weights + e.weight END) AS weights,
     reduce(sources = [], e IN records | reduce(acc = sources, value IN e.source | CASE WHEN value IN acc THEN acc ELSE acc + value END)) AS sources
SET r.semantic_uri = records[0].semantic_uri,
    r.evidence_uris = [e IN records | e.semantic_uri],
    r.evidence_state = CASE WHEN 'CONFIRMED' IN states THEN 'CONFIRMED' WHEN size(states) = 1 THEN states[0] ELSE 'UNKNOWN' END,
    r.evidence_conflict = size(states) > 1,
    r.source = sources,
    r.source_text = records[0].source_text,
    r.weight = CASE WHEN size(weights) = 1 THEN weights[0] ELSE null END,
    r.weight_conflict = size(weights) > 1
RETURN count(r) AS relationships
"""
COUNTS = """
MATCH (n:FoodPreferenceKG {dataset_id: $dataset_id})
UNWIND [label IN labels(n) WHERE label <> 'FoodPreferenceKG'] AS label
RETURN label, count(*) AS count ORDER BY label
"""
RELATIONSHIP_COUNTS = """
MATCH ()-[r {dataset_id: $dataset_id}]->()
RETURN type(r) AS type, count(*) AS count ORDER BY type
"""
EVIDENCE_COUNTS = """
MATCH (e:EvidenceRecord {dataset_id: $dataset_id})
RETURN e.evidence_state AS state, count(*) AS count ORDER BY state
"""
RELATIONSHIP_EVIDENCE_COUNTS = """
MATCH ()-[r {dataset_id: $dataset_id}]->()
RETURN r.evidence_state AS state, count(*) AS count ORDER BY state
"""
