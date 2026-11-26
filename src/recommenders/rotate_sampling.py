"""Type-constrained, tail-only negatives for the audited RotatE graph."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from pykeen.sampling import NegativeSampler


TAIL_TYPE = {
    **{f"RATED_{rating}": "Recipe" for rating in range(1, 6)},
    "CONTAINS_MAPPED_INGREDIENT": "Ingredient",
    "MAPS_TO_FOODON": "FoodOnConcept",
    "SUBCLASS_OF": "FoodOnConcept",
}


def build_sampling_inputs(factory, node_path: Path) -> dict:
    """Make relation-specific tail pools from the same verified graph vocabulary."""
    labels = {}
    with Path(node_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[row["id"]] = row["label"]
    pools = {}
    for relation, relation_id in factory.relation_to_id.items():
        if relation not in TAIL_TYPE:
            raise ValueError(f"Unsupported relation: {relation}")
        pool = [entity_id for label, entity_id in factory.entity_to_id.items()
                if labels.get(label) == TAIL_TYPE[relation]]
        if len(pool) < 2:
            raise ValueError(f"Insufficient tail pool for {relation}")
        pools[int(relation_id)] = sorted(pool)
    return {"relation_tail_ids": pools,
            "rating_relation_ids": [int(factory.relation_to_id[f"RATED_{i}"]) for i in range(1, 6)]}


class TypedTailNegativeSampler(NegativeSampler):
    """Corrupt tails within their relation range, excluding known training facts.

    No validation/test interaction is consulted. The unobserved replacements are
    contrastive samples, not negative preference observations.
    """

    def __init__(self, *, mapped_triples, relation_tail_ids, rating_relation_ids,
                 num_negs_per_pos=1, **kwargs):
        super().__init__(mapped_triples=mapped_triples,
                         num_negs_per_pos=num_negs_per_pos, filtered=False, **kwargs)
        if self.num_negs_per_pos < 1:
            raise ValueError("At least one negative per positive is required")
        self.pools = {int(relation): torch.as_tensor(ids, dtype=torch.long)
                      for relation, ids in relation_tail_ids.items()}
        self.rating_relations = set(map(int, rating_relation_ids))
        self.known = set()
        for head, relation, tail in mapped_triples.detach().cpu().tolist():
            key = (head, "RATED" if relation in self.rating_relations else relation, tail)
            self.known.add(key)
        for relation in set(mapped_triples[:, 1].detach().cpu().tolist()):
            if int(relation) not in self.pools:
                raise ValueError(f"No tail pool for relation {relation}")

    def corrupt_batch(self, positive_batch):
        originals = positive_batch.detach().cpu().tolist()
        negatives = positive_batch.unsqueeze(1).repeat(1, self.num_negs_per_pos, 1)
        for row_index, (head, relation, original_tail) in enumerate(originals):
            pool = self.pools[relation]
            key_relation = "RATED" if relation in self.rating_relations else relation
            # Try random draws, then a deterministic scan to guarantee termination.
            for column in range(self.num_negs_per_pos):
                chosen = None
                for _ in range(32):
                    candidate = int(pool[torch.randint(len(pool), (1,))].item())
                    if candidate != original_tail and (head, key_relation, candidate) not in self.known:
                        chosen = candidate
                        break
                if chosen is None:
                    for candidate in pool.tolist():
                        if candidate != original_tail and (head, key_relation, candidate) not in self.known:
                            chosen = candidate
                            break
                if chosen is None:
                    raise ValueError(f"No admissible negative for relation {relation} and head {head}")
                negatives[row_index, column, 2] = chosen
        return negatives
