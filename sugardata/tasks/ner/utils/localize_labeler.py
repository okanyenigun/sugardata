from typing import Dict, List, Tuple


def split_tokens_with_regex(text: str) -> List[str]:
    tokens = text.split(" ")
    pattern = re.compile(
        r"\p{L}[\p{L}\p{M}\p{N}_’']*|\p{N}+|[^\s]", re.UNICODE
    )
    tokens = pattern.findall(text)
    return tokens


# Use the `regex` module for \p{...} Unicode properties (pip install regex)
try:
    import regex as re
except Exception:
    import re  # fallback, but doesn't support \p{...} fully

TOKEN_PATTERN = re.compile(
    r"\p{L}[\p{L}\p{M}\p{N}_’']*|\p{N}+|[^\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def _normalize(tok: str) -> str:
    # Strip Turkish-style apostrophe suffixes and casefold for robust matching
    for apo in ("'", "’"):
        if apo in tok:
            tok = tok.split(apo, 1)[0]
            break
    return tok.casefold()


def build_entity_label_map(examples: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Scan all examples to get unique entity types and assign numeric B/I labels:
      type_1: B=1, I=2
      type_2: B=3, I=4
      ...
    Deterministic ordering via sorted unique types.
    """
    types = []
    for ex in examples:
        types.extend(ex.get("entity_map", {}).values())
    unique_types = sorted(set(types))

    label_map: Dict[str, Dict[str, int]] = {}
    cur = 1
    for t in unique_types:
        label_map[t] = {"b": cur, "i": cur + 1}
        cur += 2
    return label_map


def label_example(example: Dict, entity_label_map: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[int]]:
    """
    Tokenize the example text, then label tokens for all entities in example['entity_map'].
    Returns (tokens, merged_labels) where merged_labels is a single list via element-wise max.
    """
    text = example["text"]
    ent_map: Dict[str, str] = example.get("entity_map", {})

    tokens = tokenize(text)
    norm_tokens = [_normalize(t) for t in tokens]
    labels = [0] * len(tokens)

    for ent_text, ent_type in ent_map.items():
        # Tokenize the entity text the same way (ensures multiword/Unicode consistency)
        ent_tokens = tokenize(ent_text)
        ent_norm = [_normalize(t) for t in ent_tokens if t.strip()]
        if not ent_norm:
            continue

        b_val = entity_label_map[ent_type]["b"]
        i_val = entity_label_map[ent_type]["i"]

        m = len(ent_norm)
        i = 0
        while i <= len(norm_tokens) - m:
            if norm_tokens[i:i+m] == ent_norm:
                # Propose labels for this span (B then I's)
                span = [i_val] * m
                span[0] = b_val
                # Merge into global labels via max
                for k in range(m):
                    labels[i+k] = max(labels[i+k], span[k])
                i += m  # continue after this match (labels all occurrences)
            else:
                i += 1

    return tokens, labels


def process_examples(examples: List[Dict]):
    """
    Returns:
      entity_label_map: {entity_type: {"b": int, "i": int}}
      processed: list of {"tokens": [...], "labels": [...]}
    """
    entity_label_map = build_entity_label_map(examples)
    processed = []
    for ex in examples:
        tokens, labels = label_example(ex, entity_label_map)
        processed.append({"tokens": tokens, "labels": labels})
    return entity_label_map, processed
