import re
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Optional, Dict, Any


Span = Tuple[int, int, str]  # (start_idx, end_idx, label)


@dataclass
class NERExampleFormatter:
    """
    Generic builder for NER examples from BIO tags or spans.
    
    Parameters
    ----------
    wanted_labels : Optional[Iterable[str]] -> Example: ["PERSON", "ORG", "LOC"]
    label_style : str
        'full' or 'short', defines how labels are represented.
    """
    wanted_labels: Optional[Iterable[str]] = None
    full_to_short: Optional[Dict[str, str]] = None

    _wanted_set: Optional[set] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._wanted_set = set(self.wanted_labels) if self.wanted_labels else None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def build_from_bio(
        self,
        token_seqs: Iterable[List[str]],
        tag_seqs: Iterable[List[str]],
    ) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for tokens, tags in zip(token_seqs, tag_seqs):
            spans = self.bio_to_spans(tags)
            examples.append(self._tokens_and_spans_to_example(tokens, spans))
        return examples
    
    def build_from_spans(
        self,
        token_seqs: Iterable[List[str]],
        span_seqs: Iterable[List[Span]],
    ) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for tokens, spans in zip(token_seqs, span_seqs):
            examples.append(self._tokens_and_spans_to_example(tokens, spans))
        return examples

    # ---------------------------------------------------------------------
    # BIO -> spans
    # ---------------------------------------------------------------------
    def bio_to_spans(self, tags: List[str]) -> List[Span]:
        spans: List[Span] = []
        i, n = 0, len(tags)

        while i < n:
            tag = tags[i]

            if tag == "O":
                i += 1
                continue

            if tag.startswith(("B-", "I-")):
                prefix, label = tag.split("-", 1)

                # Normalize stray I- into B- if configured
                if prefix == "I" and self.treat_stray_I_as_B:
                    prefix = "B"

                if prefix == "B":
                    start = i
                    j = i + 1
                    # extend while contiguous I-L of same label
                    while j < n and tags[j].startswith("I-") and tags[j].endswith(label):
                        j += 1
                    end = j - 1

                    if (self._wanted_set is None) or (label in self._wanted_set):
                        spans.append((start, end, label))

                    i = j
                    continue

            # Unknown/malformed tag → skip
            i += 1

        return spans
    
    # ---------------------------------------------------------------------
    # Detokenization
    # ---------------------------------------------------------------------
    def detokenize(self, tokens: List[str]) -> str:
        if not tokens:
            return ""

        text = " ".join(tokens)

        # punctuation spacing
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # possessives & contractions
        text = re.sub(r"\s+'s\b", r"'s", text)
        text = re.sub(r"\s+'re\b", r"'re", text)
        text = re.sub(r"\s+'ve\b", r"'ve", text)
        text = re.sub(r"\s+'ll\b", r"'ll", text)
        text = re.sub(r"\s+'d\b", r"'d", text)
        text = re.sub(r"\s+n['’]t\b", r"n't", text)

        # quotes & parentheses
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        text = re.sub(r"\s+'", "'", text)
        text = re.sub(r"'\s+", "'", text)

        # percent & currency
        text = text.replace(" %", "%").replace("$ ", "$")

        # compact spaces
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    
    # ---------------------------------------------------------------------
    # Assembly
    # ---------------------------------------------------------------------
    def _label_map(self) -> Dict[str, str]:
        """
        Return the user-provided label mapping, or an empty mapping if none.
        When applied, unknown labels fall back to themselves.
        """
        return self.full_to_short or {}
    
    def _tokens_and_spans_to_example(self, tokens: List[str], spans: List[Span]) -> Dict[str, Any]:
        label_map = self._label_map()
        text = self.detokenize(tokens)

        # deterministic ordering
        spans = sorted(spans, key=lambda x: (x[0], x[1]))

        ner_list: List[Dict[str, str]] = []
        for s, e, lab in spans:
            surface = self.detokenize(tokens[s : e + 1])
            mapped = label_map.get(lab, lab)
            ner_list.append({surface: mapped})

        return {"text": text, "ner_tags": ner_list}
