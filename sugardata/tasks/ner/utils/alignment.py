from typing import Dict, List, Any, Tuple, Iterable, Optional


class TokenAlignmentWorker:
    """
    Aligns string-level entity labels to token-level numeric tags.

    - Supports WordPiece style tokens (prefix '##').
    - Respects punctuation spacing (apostrophes attach to the left, etc.).
    - Resolves overlaps by preferring longer spans, then earlier start.
    - Returns only the token-level tag ids (outside = outside_id).

    Parameters
    ----------
    outside_id : int
        Label id for non-entity tokens.
    case_insensitive : bool
        If True, matches are case-insensitive.
    right_attach : Iterable[str]
        Tokens that attach to the left side with no space (e.g., punctuation, apostrophe).
    left_attach : Iterable[str]
        Tokens that do not want a space after themselves (e.g., opening brackets/quotes).
    """

    def __init__(
        self,
        outside_id: int = 0,
        case_insensitive: bool = False,
        right_attach: Optional[Iterable[str]] = None,
        left_attach: Optional[Iterable[str]] = None,
    ) -> None:
        self.outside_id = outside_id
        self.case_insensitive = case_insensitive
        self.RIGHT_ATTACH = set(right_attach or {
            ".", ",", "!", "?", ":", ";", ")", "]", "}", "”", "’", '"', "'", "…", "–", "—", "-", "%"
        })
        self.LEFT_ATTACH = set(left_attach or {"(", "[", "{", "“", "‘"})

    # ---------- public API ----------

    def align(self, tokens: List[str], tag_labels: Dict[str, Dict[str, Any]]) -> List[int]:
        """
        Align entity strings in `tag_labels` onto `tokens` and return token-level numeric tags.

        Args
        ----
        tokens : List[str]
            Tokenized sequence (WordPiece tokens accepted).
        tag_labels : Dict[str, Dict[str, Any]]
            Mapping: entity string -> {"b": int, "i": int}.

        Returns
        -------
        List[int]
            Token-level tag ids aligned to `tokens`.
        """
        n = len(tokens)
        tags = [self.outside_id] * n

        # Pre-cache joined windows for efficiency
        joined_cache: Dict[Tuple[int, int], str] = {}

        def join_window(i: int, j: int) -> str:
            key = (i, j)
            if key not in joined_cache:
                joined_cache[key] = self._join_tokens(tokens, i, j)
            return joined_cache[key]

        # Prepare entities (optionally case-normalized)
        if self.case_insensitive:
            def norm(s): return s.casefold()
            ent_texts = {norm(e): e for e in tag_labels}
            entity_keys = list(ent_texts.keys())
        else:
            def norm(s): return s
            ent_texts = {e: e for e in tag_labels}
            entity_keys = list(tag_labels.keys())

        # Find candidate spans (start, end, label_dict)
        candidates: List[Tuple[int, int, Dict[str, Any]]] = []
        for ent_norm_key in entity_keys:
            # original entity key (for fetching label ids)
            original_ent = ent_texts[ent_norm_key]
            for i in range(n):
                for j in range(i + 1, n + 1):
                    s = join_window(i, j)
                    if norm(s) == ent_norm_key:
                        candidates.append((i, j, tag_labels[original_ent]))

        # Prefer longer spans, then earlier start; avoid overlaps
        candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        occupied = [False] * n

        for s, e, lbls in candidates:
            if any(occupied[k] for k in range(s, e)):
                continue
            # write tags
            tags[s] = int(lbls["b"])
            for k in range(s + 1, e):
                tags[k] = int(lbls["i"])
            for k in range(s, e):
                occupied[k] = True

        return tags

    # ---------- helpers ----------

    def _surface(self, tok: str) -> str:
        return tok[2:] if tok.startswith("##") else tok

    def _needs_space(self, prev_tok: Optional[str], cur_tok: str) -> bool:
        if prev_tok is None:
            return False
        if cur_tok.startswith("##"):
            return False
        if cur_tok in self.RIGHT_ATTACH:
            return False
        if prev_tok in self.LEFT_ATTACH:
            return False
        return True

    def _join_tokens(self, tokens: List[str], start: int, end: int) -> str:
        out: List[str] = []
        prev: Optional[str] = None
        for i in range(start, end):
            t = tokens[i]
            if self._needs_space(prev, t):
                out.append(" ")
            out.append(self._surface(t))
            prev = t
        return "".join(out)
