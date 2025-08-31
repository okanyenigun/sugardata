import regex as re
from typing import List, Dict, Any
from .schemas import NERConfig, LocalNERText, NERResponse, NerOutput
from ..base import NlpTask
from ...components.standard_chain_builder import StandardChainBuilder


class NERLocalizer(NlpTask):

    def __init__(self, config: NERConfig):
        self.config = config

    def generate(self, examples: List[Dict[str, Any]], target_language: str) -> NerOutput:
        label_mapping = self._assign_labels_to_entities(examples)

        batches = self._compose_batches(examples, target_language)

        generated_text_result = self._generate_text(batches)

        remap_label_entity = self._remap_labels_to_entities(
            generated_text_result, label_mapping, examples)

        results = self._tokenize_and_label(
            generated_text_result, remap_label_entity)

        parsed_data = self._parse_data(
            generated_text_result, results, remap_label_entity, label_mapping)

        output = self._convert_to_output(parsed_data, NERResponse)
        return output

    def _assign_labels_to_entities(self, examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
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

    def _compose_batches(self, examples: List[Dict[str, Any]], target_language: str) -> List[Dict[str, Any]]:
        batches = []
        for i in range(len(examples)):
            batch = {
                "index": i,
                "reference_text": examples[i]["text"],
                "target_word_mapping": examples[i]["entity_map"],
                "target_language": target_language,
            }
            batches.append(batch)
        return batches

    def _generate_text(self, batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chain = StandardChainBuilder(
            prompt_template=self.config.localization_prompt,
            llm=self.config.llm,
            entity_model=LocalNERText,
        ).build_chain()

        results = []
        for i in range(0, len(batches), self.config.batch_size):
            batch = batches[i:i + self.config.batch_size]
            try:
                responses = chain.batch(batch)
            except Exception as e:
                if self.config.verbose:
                    print(
                        f"Warning: Error processing batch {i//self.config.batch_size}: {e}. Continuing with next batch.")
            if not responses:
                raise ValueError(
                    "No responses received from the chain. Please check your configuration and input data.")
            for response in responses:
                response_dict = response.model_dump()
                results.append(response_dict)
        return results

    def _remap_labels_to_entities(
            self,
            generated_text_result:  List[Dict[str, Any]],
            label_mapping: Dict[str, Dict[str, int]],
            examples:  List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        remap = {}
        for i in range(len(generated_text_result)):
            gen_record = generated_text_result[i]
            org_record_entity_map = examples[i]["entity_map"]

            for key, value in gen_record["localized_word_mappings"].items():
                entity_type = org_record_entity_map.get(key)
                remap[key] = label_mapping.get(entity_type)
                remap[value] = label_mapping.get(entity_type)

        return remap

    def _tokenize_and_label(
            self,
            generated_text_result: List[Dict[str, Any]],
            remap_label_entity: Dict[str, Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        results = []

        for item in generated_text_result:
            text = item.get("localized_text", "") or ""
            tokens = self._get_tokens(text)
            norm_tokens = [self._normalize(t) for t in tokens]
            merged_labels = [0] * len(tokens)

            lwm = item.get("localized_word_mappings", {}) or {}
            entities = set()
            for k, v in lwm.items():
                if isinstance(k, str) and k.strip():
                    entities.add(k)
                if isinstance(v, str) and v.strip():
                    entities.add(v)

            for ent in entities:
                # Skip entities that don't have b/i mapping
                label_pair = remap_label_entity.get(ent)
                if not label_pair:
                    continue

                b_val = label_pair["b"]
                i_val = label_pair["i"]

                ent_tokens = self._get_tokens(ent)
                ent_norm = [self._normalize(t)
                            for t in ent_tokens if t.strip()]
                if not ent_norm:
                    continue

                m = len(ent_norm)
                i = 0
                while i <= len(norm_tokens) - m:
                    if norm_tokens[i:i+m] == ent_norm:
                        # Assign B to first, I to rest; merge via max
                        merged_labels[i] = max(merged_labels[i], b_val)
                        for k in range(1, m):
                            merged_labels[i+k] = max(merged_labels[i+k], i_val)
                        i += m
                    else:
                        i += 1
            results.append({
                "index": item.get("index"),
                "tokens": tokens,
                "labels": merged_labels,
            })

        return results

    def _normalize(self, tok: str) -> str:
        for apo in ("'", "’"):
            if apo in tok:
                tok = tok.split(apo, 1)[0]
                break
        return tok.casefold()

    def _get_tokens(self, text: str) -> List[str]:
        tokens = text.split(" ")
        pattern = re.compile(
            r"\p{L}[\p{L}\p{M}\p{N}_’']*|\p{N}+|[^\s]", re.UNICODE
        )
        tokens = pattern.findall(text)
        return tokens

    def _parse_data(
        self,
        generated_text_result: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        remap_label_entity: Dict[str, Dict[str, int]],
        label_mapping: Dict[str, Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        output = []
        for i in range(len(generated_text_result)):
            index = results[i]["index"]
            text = generated_text_result[i]["localized_text"]
            tokens = results[i]["tokens"]
            labels = results[i]["labels"]
            entities = list(
                generated_text_result[i]["localized_word_mappings"].values())

            label_mappings = []
            for entity in entities:
                label_maps = remap_label_entity.get(entity)
                for key, value in label_mapping.items():
                    if value == label_maps:
                        entity_type = key
                        break

                label_mappings.append({
                    "entity_type": entity_type,
                    "labeling": label_maps,
                    "entity": entity
                })
            output.append({
                "index": index,
                "text": text,
                "tokens": tokens,
                "labels": labels,
                "label_mappings": label_mappings,
            })

        return output

    def _assign_labels_to_entities(self, examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """
         {'organization': {'b': 1, 'i': 2}, ...}
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
