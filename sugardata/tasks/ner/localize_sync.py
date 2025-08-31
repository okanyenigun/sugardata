import regex as re
from typing import List, Dict, Any, Tuple
from .schemas import NERConfig, LocalNERText, NERResponse, NerOutput
from .utils.localize_labeler import split_tokens_with_regex
from .utils.alignment import TokenAlignmentWorker
from ..base import NlpTask
from ...components.standard_chain_builder import StandardChainBuilder


class NERLocalizer(NlpTask):

    def __init__(self, config: NERConfig):
        self.config = config

    def generate(self, examples: List[Dict[str, Any]]) -> NerOutput:
        """
        examples:
            {
                "text": "Arsenal won the First Division.", 
                "ner_tags": {"Arsenal": "organization", "First Division": "organization"}
            },
        """
        self._validate(examples)
        batches = self._compose_batches(examples)
        responses = self._generate_text(batches)
        responses = self._tokenize(responses)
        _, tag_label_maps = self._map_entity_labels(examples)
        responses = self._add_response_ner_tags(
            responses, examples, tag_label_maps)
        responses = self._align_ner_tags(responses)
        return responses

    def _validate(self, examples: List[Dict[str, Any]]):
        for example in examples:
            if "text" not in example or "ner_tags" not in example:
                raise ValueError(
                    "Each example must contain 'text' and 'ner_tags'. For example: {'text': 'Arsenal won the First Division.', 'ner_tags': {'Arsenal': 'organization', 'First Division': 'organization'}}")

    def _compose_batches(self, examples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        rows = []
        for i in range(len(examples)):
            row = {
                "index": i,
                "original_text": examples[i]["text"],
                "ner_tags": examples[i]["ner_tags"],
                "target_language": self.config.target_language,
            }
            rows.append(row)

        batches = []
        for b in range(0, len(rows), self.config.batch_size):
            batches.append(rows[b:b + self.config.batch_size])
        return batches

    def _generate_text(self, batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        chain = StandardChainBuilder(
            prompt_template=self.config.localization_prompt,
            llm=self.config.llm,
            entity_model=LocalNERText,
        ).build_chain()

        results = []
        for batch in batches:
            responses = chain.batch(batch)
            responses = [x.model_dump() for x in responses]
            results.extend(responses)
        return results

    def _tokenize(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tokenizer = self.config.tokenizer.tokenize if self.config.tokenizer else split_tokens_with_regex

        for response in responses:
            response["tokens"] = tokenizer(response["localized_text"])

        return responses

    def _map_entity_labels(self, examples: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        unique_tags = self._get_unique_entity_types(examples)
        tag_label_maps = self._generate_label_maps(unique_tags)
        return unique_tags, tag_label_maps

    def _get_unique_entity_types(self, examples: List[Dict[str, Any]]) -> List[str]:
        unique_tags = set()
        for item in examples:
            unique_tags.update(item["ner_tags"].values())
        unique_list = sorted(list(unique_tags))
        return unique_list

    def _generate_label_maps(self, unique_tags: List[str]) -> Dict[str, Dict[str, int]]:
        tag_label_maps = {}
        for i, tag in enumerate(unique_tags):
            tag_label_maps[tag] = {"b": i * 2 + 1, "i": i * 2 + 2}
        return tag_label_maps

    def _add_response_ner_tags(self, responses: List[Dict[str, Any]], examples: List[Dict[str, Any]], tag_label_maps: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
        for i, response in enumerate(responses):
            ner_tag_labels = {}
            original_ner_tags = examples[i]["ner_tags"]
            for key, value in response["localized_word_mappings"].items():
                entity_type = original_ner_tags.get(key)
                val = tag_label_maps.get(entity_type)
                ner_tag_labels[key] = val
                ner_tag_labels[value] = val
            response["ner_tag_labels"] = ner_tag_labels

        return responses

    def _align_ner_tags(self, responses):
        worker = TokenAlignmentWorker(outside_id=0, case_insensitive=False)
        for response in responses:
            tokens = response["tokens"]
            tag_labels = response["ner_tag_labels"]
            token_tags = worker.align(tokens, tag_labels)
            response["token_tags"] = token_tags
        return responses
