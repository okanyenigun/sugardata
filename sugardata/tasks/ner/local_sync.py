from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
from .schemas import NERLocalizerConfig, NERLocalText, NERLocalResponse, NerOutput
from ..base import NlpTask
from ...components.standard_chain_builder import StandardChainBuilder


class NERLocalizer(NlpTask):

    def __init__(self, config: NERLocalizerConfig):
        self.config = config
        if isinstance(self.config.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        else:
            self.tokenizer = self.config.tokenizer

    def generate(self, examples: List[Dict[str, Any]]) -> NerOutput:
        batches = self._compose_batches(examples)
        generated_text_results = self._generate_text(batches)
        generated_text_results = self._label_generated_text_tokens(examples, generated_text_results)
        return self._convert_to_output(generated_text_results, NERLocalResponse)

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
            prompt_template=self.config.prompt,
            llm=self.config.llm,
            entity_model=NERLocalText,
        ).build_chain()

        results = []
        desc = f"Generating with {str(self.config.model)}"
        iterator = tqdm(batches, desc=desc, disable=not self.config.verbose)
        for batch in iterator:
            responses = chain.batch(batch)
            responses = [x.model_dump() for x in responses]
            results.extend(responses)
        return results
    
    def _label_generated_text_tokens(self, examples: List[Dict[str, Any]], localized_texts:  List[Dict[str, Any]]):
        desc = f"Labeling with {str(self.config.model)}"
        iterator = tqdm(localized_texts, desc=desc, disable=not self.config.verbose)
        for record in iterator:
            localized_text = record["localized_text"]
            index = record["index"]
            example = examples[index]
            ner_tags = example["ner_tags"]
            localized_word_mappings = record["localized_word_mappings"]

            text_tokens = self.tokenizer.tokenize(localized_text)
            labels = [0] * len(text_tokens)

            localized_entity_flattened_list = self._flatten_ner_tags(localized_word_mappings)

            for entity in localized_entity_flattened_list:
                entity_type = self._get_entity_type(ner_tags, entity, localized_word_mappings)
                start_label, end_label = self._get_entity_type_labels(entity_type)
                labels = self._label_entities(entity, text_tokens, labels, start_label, end_label)
            
            record["tokens"] = text_tokens
            record["ner_tags"] = labels
            record["ner_tag_labels"] = self.config.entity_labels
        return localized_texts

    def _flatten_ner_tags(self, localized_word_mappings: Dict[str, str]) -> List[str]:
        localized_word_mappings_reversed = {v: k for k, v in localized_word_mappings.items()}
        localized_word_mappings.update(localized_word_mappings_reversed)
        localized_entity_flattened_list = list(set(list(localized_word_mappings.values()) + list(localized_word_mappings.keys())))
        return localized_entity_flattened_list

    def _get_entity_type(self, ner_tags: List[Dict[str, str]], entity: str, localized_word_mappings: Dict[str, str]) -> Optional[str]:
        entity_type = None
        for ner_tag_item in ner_tags:
            if entity in ner_tag_item:
                entity_type = ner_tag_item[entity]
                break
            elif localized_word_mappings.get(entity) in ner_tag_item:
                entity_type = ner_tag_item[localized_word_mappings.get(entity)]
                break
        return entity_type

    def _get_entity_type_labels(self, entity_type: Optional[str]) -> Tuple[int, int]:
        if not entity_type:
            return 0,0
        entity_label_tuple = self.config.entity_labels.get(entity_type)
        if not entity_label_tuple:
            raise ValueError(f"Entity type {entity_type} not found in entity_labels mapping.")
        start_label, end_label = entity_label_tuple
        return start_label, end_label
    
    def _label_entities(self, entity: str, text_tokens: List[str], labels: List[int], start_label: int, end_label: int) -> List[int]:
        entity_tokens = self.tokenizer.tokenize(entity)

        for i in range(len(text_tokens) - len(entity_tokens) + 1):
            if text_tokens[i:i+len(entity_tokens)] == entity_tokens:
                for j in range(len(entity_tokens)):
                    if j == 0:
                        labels[i+j] = start_label
                    else:
                        labels[i+j] = end_label
        return labels