import asyncio
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
from .schemas import NERLocalizerConfig, NERLocalText, NERLocalResponse, NerOutput
from ..base import NlpTask
from ...components.standard_chain_builder import StandardChainBuilder


class NERLocalizerAsync(NlpTask):

    def __init__(self, config: NERLocalizerConfig):
        self.config = config
        self.tokenizer = None
        self._tokenizer_loaded = False

    async def generate(self, examples: List[Dict[str, Any]]) -> NerOutput:
        # Ensure tokenizer is loaded before processing
        await self._ensure_tokenizer_loaded()
        
        batches = await self._compose_batches(examples)
        generated_text_results = await self._generate_text(batches)
        generated_text_results = await self._label_generated_text_tokens(examples, generated_text_results)
        return await self._convert_to_output_async(generated_text_results, NERLocalResponse)
    
    async def _ensure_tokenizer_loaded(self):
        """Ensure tokenizer is loaded before use."""
        if not self._tokenizer_loaded:
            await self._load_tokenizer()
            self._tokenizer_loaded = True
    
    async def _load_tokenizer(self):
        """Load tokenizer asynchronously."""
        if isinstance(self.config.tokenizer, str):
            # Run the blocking call in a thread pool
            loop = asyncio.get_event_loop()
            self.tokenizer = await loop.run_in_executor(
                None, 
                AutoTokenizer.from_pretrained, 
                self.config.tokenizer
            )
        else:
            self.tokenizer = self.config.tokenizer
    
    async def _compose_batches(self, examples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
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
    
    async def _generate_text(self, batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        chain = StandardChainBuilder(
            prompt_template=self.config.prompt,
            llm=self.config.llm,
            entity_model=NERLocalText,
        ).build_chain()

        results = []
        total_batches = len(batches)
        if self.config.verbose:
            print(f"[{self.config.model}] Starting text generation: {total_batches} batches", flush=True)
        
        for i, batch in enumerate(batches, 1):
            responses = await chain.abatch(batch)
            responses = [x.model_dump() for x in responses]
            results.extend(responses)
            if self.config.verbose:
                print(f"[{self.config.model}] Generated batch {i}/{total_batches}", flush=True)
        
        if self.config.verbose:
            print(f"[{self.config.model}] Text generation complete", flush=True)
        return results
    
    async def _label_generated_text_tokens(self, examples: List[Dict[str, Any]], localized_texts:  List[Dict[str, Any]]):
        """Label generated text tokens asynchronously."""
        loop = asyncio.get_event_loop()
        
        async def process_record(record):
            localized_text = record["localized_text"]
            index = record["index"]
            example = examples[index]
            ner_tags = example["ner_tags"]
            localized_word_mappings = record["localized_word_mappings"]

            # Run tokenization in executor since it's CPU-bound
            text_tokens = await loop.run_in_executor(
                None,
                self.tokenizer.tokenize,
                localized_text
            )
            labels = [0] * len(text_tokens)

            localized_entity_flattened_list = await self._flatten_ner_tags(localized_word_mappings)

            for entity in localized_entity_flattened_list:
                entity_type = await self._get_entity_type(ner_tags, entity, localized_word_mappings)
                start_label, end_label = await self._get_entity_type_labels(entity_type)
                labels = await self._label_entities(entity, text_tokens, labels, start_label, end_label)
            
            record["tokens"] = text_tokens
            record["ner_tags"] = labels
            record["ner_tag_labels"] = self.config.entity_labels
            return record
        
        # Process all records concurrently
        tasks = [process_record(record) for record in localized_texts]
        total_tasks = len(tasks)
        if self.config.verbose:
            print(f"[{self.config.model}] Starting entity labeling: {total_tasks} records", flush=True)
        
        results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if self.config.verbose and (completed % 50 == 0 or completed == total_tasks):
                print(f"[{self.config.model}] Labeled {completed}/{total_tasks} records", flush=True)
        
        if self.config.verbose:
            print(f"[{self.config.model}] Entity labeling complete", flush=True)
        return results

    async def _flatten_ner_tags(self, localized_word_mappings: Dict[str, str]) -> List[str]:
        """Flatten NER tags asynchronously."""
        localized_word_mappings_reversed = {v: k for k, v in localized_word_mappings.items()}
        localized_word_mappings.update(localized_word_mappings_reversed)
        localized_entity_flattened_list = list(set(list(localized_word_mappings.values()) + list(localized_word_mappings.keys())))
        return localized_entity_flattened_list

    async def _get_entity_type(self, ner_tags: List[Dict[str, str]], entity: str, localized_word_mappings: Dict[str, str]) -> Optional[str]:
        """Get entity type asynchronously."""
        entity_type = None
        for ner_tag_item in ner_tags:
            if entity in ner_tag_item:
                entity_type = ner_tag_item[entity]
                break
            elif localized_word_mappings.get(entity) in ner_tag_item:
                entity_type = ner_tag_item[localized_word_mappings.get(entity)]
                break
        return entity_type

    async def _get_entity_type_labels(self, entity_type: Optional[str]) -> Tuple[int, int]:
        """Get entity type labels asynchronously."""
        if not entity_type:
            return 0,0
        entity_label_tuple = self.config.entity_labels.get(entity_type)
        if not entity_label_tuple:
            raise ValueError(f"Entity type {entity_type} not found in entity_labels mapping.")
        start_label, end_label = entity_label_tuple
        return start_label, end_label
    
    async def _label_entities(self, entity: str, text_tokens: List[str], labels: List[int], start_label: int, end_label: int) -> List[int]:
        """Label entities in tokens asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run tokenization in executor since it's CPU-bound
        entity_tokens = await loop.run_in_executor(
            None,
            self.tokenizer.tokenize,
            entity
        )

        for i in range(len(text_tokens) - len(entity_tokens) + 1):
            if text_tokens[i:i+len(entity_tokens)] == entity_tokens:
                for j in range(len(entity_tokens)):
                    if j == 0:
                        labels[i+j] = start_label
                    else:
                        labels[i+j] = end_label
        return labels