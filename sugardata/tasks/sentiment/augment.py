from itertools import product
from typing import Dict, Any, List
from .schemas import Text, SentimentResponse, SentimentStructure, SentimentConfig, SentimentOutput
from ...components.standard_chain_builder import StandardChainBuilder
from ..base import NlpTask


class SentimentAugmenter(NlpTask):

    def __init__(self, config: SentimentConfig):
        self.config = config

    def generate(self, examples: List[str]) -> SentimentOutput:
        structure_list = self._extract_structures(examples=examples)
        batches = self._compose_batches(structure_list)
        sentences = self._generate_sentences(batches)
        parsed_rows = self._parse_sentences(sentences, batches)
        return self._convert_to_output(parsed_rows, SentimentResponse)
    
    def _extract_structures(self, examples: List[str]) -> List[Dict[str, Any]]:
        chain = StandardChainBuilder(
            prompt_template=self.config.structure_prompt,
            llm=self.config.llm,
            entity_model=SentimentStructure,   
        ).build_chain()

        batches = [{"text": example} for example in examples]

        results = []
        for i in range(0, len(batches), self.config.batch_size):
            batch = batches[i:i + self.config.batch_size]
            try:
                responses = chain.batch(batch)
            except Exception as e:
                print(f"Warning: Error processing batch {i//self.config.batch_size}: {e}. Continuing with next batch.")
                continue
            if not responses:
                raise ValueError("No responses received from the chain. Please check your configuration and input data.")
            for response in responses:
                response_dict = response.model_dump()
                results.append(response_dict)
        return results
    
    def _combine_aspects_for_structure(self, structure: Dict[str, Any]) -> List[str]:
        aspects = structure.get("aspects", [])
        label_combinations = list(product(self.config.label_options, repeat=len(aspects)))

        if not self.config.aspect_based_generation:
            # Keep only combinations where all labels are the same
            label_combinations = [
                combo for combo in label_combinations
                if all(x == combo[0] for x in combo)
            ]

        aspect_combinations = []
        for combo in label_combinations:
            s = " | ".join([f"Aspect: {aspect} -> Sentiment: {label}" for aspect, label in zip(aspects, combo)])
            aspect_combinations.append(s)

        return aspect_combinations
    
    def _compose_batches(self, structure_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batches = []
        counter = 0
        for structure in structure_list:
            aspect_combinations = self._combine_aspects_for_structure(structure)
            for aspect in aspect_combinations:
                batch = {
                    "index": counter,
                    "concept": structure.get("concept", ""),
                    "aspect": aspect,
                    "writing_style": structure.get("writing_style", ""),
                    "medium": structure.get("medium", ""),
                    "persona": structure.get("persona", ""),
                    "intention": structure.get("intention", ""),
                    "sentence_length": structure.get("sentence_length", ""),
                    "given_text": structure.get("given_text", "")
                }
                batches.append(batch)
                counter += 1
        return batches
    
    def _generate_sentences(self, batches: List[Dict[str, Any]]) -> List[Text]:
        chain = StandardChainBuilder(
            prompt_template=self.config.sentence_prompt,
            llm=self.config.llm,
            entity_model=Text
        ).build_chain()

        results = []
        for i in range(0, len(batches), self.config.batch_size):
            batch = batches[i:i + self.config.batch_size]
            try:
                responses = chain.batch(batch)
            except Exception as e:
                print(f"Warning: Error processing batch {i//self.config.batch_size}: {e}. Continuing with next batch.")
                continue
            results.extend(responses)

        return results
    
    def _parse_sentences(self, sentences: List[Text], batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for sentence in sentences:
            sentence_dict = sentence.model_dump()
            index = sentence_dict.get("index")
            batch = next((b for b in batches if b["index"] == index), None)
            if not batch:
                continue

            aspect_list = batch["aspect"].split(" | ")
            aspect_list = [x.strip() for x in aspect_list if x.strip()]

            text = sentence_dict.get("generated_text", "")

            for asp in aspect_list:
                aspect_parsed = self.parse_line(asp)
                row = {
                    "generated_text": text,
                    "label": aspect_parsed["sentiment"],
                    **batch
                }
                row["aspect"] = aspect_parsed["aspect"]

                results.append(row)

        if not self.config.aspect_based_generation:
            seen = set()
            unique_results = []

            for row in results:
                unique = f"{row['generated_text']}, {row['label']}"
                if unique not in seen:
                    unique_results.append(row)
                    seen.add(unique)
                    
        return unique_results

    async def agenerate(self, examples: List[str]) -> SentimentOutput:
        structure_list = await self._extract_structures_async(examples=examples)
        batches = await self._compose_batches_async(structure_list)
        sentences = await self._generate_sentences_async(batches)
        parsed_rows = await self._parse_sentences_async(sentences, batches)
        return await self._convert_to_output_async(parsed_rows, SentimentResponse)
    
    
    
    async def _extract_structures_async(self, examples: List[str]) -> List[Dict[str, Any]]:
        chain = StandardChainBuilder(
            prompt_template=self.config.structure_prompt,
            llm=self.config.llm,
            entity_model=SentimentStructure,
        ).build_chain()

        batches = [{"text": example} for example in examples]

        results = []
        for i in range(0, len(batches), self.config.batch_size):
            batch = batches[i:i + self.config.batch_size]
            try:
                responses = await chain.abatch(batch)  
            except Exception as e:
                print(f"Warning: Error processing batch {i//self.config.batch_size}: {e}. Continuing with next batch.")
                continue
            if not responses:
                raise ValueError("No responses received from the chain. Please check your configuration and input data.")
            for response in responses:
                response_dict = response.model_dump()
                results.append(response_dict)
        return results
    
    
    
    async def _combine_aspects_for_structure_async(self, structure: Dict[str, Any]) -> List[str]:
        aspects = structure.get("aspects", [])
        label_combinations = list(product(self.config.label_options, repeat=len(aspects)))

        aspect_combinations = []
        for combo in label_combinations:
            s = " | ".join([f"Aspect: {aspect} -> Sentiment: {label}" for aspect, label in zip(aspects, combo)])
            aspect_combinations.append(s)

        return aspect_combinations
    
    
    
    async def _compose_batches_async(self, structure_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batches = []
        counter = 0
        for structure in structure_list:
            aspect_combinations = await self._combine_aspects_for_structure_async(structure)
            for aspect in aspect_combinations:
                batch = {
                    "index": counter,
                    "concept": structure.get("concept", ""),
                    "aspect": aspect,
                    "writing_style": structure.get("writing_style", ""),
                    "medium": structure.get("medium", ""),
                    "persona": structure.get("persona", ""),
                    "intention": structure.get("intention", ""),
                    "sentence_length": structure.get("sentence_length", "")
                }
                batches.append(batch)
                counter += 1
        return batches
    
    
    
    async def _generate_sentences_async(self, batches: List[Dict[str, Any]]) -> List[Text]:
        chain = StandardChainBuilder(
            prompt_template=self.config.sentence_prompt,
            llm=self.config.llm,
            entity_model=Text
        ).build_chain()

        results = []
        for i in range(0, len(batches), self.config.batch_size):
            batch = batches[i:i + self.config.batch_size]
            try:
                responses = await chain.abatch(batch) 
            except Exception as e:
                print(f"Warning: Error processing batch {i//self.config.batch_size}: {e}. Continuing with next batch.")
                continue
            results.extend(responses)
        return results

    
    
    async def _parse_sentences_async(self, sentences: List[Text], batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for sentence in sentences:
            sentence_dict = sentence.model_dump()
            index = sentence_dict.get("index")
            batch = next((b for b in batches if b["index"] == index), None)
            if not batch:
                continue
            
            aspect_list = batch["aspect"].split(" | ")
            aspect_list = [x.strip() for x in aspect_list if x.strip()]

            text = sentence_dict.get("generated_text", "")

            for asp in aspect_list:
                aspect_parsed = self.parse_line(asp)
                row = {
                    "generated_text": text,
                    "label": aspect_parsed["sentiment"],
                    **batch
                }
                row["aspect"] = aspect_parsed["aspect"]
                results.append(row)

        return results
    
    @staticmethod
    def parse_line(line: str) -> Dict:
        parts = [p.strip() for p in line.split("->")]

        result = {}
        for part in parts:
            key, value = part.split(":", 1)
            result[key.strip()] = value.strip()
        
        return {
            "aspect":       result.get("Aspect"),
            "sentiment":    result.get("Sentiment"),
        }