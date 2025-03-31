import uuid
import json
import random
import pandas as pd
from langchain import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from datasets import Dataset
from typing import Any, Dict, Optional, List, Literal
from .schemas import OntologyConcepts, GeneratedText
from .prompts import CONCEPT_EXAMPLE_TEMPLATE, CONCEPT_EXAMPLES, CONCEPT_PREFIX, CONCEPT_SUFFIX, GENERATE_PROMPT
from ..base import NLPTask
from ...concepts import SENTIMENT_LABEL_MAPPING
from ...utility.translate import TranslationUtility
from ...utility.draw import DrawUtility


class SentimentGenerator(NLPTask):

    def __init__(
            self, 
            model_provider: str, 
            model_name: str, 
            model_kwargs: Optional[Dict[str, Any]]=None, 
            batch_size: int=16, 
            language: str="en",
            n_labels=2
            ):
        super().__init__(model_provider, model_name, model_kwargs, batch_size, language)

        self.labels = SENTIMENT_LABEL_MAPPING.get(n_labels)
        if not self.labels:
            raise ValueError(f"Invalid number of labels: {n_labels}. Must be one of {list(SENTIMENT_LABEL_MAPPING.keys())}.")

    def generate(
            self, 
            concept: str, 
            n_samples: int = 100, 
            output_format: Literal["pandas", "json", "dictionary", "hg"]="pandas"
            ) -> Any:
        
        run_id = str(uuid.uuid4())[-6:]

        concept_en = self._translate_concept(concept)
        
        aspects = self._generate_aspects(concept_en)
        
        batches = self._create_batches(concept, aspects, n_samples)
        
        sentences = self._generate_sentences(batches)
        
        sentences = self._add_additional_info(sentences, batches, run_id)
        
        sentences = self._translate_to_original(sentences)
        
        output = self._create_output(sentences, output_format)

        return output
    
    def _set_prompts(self) -> None:
        self._prompts = {
            "concept_examples": CONCEPT_EXAMPLES,
            "concept_example_template": CONCEPT_EXAMPLE_TEMPLATE,
            "concept_prefix": CONCEPT_PREFIX,
            "concept_suffix": CONCEPT_SUFFIX,
            "generate_prompt": GENERATE_PROMPT,
        }
        return

    def _translate_concept(self, concept: str) -> str:
        if self.language == "en":
            return concept

        return TranslationUtility.translate(concept, target_language="en", source_language=self.language)
 
    def _generate_aspects(self, concept: str) -> List[Dict[str, str]]:
        try:
            prompt = FewShotPromptTemplate(
                examples=self._prompts["concept_examples"],
                example_prompt=PromptTemplate(
                    input_variables=["concept", "generated_concept", "explanation"], 
                    template=self._prompts["concept_example_template"]
                ),
                prefix=self._prompts["concept_prefix"],
                suffix=self._prompts["concept_suffix"],
                input_variables=["concept"],
                example_separator="\n\n"
            )

            parser = PydanticOutputParser(pydantic_object=OntologyConcepts)
            format_instructions = parser.get_format_instructions()
            chain = prompt | self.llm | parser

            response = chain.invoke({
                "concept":concept,
                "format_instructions": format_instructions
                }
            )

            return [concept.model_dump() for concept in response.concepts]
        except Exception as e:
            print("Error in extending concept: ", e)
            raise e
        
    def _create_batches(self, concept: str, aspects: List[Dict[str, str]], n_samples: int) -> List[Dict[str, str]]:
        data = []
        
        for i in range(n_samples):
            aspects_record = random.choice(aspects)
            data.append({
                "id": i,
                "concept": concept,
                "extended_concept": aspects_record["generated_concept"],
                "explanation": aspects_record["explanation"],
                "sentiment_label": random.choice(self.labels),
                **DrawUtility.draw_style()
            })

        return data
    
    def _generate_sentences(self, batches: List[Dict[str, str]]) -> List[Dict[str, str]]:
        output = []

        prompt = PromptTemplate.from_template(self._prompts["generate_prompt"])

        parser = PydanticOutputParser(pydantic_object=GeneratedText)
        format_instructions = parser.get_format_instructions()

        chain = prompt | self.llm | parser

        for i in range(0, len(batches), self.batch_size):
            batch = batches[i:i+self.batch_size]
            
            for item in batch:
                item["format_instructions"] = format_instructions

            response = chain.batch(batch)
            response = [resp.model_dump() for resp in response]

            output.extend(response)

        return output
    
    def _add_additional_info(self, data: List[Dict[str, str]], batches: List[Dict[str, str]], run_id) -> List[Dict[str, str]]:
        batch_map = {batch["id"]: batch for batch in batches}

        for item in data:
            batch = batch_map.get(int(item["id"]))

            if batch:
                item["concept"] = batch["concept"]
                item["label"] = batch["sentiment_label"]
                item["extended_concept"] = batch["extended_concept"]
                item["writing_style"] = batch["writing_style"]
                item["medium"] = batch["medium"]
                item["persona"] = batch["persona"]
                item["intention"] = batch["intention"]
                item["tone"] = batch["tone"]
                item["audience"] = batch["audience"]
                item["context"] = batch["context"]
                item["language_register"] = batch["language_register"]
                item["run_id"] = run_id

        return data
    
    def _translate_to_original(self, sentences: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self.language == "en":
            return sentences
        
        for sentence in sentences:
            sentence["text"]= TranslationUtility.translate(
                sentence["text"], 
                target_language=self.language, 
                source_language="en"
            )

        return sentences
    
    def _create_output(self, sentences: List[Dict[str, str]], output_format: str) -> Any:
        if output_format == "pandas":
            return pd.DataFrame(sentences)
        elif output_format == "json":
            return json.dumps(sentences, indent=4)
        elif output_format == "dictionary":
            return sentences
        elif output_format == "hg":
            return Dataset.from_pandas(pd.DataFrame(sentences))
        else:
            raise ValueError(f"Invalid output format: {output_format}. Must be one of ['pandas', 'json', 'dictionary', 'hg'].")




        