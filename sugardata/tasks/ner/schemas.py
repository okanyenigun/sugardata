import pandas as pd
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union


class NERConfig(BaseModel):
    localization_prompt: Optional[str] = Field(
        default=None, description="Prompt for the localization task")
    llm: object = Field(
        description="LLM object used for generating text, e.g., OpenAI's GPT model")
    batch_size: int = Field(
        default=10, description="Number of sentences to generate in each batch")
    export_type: str = Field(
        default="default", description="Output format of the generated data, e.g., 'dataframe' or 'dataset'")
    verbose: bool = Field(
        default=False, description="Whether to print verbose output during processing")


class LocalNERText(BaseModel):
    index: int = Field(description="Index of the given request")
    localized_text: str = Field(
        description="Localized text in the target language")
    localized_word_mappings: dict = Field(
        description="The original word and the localized word mappings in the target language")


class NERResponse(BaseModel):
    index: int = Field(description="Index of the given request")
    text: str = Field(description="Localized text in the target language")
    tokens: List[str] = Field(
        description="List of tokens in the localized text")
    labels: List[int] = Field(
        description="List of labels corresponding to the tokens")
    label_mappings: List = Field(
        description="Mapping of entity to its label")


NerOutput = Union[
    List[Dict],
    pd.DataFrame,
    Dataset,
    List[NERResponse]
]
