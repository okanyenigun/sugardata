import pandas as pd
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union, Tuple
from .prompts import get_ner_localization_prompt


class NERLocalizerConfig(BaseModel):
    target_language: str = Field(description="Target language for localization", examples=["French", "Spanish", "German"])
    batch_size: int = Field(default=10, description="Number of sentences to generate in each batch")
    tokenizer: Union[object, str] = Field(description="Tokenizer used for text splitting and encoding, can be AutoTokenizer object or a string identifier")
    prompt: str = Field(default=get_ner_localization_prompt(), description="Prompt for the localization task")
    entity_list: Optional[List[str]] = Field(default=None, description="List of entity names to be recognized", examples=[["Person", "Organization", "Location"]])
    entity_labels: Optional[Dict[str, Tuple[int, int]]] = Field(default=None, description="Mapping of entity names to their label ranges", examples=[{"PER": (1, 2), "ORG": (3, 4), "LOC": (5, 6)}])
    llm: object = Field(description="LLM object used for generating text, e.g., OpenAI's GPT model")
    model: Optional[str] = Field(default=None, description="Model name or identifier for the LLM being used", examples=["gpt-3.5-turbo", "gemma3:12b"])
    export_type: str = Field(default="default", description="Output format of the generated data, e.g., 'dataframe' or 'dataset'")
    verbose: bool = Field(default=False, description="Flag to enable verbose logging during processing")

    def model_post_init(self, __context):
        """Automatically assign model name from llm object after initialization."""
        if self.model is None:
            if hasattr(self.llm, 'model_name'):
                self.model = self.llm.model_name
            elif hasattr(self.llm, 'model'):
                self.model = self.llm.model
            else:
                self.model = "unknown model"


class NERLocalText(BaseModel):
    index: int = Field(description="Index of the given request")
    localized_text: str = Field(description="Localized text in the target language")
    localized_word_mappings: Dict[str, str] = Field(description="The original word and the localized word mappings in the target language. These should be just like they used in the texts. Don't lemmatize or lowercase them. Just use the exact words.", examples=[{"John": "Jean", "Acme Corp": "Acme Société"}])


class NERLocalResponse(BaseModel):
    index: int = Field(description="Index of the given request")
    localized_text: str = Field(description="Localized text in the target language")
    localized_word_mappings: Dict[str, str] = Field(description="The original word and the localized word mappings in the target language.")
    tokens: List[str] = Field(description="List of tokens in the localized text")
    ner_tags: List[int] = Field(description="List of labels corresponding to the tokens")
    ner_tag_labels: Dict[str, Tuple[int, int]] = Field(description="Mapping of entity to its label")


NerOutput = Union[
    List[Dict],
    pd.DataFrame,
    Dataset,
    List[NERLocalResponse]
]
