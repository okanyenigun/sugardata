import pandas as pd
from datasets import Dataset
from typing import Optional, Dict, List, Union
from .schemas import SentimentConfig
from .generator import SentimentGenerator
from .augment import SentimentAugmenter
from .prompts import get_dimension_prompt, get_aspect_prompt, get_sentence_prompt, get_structure_prompt
from ...components.factory import create_llm_object
from ...utility.translate import TranslationUtility

def _build_sentiment_config(
        concept=None, language=None, vendor="openai", model="gpt-4o-mini",
        model_params=None, n_aspect=1, n_sentence=100, batch_size=10,
        label_options=["positive", "negative"], export_type="default",
        dimensions=None, aspects=None, examples=None, **kwargs
    ):
    # Detect language
    if not language:
        if concept:
            language = TranslationUtility.detect_language(concept)
        if examples:
            language = TranslationUtility.detect_language(examples[0])

    # Default model params
    if not model_params:
        model_params = {"temperature": 0.95}
    if "temperature" not in model_params:
        model_params["temperature"] = 0.95

    llm = create_llm_object(vendor=vendor, model=model, **model_params)

    config = SentimentConfig(
        language=language,
        dimension_prompt=get_dimension_prompt(language=language),
        aspect_prompt=get_aspect_prompt(language=language),
        sentence_prompt=get_sentence_prompt(language=language),
        structure_prompt=get_structure_prompt(language=language),
        llm=llm,
        n_aspect=n_aspect,
        n_sentence=n_sentence,
        batch_size=batch_size,
        label_options=label_options,
        export_type=export_type,
    )
    return config, dimensions, aspects, examples


def generate_sentiment_data(
        concept: Optional[str]=None,
        language: Optional[str]=None,
        vendor: str="openai",
        model: str="gpt-4o-mini",
        model_params: Optional[Dict]=None,
        n_aspect: int=1,
        n_sentence: int=100,
        batch_size: int=10,
        label_options: Optional[List]=["positive", "negative"],
        export_type: str="default",
        dimensions: Optional[List[str]]=None,
        aspects: Optional[List[str]]=None,
        examples: Optional[List[str]]=None,
        **kwargs
    ) -> Union[Dict, pd.DataFrame, Dataset]:

    config, dimensions, aspects, examples = _build_sentiment_config(
        concept=concept, language=language, vendor=vendor, model=model,
        model_params=model_params, n_aspect=n_aspect, n_sentence=n_sentence,
        batch_size=batch_size, label_options=label_options, export_type=export_type,
        dimensions=dimensions, aspects=aspects, examples=examples, **kwargs
    )

    if examples:
        return SentimentAugmenter(config=config).generate(examples=examples)
    else:
        return SentimentGenerator(config=config).generate(concept=concept, dimensions=dimensions, aspects=aspects)


async def agenerate_sentiment_data(
        concept: Optional[str]=None,
        language: Optional[str]=None,
        vendor: str="openai",
        model: str="gpt-4o-mini",
        model_params: Optional[Dict]=None,
        n_aspect: int=1,
        n_sentence: int=100,
        batch_size: int=10,
        label_options: Optional[List]=["positive", "negative"],
        export_type: str="default",
        dimensions: Optional[List[str]]=None,
        aspects: Optional[List[str]]=None,
        examples: Optional[List[str]]=None,
        **kwargs
    ) -> Union[Dict, pd.DataFrame, Dataset]:

    config, dimensions, aspects, examples = _build_sentiment_config(
        concept=concept, language=language, vendor=vendor, model=model,
        model_params=model_params, n_aspect=n_aspect, n_sentence=n_sentence,
        batch_size=batch_size, label_options=label_options, export_type=export_type,
        dimensions=dimensions, aspects=aspects, examples=examples, **kwargs
    )

    if examples:
        return await SentimentAugmenter(config=config).agenerate(examples=examples)
    else:
        return await SentimentGenerator(config=config).agenerate(concept=concept, dimensions=dimensions, aspects=aspects)
