import asyncio
from typing import Any, Dict, Optional, Union, Tuple, List
from .local_sync import NERLocalizer
from .local_async import NERLocalizerAsync
from .schemas import NERLocalizerConfig
from .helpers import assign_entity_labels, validate_localize_ner_input_examples, validate_entity_labels
from .errors import NERValidationError
from ...components.factory import create_llm_object
from ...utility.config import DEFAULT_VENDORS


def localize_ner_data(
        examples: List[Dict[str, Any]],
        language: str,
        vendor: str = "openai",
        model: str = "gpt-4o-mini",
        model_params: Optional[Dict] = None,
        llm: Optional[Any] = None,
        batch_size: int = 10,
        tokenizer: Optional[Union[object, str]] = None,
        entity_list: Optional[List[str]] = None,
        entity_labels: Optional[Dict[str, Tuple[int, int]]] = None,
        export_type: str = "default",
        verbose: bool = False,
        **kwargs
        ):
    
    validate_localize_ner_input_examples(examples)

    if entity_labels:
        validate_entity_labels(entity_labels)
    else:
        if not entity_list:
            raise NERValidationError("Either entity_list or entity_labels must be provided.")
        entity_labels = assign_entity_labels(entity_list)

    if not model_params:
        model_params = {}

    if not llm:
        llm = create_llm_object(vendor=vendor, model=model, **model_params)
    
    if not tokenizer:
        print("No tokenizer provided, using default 'bert-base-uncased'")
        tokenizer = "bert-base-uncased"

    config = NERLocalizerConfig(
        target_language=language,
        batch_size=batch_size,
        tokenizer=tokenizer,
        entity_list=entity_list,
        entity_labels=entity_labels,
        llm=llm,
        export_type=export_type,
        verbose=verbose,
    )

    service = NERLocalizer(config=config)
    results = service.generate(examples=examples)

    return results
        

async def localize_ner_data_async(
        examples: List[Dict[str, Any]],
        language: str,
        vendor: str = "openai",
        model: str = "gpt-4o-mini",
        model_params: Optional[Dict] = None,
        llm: Optional[Any] = None,
        batch_size: int = 10,
        tokenizer: Optional[Union[object, str]] = None,
        entity_list: Optional[List[str]] = None,
        entity_labels: Optional[Dict[str, Tuple[int, int]]] = None,
        export_type: str = "default",
        verbose: bool = False,
        **kwargs
        ):
    
    validate_localize_ner_input_examples(examples)

    if entity_labels:
        validate_entity_labels(entity_labels)
    else:
        if not entity_list:
            raise NERValidationError("Either entity_list or entity_labels must be provided.")
        entity_labels = assign_entity_labels(entity_list)

    if not model_params:
        model_params = {}

    if not llm:
        llm = create_llm_object(vendor=vendor, model=model, **model_params)
    
    if not tokenizer:
        print("No tokenizer provided, using default 'bert-base-uncased'")
        tokenizer = "bert-base-uncased"

    config = NERLocalizerConfig(
        target_language=language,
        batch_size=batch_size,
        tokenizer=tokenizer,
        entity_list=entity_list,
        entity_labels=entity_labels,
        llm=llm,
        export_type=export_type,
        verbose=verbose,
    )

    service = NERLocalizerAsync(config=config)
    results = await service.generate(examples=examples)

    return results

async def localize_ner_data_multi_vendor_async(
        examples: List[Dict[str, Any]],
        language: str,
        vendors: Optional[Dict[str, str]] = None,
        batch_size: int = 10,
        tokenizer: Optional[Union[object, str]] = None,
        entity_list: Optional[List[str]] = None,
        entity_labels: Optional[Dict[str, Tuple[int, int]]] = None,
        export_type: str = "default",
        verbose: bool = False,
        **kwargs
    ):
    if not vendors:
        vendors = DEFAULT_VENDORS

    # Split examples equally among vendors
    num_vendors = len(vendors)
    examples_per_vendor = len(examples) // num_vendors
    remainder = len(examples) % num_vendors
    
    vendor_examples = []
    start_idx = 0
    
    for i, (vendor, model) in enumerate(vendors.items()):
        # Distribute remainder examples to first vendors
        end_idx = start_idx + examples_per_vendor + (1 if i < remainder else 0)
        vendor_examples.append((vendor, model, examples[start_idx:end_idx]))
        start_idx = end_idx

    tasks = [
        asyncio.create_task(
            localize_ner_data_async(
                examples=vendor_data,
                language=language,
                vendor=vendor,
                model=model,
                batch_size=batch_size,
                tokenizer=tokenizer,
                entity_list=entity_list,
                entity_labels=entity_labels,
                export_type=export_type,
                verbose=verbose,
                **kwargs
            )
        ) 
        for vendor, model, vendor_data in vendor_examples
    ]

    results_list = await asyncio.gather(*tasks)

    return dict(zip(vendors.keys(), results_list))
