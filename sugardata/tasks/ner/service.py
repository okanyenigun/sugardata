import asyncio
from typing import Optional, Dict, List, Any
from .localize_sync import NERLocalizer
from .localize_async import NERLocalizerAsync
from .schemas import NERConfig, NerOutput
from .prompts import get_ner_localization_prompt
from ...components.factory import create_llm_object
from ...utility.config import DEFAULT_VENDORS


def localize_ner_data(
        examples: List[Dict[str, Any]],
        target_language: str,
        vendor: str = "openai",
        model: str = "gpt-4o-mini",
        model_params: Optional[Dict] = None,
        batch_size: int = 10,
        export_type: str = "default",
        entity_label_mapping: Optional[Dict[str, Dict[str, int]]] = None,
        tokenizer: Optional[object] = None,
        verbose: bool = False,
        **kwargs
) -> NerOutput:
    if not model_params:
        model_params = {"temperature": 0.95}
    if "temperature" not in model_params:
        model_params["temperature"] = 0.95

    llm = create_llm_object(vendor=vendor, model=model, **model_params)

    config = NERConfig(
        localization_prompt=get_ner_localization_prompt(),
        llm=llm,
        batch_size=batch_size,
        export_type=export_type,
        verbose=verbose,
        target_language=target_language,
        entity_label_mapping=entity_label_mapping,
        tokenizer=tokenizer
    )

    return NERLocalizer(config=config).generate(examples=examples)


async def localize_ner_data_async(
        examples: List[Dict[str, Any]],
        target_language: str,
        vendor: str = "openai",
        model: str = "gpt-4o-mini",
        model_params: Optional[Dict] = None,
        batch_size: int = 10,
        export_type: str = "default",
        verbose: bool = False,
        **kwargs
) -> NerOutput:
    if not model_params:
        model_params = {"temperature": 0.95}
    if "temperature" not in model_params:
        model_params["temperature"] = 0.95

    llm = create_llm_object(vendor=vendor, model=model, **model_params)

    config = NERConfig(
        localization_prompt=get_ner_localization_prompt(),
        llm=llm,
        batch_size=batch_size,
        export_type=export_type,
        verbose=verbose
    )

    return await NERLocalizerAsync(config=config).generate(examples=examples, target_language=target_language)


async def localize_ner_multi_vendor_async(
        examples: List[Dict[str, Any]],
        target_language: str,
        vendors: Optional[Dict[str, str]] = None,
        batch_size: int = 10,
        export_type: str = "default",
        verbose: bool = False,
        **kwargs
) -> Dict[str, NerOutput]:
    if not vendors:
        vendors = DEFAULT_VENDORS

    tasks = [
        asyncio.create_task(
            localize_ner_data_async(
                examples=examples,
                target_language=target_language,
                vendor=vendor,
                model=model,
                batch_size=batch_size,
                export_type=export_type,
                verbose=verbose,
                **kwargs
            )
        )
        for vendor, model in vendors.items()
    ]

    results_list = await asyncio.gather(*tasks)

    return dict(zip(vendors.keys(), results_list))
