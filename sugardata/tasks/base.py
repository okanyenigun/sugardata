from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal
from ..models.llms.factory import create_langchain_llm


class NLPTask(ABC):

    def __init__(
            self, 
            model_provider: str, 
            model_name: str, 
            model_kwargs: Optional[Dict[str, Any]]=None, 
            batch_size: int=16, 
            language: str="en", 
            **kwargs
            ):
        self.language = language
        self.batch_size = batch_size

        if model_kwargs is None:
            model_kwargs = {}

        self.llm = create_langchain_llm(provider=model_provider, model=model_name, **model_kwargs)
        
        self._set_prompts()

    @abstractmethod
    def generate(self, n_samples: int = 100, output_format: Literal["pandas", "json", "dictionary", "hg"]="pandas", **kwargs) -> Any:
        pass