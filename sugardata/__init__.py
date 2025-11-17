from .tasks.sentiment.service import (
    augment_sentiment_data, augment_sentiment_data_async, augment_sentiment_multi_vendor_async,
    generate_sentiment_data, generate_sentiment_data_async, generate_sentiment_multi_vendor_async
)
from .tasks.ner.service import (
    localize_ner_data, localize_ner_data_async,localize_ner_data_multi_vendor_async
)

import warnings
warnings.filterwarnings(
    "ignore", message=".*Series.__getitem__ treating keys as positions is deprecated.*")


__all__ = [
    "augment_sentiment_data",
    "augment_sentiment_data_async",
    "augment_sentiment_multi_vendor_async",
    "generate_sentiment_data",
    "generate_sentiment_data_async",
    "generate_sentiment_multi_vendor_async",
    "localize_ner_data",
    "localize_ner_data_async",
    "localize_ner_data_multi_vendor_async"
]

__version__ = '0.0.5'
