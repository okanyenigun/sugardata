import warnings
warnings.filterwarnings("ignore", message=".*Series.__getitem__ treating keys as positions is deprecated.*")

from .tasks.sentiment.service import generate_sentiment_data, agenerate_sentiment_data

__all__ = [
    "generate_sentiment_data",
    "agenerate_sentiment_data"
]

__version__ = '0.0.4'