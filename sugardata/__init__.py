import warnings
warnings.filterwarnings("ignore", message=".*Series.__getitem__ treating keys as positions is deprecated.*")

from .api import *

__version__ = '0.0.3'