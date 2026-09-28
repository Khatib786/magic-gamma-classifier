"""MAGIC Gamma Telescope: physics-informed gamma/hadron separation."""
from sklearn import set_config

# Modern Pipeline API: transformers return pandas DataFrames so feature
# names propagate through the whole Pipeline.
set_config(transform_output="pandas")

__version__ = "1.0.0"
