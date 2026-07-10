"""cheezEMG host-side signal pipeline.

Data flow:

    Source (serial | synthetic)
        -> EmgFilterChain   (high-pass + mains notch + low-pass + envelope)
        -> ProcessedSample
        -> sinks:  plot buffers | Recorder | FeatureBank | Sonifier

Public building blocks are re-exported here for convenience. See the module
docstrings for details, and `emg/features.py` for how to add a new signal
"quality" (categorizer) without touching the pipeline.
"""

from .config import Config
from .types import Sample, ProcessedSample, FeatureResult
from .streaming import EmgFilterChain, StreamingFilter, build_chain
from .source import Source, SerialSource, SyntheticSource, open_source
from .features import (
    FeatureExtractor,
    SlidingWindowExtractor,
    FeatureBank,
    register_feature,
    build_feature_bank,
)
from .recorder import Recorder
from .sonify import Sonifier
from .pipeline import Pipeline

__all__ = [
    "Config",
    "Sample",
    "ProcessedSample",
    "FeatureResult",
    "EmgFilterChain",
    "StreamingFilter",
    "build_chain",
    "Source",
    "SerialSource",
    "SyntheticSource",
    "open_source",
    "FeatureExtractor",
    "SlidingWindowExtractor",
    "FeatureBank",
    "register_feature",
    "build_feature_bank",
    "Recorder",
    "Sonifier",
    "Pipeline",
]
