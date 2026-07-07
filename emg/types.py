"""Small data records passed between pipeline stages.

Kept in their own module so every stage (source, filter, features, recorder)
can import them without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Sample:
    """One raw sample as it arrives from the acquisition source.

    `t` is seconds since the source started, derived from the (fixed) sample
    rate rather than wall-clock, so spacing is uniform for offline analysis.
    The firmware-side columns are optional: a serial line may carry only `raw`.
    """

    t: float
    raw: float
    fw_filtered: Optional[float] = None   # firmware band-pass (reference)
    fw_envelope: Optional[float] = None   # firmware envelope (reference)
    detect: Optional[float] = None        # wear/contact flag, 0/1


@dataclass(slots=True)
class ProcessedSample:
    """A sample after the host filter chain has run.

    This is the record every downstream sink receives (plot, recorder,
    feature extractors, sonifier).
    """

    t: float
    raw: float
    filtered: float          # host high-pass + notch + low-pass
    envelope: float          # host smoothed magnitude (drives sonification)
    detect: Optional[float]  # passthrough wear/contact flag
    contact_ok: bool         # interpreted contact state


@dataclass(slots=True)
class FeatureResult:
    """Output of a FeatureExtractor.

    `value` is None until the extractor has enough data to produce a result.
    `meta` carries anything extra a categorizer wants to expose (e.g. a label,
    sub-scores, confidence) without changing this record.
    """

    name: str
    value: Optional[float] = None
    meta: dict = field(default_factory=dict)
