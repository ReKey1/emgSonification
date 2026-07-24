"""Feature / "quality categorizer" framework.

    >>> This is scaffolding. The actual signal categorizations are intentionally
    >>> NOT implemented here — only the structure that makes adding them a
    >>> few-line job. See "HOW TO ADD A CATEGORIZER" below.

Every extractor is fed each ProcessedSample as it streams by, maintains
whatever state it needs, and can be asked for its current result at any time.
The pipeline owns a FeatureBank that fans samples out to all active extractors
and collects their results for the UI, the recorder, or the sonifier.

--------------------------------------------------------------------------
HOW TO ADD A CATEGORIZER
--------------------------------------------------------------------------
1. Subclass SlidingWindowExtractor (gives you a ready ring buffer) or
   FeatureExtractor (full manual control).
2. Set a unique `name` and, for the windowed base, the `field` you want
   ("raw" | "filtered" | "envelope") and `window_s`.
3. Implement `compute()` -> FeatureResult using `self.window()`.
4. Decorate the class with `@register_feature` so it can be enabled by name
   via Config.enabled_features, or just add an instance to a FeatureBank.

That's it — no pipeline changes needed. A worked template
(`WindowMeanAbs`, registered as "example") is included to show the mechanics;
delete it once you have real extractors.

--------------------------------------------------------------------------
RELATIONSHIP TO OFFLINE SCORING
--------------------------------------------------------------------------
`emg/scoring.py` is authoritative for the study's numbers. It implements the five
features of llm_director_extension.md §10.3 (T1 rate of EMG rise, T2 offset rate,
T3 active fraction, A1 peak amplitude, S1 median frequency) over finished
recordings, where the rep boundaries are known and the whole signal is available.

Nothing here duplicates that, and nothing here should. These stubs are the *live*
path, and live extraction is a strictly harder problem: T1/T2 need a peak that has
not happened yet, T3 needs a window that has not closed, and A1 needs the session
MVC. They stay unimplemented on purpose — which quality is worth sonifying in real
time is an open thesis question, and answering it early would bake in an answer the
experiment is supposed to produce. The stubs below are named for the §10.3 features
so the two modules at least speak one vocabulary.
"""

from __future__ import annotations

import abc
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Type

import numpy as np

from .config import Config
from .types import FeatureResult, ProcessedSample


# --------------------------------------------------------------------------- #
#  Base classes
# --------------------------------------------------------------------------- #
class FeatureExtractor(abc.ABC):
    """Derives one "quality" from the streaming signal.

    Lifecycle:  reset() -> update(sample) x many -> compute() -> (repeat).
    `compute()` may be called at any time and must never raise; return a
    FeatureResult with value=None until enough data has accumulated.
    """

    name: str = "unnamed"

    @abc.abstractmethod
    def update(self, sample: ProcessedSample) -> None: ...

    @abc.abstractmethod
    def compute(self) -> FeatureResult: ...

    def reset(self) -> None:  # override if you hold state
        pass


class SlidingWindowExtractor(FeatureExtractor):
    """Convenience base: keeps the last `window_s` seconds of one field.

    Subclasses set `field` and `window_s`, then read `self.window()` (a numpy
    array, oldest-first) inside `compute()`.
    """

    field: str = "filtered"     # "raw" | "filtered" | "envelope"
    window_s: float = 1.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        n = max(1, int(round(self.window_s * cfg.sample_rate)))
        self._buf: Deque[float] = deque(maxlen=n)

    def update(self, sample: ProcessedSample) -> None:
        self._buf.append(getattr(sample, self.field))

    def window(self) -> np.ndarray:
        return np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))

    def full(self) -> bool:
        return self._buf.maxlen is not None and len(self._buf) >= self._buf.maxlen

    def reset(self) -> None:
        self._buf.clear()


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Type[FeatureExtractor]] = {}


def register_feature(cls: Type[FeatureExtractor]) -> Type[FeatureExtractor]:
    """Class decorator: make an extractor available by its `name`."""
    key = cls.name
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"feature name already registered: {key!r}")
    _REGISTRY[key] = cls
    return cls


def available_features() -> List[str]:
    return sorted(_REGISTRY)


def create_feature(name: str, cfg: Config) -> FeatureExtractor:
    if name not in _REGISTRY:
        raise KeyError(f"unknown feature {name!r}; available: {available_features()}")
    return _REGISTRY[name](cfg)


# --------------------------------------------------------------------------- #
#  Bank: runs a set of extractors over the stream
# --------------------------------------------------------------------------- #
class FeatureBank:
    """Fans each sample out to all extractors and collects their results."""

    def __init__(self, extractors: Optional[List[FeatureExtractor]] = None):
        self.extractors: List[FeatureExtractor] = list(extractors or [])

    def add(self, extractor: FeatureExtractor) -> None:
        self.extractors.append(extractor)

    def update(self, sample: ProcessedSample) -> None:
        for e in self.extractors:
            e.update(sample)

    def compute(self) -> Dict[str, FeatureResult]:
        return {e.name: e.compute() for e in self.extractors}

    def reset(self) -> None:
        for e in self.extractors:
            e.reset()

    def names(self) -> List[str]:
        return [e.name for e in self.extractors]


def build_feature_bank(cfg: Config) -> FeatureBank:
    """Instantiate the extractors named in cfg.enabled_features."""
    return FeatureBank([create_feature(n, cfg) for n in cfg.enabled_features])


# --------------------------------------------------------------------------- #
#  EXAMPLE — plumbing demo only (delete once you have real extractors).
#  It computes mean(|window|): enough to prove the framework end-to-end
#  without standing in for any of the real qualities below.
# --------------------------------------------------------------------------- #
@register_feature
class WindowMeanAbs(SlidingWindowExtractor):
    name = "example"
    field = "filtered"
    window_s = 0.5

    def compute(self) -> FeatureResult:
        if len(self._buf) == 0:
            return FeatureResult(self.name, None)
        w = self.window()
        return FeatureResult(self.name, float(np.mean(np.abs(w))),
                             meta={"n": int(w.size)})


# --------------------------------------------------------------------------- #
#  STUBS — live counterparts of the §10.3 feature set.
#  Each is a ready-to-fill slot: set field/window_s if needed and implement
#  compute(). They are NOT registered (so they never run until you finish
#  them); register with @register_feature when ready. Delete any you don't want.
#  See emg/scoring.py for the offline definition of each, which is authoritative.
# --------------------------------------------------------------------------- #
class RateOfRise(SlidingWindowExtractor):
    """T1 — rate of EMG rise: peak of dE/dt over the rising phase.

    Offline this is max(dE/dt) from cue to peak. Live, "the rising phase" is not
    yet delimited, so a windowed running max of the positive derivative is the
    natural approximation — decide and document how it relates to T1 before use.
    """
    name = "t1_rate_of_rise"
    field = "envelope"
    window_s = 0.5

    def compute(self) -> FeatureResult:
        raise NotImplementedError("categorizer stub — implement windowed max dE/dt")


class OffsetRate(SlidingWindowExtractor):
    """T2 — offset rate: peak of -dE/dt as the muscle deactivates.

    Symmetric to T1 and subject to the same caveat: the falling phase is only
    identifiable after the peak has passed, so a live version necessarily lags.
    """
    name = "t2_offset_rate"
    field = "envelope"
    window_s = 0.5

    def compute(self) -> FeatureResult:
        raise NotImplementedError("categorizer stub — implement windowed max -dE/dt")


class BurstDuration(SlidingWindowExtractor):
    """T3 — burst duration (onset→offset) over the movement window.

    Needs a resting baseline to threshold against and a detected offset, neither of
    which is available live until a rest period and a deactivation have been observed.
    """
    name = "t3_burst_frac"
    field = "envelope"
    window_s = 1.0

    def compute(self) -> FeatureResult:
        raise NotImplementedError("categorizer stub — implement burst duration here")


class PeakAmplitude(SlidingWindowExtractor):
    """A1 — peak envelope as %MVC.

    Requires the session MVC reference, so it cannot be computed from the stream
    alone; the pipeline must supply the normaliser.
    """
    name = "a1_peak_pct_mvc"
    field = "envelope"
    window_s = 1.0

    def compute(self) -> FeatureResult:
        raise NotImplementedError("categorizer stub — needs a session MVC reference")


class MedianFrequency(SlidingWindowExtractor):
    """S1 — median power frequency.

    Computed on the *filtered* signal, never the envelope: rectifying and
    smoothing destroys the spectrum this measures.
    """
    name = "s1_mdf_hz"
    field = "filtered"
    window_s = 0.25

    def compute(self) -> FeatureResult:
        raise NotImplementedError("categorizer stub — implement MDF on the PSD")


# Two-channel features. Out of reach on this rig, which is single-channel
# (triceps long head): the antagonist brake of the triphasic pattern is simply
# not observable. Kept as named slots for the planned 2-channel follow-up.
class CoContractionRatio(FeatureExtractor):
    """Antagonist/agonist activation ratio. Needs a second EMG channel."""
    name = "co_contraction"

    def update(self, sample: ProcessedSample) -> None:
        raise NotImplementedError("categorizer stub — requires multi-channel input")

    def compute(self) -> FeatureResult:
        raise NotImplementedError("categorizer stub — implement antagonist/agonist ratio")


class RecruitmentSpecificity(FeatureExtractor):
    """Target-muscle activation / total activation across channels."""
    name = "recruitment_specificity"

    def update(self, sample: ProcessedSample) -> None:
        raise NotImplementedError("categorizer stub — requires multi-channel input")

    def compute(self) -> FeatureResult:
        raise NotImplementedError("categorizer stub — implement target/total ratio")
