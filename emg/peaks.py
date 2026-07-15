"""Track the strongest burst's peaks, keeping raw and filtered co-located.

`peak_filtered` is the largest |filtered| excursion (the real EMG event). `peak_raw`
is the largest |raw| within +/- `window_s` of that filtered peak — so both describe
the *same* muscle burst, never a stray raw spike (mains / motion) the band-pass would
remove. The band-pass's group delay puts the raw burst a hair before the filtered
peak, so the symmetric window catches it.

Shared by the Recorder (writes the peaks into session.json) and the Pipeline (feeds
the live peak readout in the UI), so the number you watch equals the number you save.
"""

from __future__ import annotations

from collections import deque

PEAK_WINDOW_S = 0.05  # half-window tying the raw peak to the filtered peak


class PeakTracker:
    def __init__(self, sample_rate: float, window_s: float = PEAK_WINDOW_S):
        self._half = max(1, int(round(window_s * sample_rate)))
        self.peak_filtered = 0.0
        self.peak_filtered_t = 0.0
        self.peak_raw = 0.0
        self._recent: "deque[float]" = deque(maxlen=self._half + 1)
        self._forward = 0

    def reset(self) -> None:
        self.peak_filtered = 0.0
        self.peak_filtered_t = 0.0
        self.peak_raw = 0.0
        self._recent.clear()
        self._forward = 0

    def update(self, raw: float, filtered: float, t: float) -> None:
        araw = abs(raw)
        self._recent.append(araw)                 # backward half of the window
        if self._forward > 0:                     # still inside a filtered peak's window
            if araw > self.peak_raw:
                self.peak_raw = araw
            self._forward -= 1
        if abs(filtered) > self.peak_filtered:
            self.peak_filtered = abs(filtered)
            self.peak_filtered_t = t
            self.peak_raw = max(self._recent)     # backward half
            self._forward = self._half            # then fold in the forward half
