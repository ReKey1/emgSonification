# cheezEMG

Host software for the **cheezEMG** sensor — a dry-contact PCB EMG board (conductive
pads, not gel stickers). It cleanly filters the noisy incoming signal, records
labelled sessions to disk, sonifies muscle activation in real time, and provides
an extensible framework for categorizing signal "qualities".

Built for EMG-sonification / auditory-biofeedback motor-learning research.

---

## Why this exists

The board's dry PCB pads pick up far more **mains hum** and **baseline drift** than
gel electrodes, and the firmware's built-in band-pass has **no mains notch**. So the
firmware here streams the *raw* ADC signal and all real filtering happens on the host,
where the notch can be placed at the correct local power-line frequency and where
recordings can be re-processed offline.

> **Mains frequency (you're in Japan):** eastern Japan (Tokyo, Tohoku, Hokkaido) is
> **50 Hz**; western Japan (Osaka, Nagoya, Kyushu) is **60 Hz**. The default is 50 Hz.
> Switch it with the radio buttons in the UI, `--mains 60` on the CLI, or `mains_hz`
> in config. Getting this right is the single biggest noise win for this sensor.

---

## Architecture

```
Source (serial | synthetic)
   |- EmgFilterChain      high-pass 20 Hz -> mains notch(es) -> low-pass 200 Hz -> envelope
        |- ProcessedSample
             |- live plot        (UI)
             |- Recorder         (start/stop -> recordings/<title>/<timestamp>_<notes>/)
             |- FeatureBank       (categorizers — pluggable, see below)
             |- Sonifier         (envelope -> pitch + loudness)

recordings/  --(offline)-->  score_cli.py + emg/scoring.py  -->  scores.csv
```

| File | Role |
|------|------|
| `cheezEMG.ino`      | Firmware: streams `raw,filtered,envelope,detect` CSV at 500 Hz |
| `emg/config.py`     | All tunable settings (one dataclass, JSON-serialisable) |
| `emg/streaming.py`  | Streaming IIR filter chain (SOS, stateful, low-latency) |
| `emg/source.py`     | Serial source + synthetic generator (runs with no hardware) |
| `emg/features.py`   | **Categorizer framework + stubs** (live, add qualities here) |
| `emg/recorder.py`   | Thread-safe session recorder |
| `emg/sonify.py`     | Real-time audio synthesis |
| `emg/pipeline.py`   | Wires source -> filter -> sinks on a background thread |
| `emg/scoring.py`    | **Offline dataset scorer** — research EMG metrics per recording |
| `app.py`            | Tkinter UI |
| `record_cli.py`     | Headless recorder |
| `score_cli.py`      | Batch-score recordings into a clean `scores.csv` |

---

## Setup

```bash
pip install -r requirements.txt
```

Flash `cheezEMG.ino` to the board (needs the `CheezsEMG` Arduino library). Note the
serial port (e.g. `COM7`).

## Run

**UI:**
```bash
python app.py
```
Pick *Serial* + your port (or *Synthetic* to try it with no hardware), choose 50/60 Hz,
click **Connect**. You get a live plot (raw vs cleaned + envelope), a skin-contact
indicator, a **live peak readout** (strongest-burst raw & filtered, co-located), a sound
toggle, and **Start/Stop Recording**. With the *3·2·1 countdown tones* box ticked,
Start plays three spaced tones: recording begins on the **second** tone (so no pre-movement
data is lost) and the **third** tone is the "go" cue to start the movement. The go moment is
logged to `session.json` as `movement_onset_t` (matches the `t` column) / `movement_onset_s`,
so you can mark exactly where the movement began. Untick the box to start instantly. The gap
between tones is `COUNTDOWN_INTERVAL_MS` in `app.py` (default 1000 ms).

**Headless recording:** `--title` is the test subject (its directory); `--notes`
names the dataset inside it.
```bash
python record_cli.py --port COM7 --title alice --notes bicep-left --seconds 30
python record_cli.py --synthetic --seconds 15 --title subject-01 --notes warmup  # no hardware
python record_cli.py --port COM7 --mains 60           # western Japan
```

---

## Recording format

Recordings are grouped **one directory per test subject** (`--title`), with one
dataset per recording (`--notes`) inside it:

```
recordings/
   alice/                              <- title  (test subject)
      2026-07-07_143012_bicep-left/    <- notes  (this dataset)
         signal.csv     t, raw, filtered, envelope, detect, contact_ok   (per sample)
         features.csv   t, <feature columns>                             (optional, per window)
         session.json   config snapshot + title, notes, duration, sample count,
                        the strongest burst's peaks (peak_filtered, peak_raw, peak_t),
                        and the movement cue (movement_onset_t / movement_onset_s)
      2026-07-07_145533_bicep-right/
         ...
```
Both `raw` and host-`filtered` EMG are stored every sample, so a session is fully
self-contained — the scorer reads them directly and never has to re-derive. The
saved peaks describe the strongest contraction: `peak_filtered` is the largest
cleaned-signal excursion, `peak_raw` is the raw amplitude **at that same moment**
(within ±50 ms of `peak_t`), so the two always refer to the same muscle burst rather
than a stray raw spike the band-pass would remove.
Raw is stored alongside the filtered signal, so any session can be re-filtered offline
with different settings.

---

## Scoring recorded sessions

`score_cli.py` reads every recording and writes one tidy row per dataset to a CSV of
research-grounded single-channel EMG properties (implemented in `emg/scoring.py`; the
literature basis and citations are in the module docstring and `../research`):

```bash
python score_cli.py                 # score ./recordings -> recordings/scores.csv
python score_cli.py --refilter      # re-derive filtered/envelope from raw first
python score_cli.py --mains 60      # override mains freq for notch/quality metrics
```

| Column | Property | Basis |
|---|---|---|
| `rms_amplitude`, `mav` | Activation level (RMS, mean-abs-value) | standard EMG amplitude features |
| `baseline_noise`, `snr_db` | Rest noise floor + active-vs-rest SNR | required signal-quality gate |
| `mains_residual` | Power left at mains + harmonics after notching | dry-PCB hum check |
| `median_freq_hz` | Median power frequency | standard spectral descriptor |
| `n_reps`, `rise_time_ms`, `onset_sharpness` | Contraction count + onset rise time | sharper onset tracks skill |
| `inter_rep_consistency` | `1 - mean(CV of per-rep envelope)` | thesis's recommended primary reward |
| `contact_frac` | Fraction of samples with good contact | data-quality gate |
| `quality_score` | Composite 0–1 **signal-integrity** gate | transparent, reconfigurable |

The CSV headers carry units, e.g. `snr (dB)`, `rms_amplitude (counts)`, `median_freq (Hz)`,
`onset_sharpness (1/s)` (amplitudes are raw ADC counts — the signal is uncalibrated).

Cells are left blank when a metric can't be computed (e.g. `inter_rep_consistency`
needs ≥2 detected reps). The motor-learning metrics are reported raw, not baked into a
single verdict — which of them best predicts learning is exactly the open question the
thesis is investigating, so `quality_score` gates only on *signal cleanliness*
(SNR / mains / contact). Add a metric by registering a `Scorer` in `emg/scoring.py`.
Co-contraction and recruitment specificity need a 2nd channel and are deferred.

---

## Adding a signal categorizer

The categorizations themselves are intentionally **not implemented** — only the
framework. Adding one takes a few lines and touches nothing else:

```python
# in emg/features.py
@register_feature
class MyQuality(SlidingWindowExtractor):
    name = "my_quality"
    field = "envelope"      # "raw" | "filtered" | "envelope"
    window_s = 0.5

    def compute(self) -> FeatureResult:
        w = self.window()            # numpy array, oldest-first
        return FeatureResult(self.name, float(...))
```

Enable it via `Config.enabled_features = ["my_quality"]` (or add the instance to a
`FeatureBank`). It then streams live in the UI and can be logged to `features.csv`.
`emg/features.py` ships stubs for the research targets — RMS amplitude, onset
sharpness, inter-rep consistency, co-contraction, recruitment specificity — as
ready-to-fill starting points.

---

## Filtering notes

- **High-pass 20 Hz** — removes baseline drift / motion artifact.
- **Mains notch** — fundamental + harmonics under Nyquist. Each notch also removes a
  little real EMG, so the default is fundamental + 2nd harmonic (~96% of EMG energy
  retained). Raise `notch_max_harmonics` in a noisy room, lower it if clean.
- **Low-pass 200 Hz** — below the 250 Hz Nyquist (500 Hz sampling).
- **Envelope** — rectify + 6 Hz low-pass; a smooth control signal for sonification.
  This is what fixes the old "low latency *and* good sound" trade-off: the audio
  callback reads a smooth envelope, so it stays low-latency *and* clean.
