"""Headless recorder — record a session without the GUI.

Examples:
    python record_cli.py --synthetic --seconds 5 --name test
    python record_cli.py --port COM7 --name bicep-curl --notes "left arm"
    python record_cli.py --port COM7 --mains 60          # western Japan

Handy for quick captures, for scripting, and for smoke-testing the pipeline on
a machine with no hardware (--synthetic).
"""

from __future__ import annotations

import argparse
import sys
import time

from emg.config import Config
from emg.pipeline import Pipeline
from emg.source import open_source


def main() -> int:
    ap = argparse.ArgumentParser(description="Record a cheezEMG session to disk.")
    ap.add_argument("--synthetic", action="store_true", help="use the built-in fake source")
    ap.add_argument("--port", help="serial port (e.g. COM7)")
    ap.add_argument("--seconds", type=float, default=10.0, help="recording length")
    ap.add_argument("--name", default="session", help="session name (folder suffix)")
    ap.add_argument("--notes", default="", help="free-text notes stored in session.json")
    ap.add_argument("--mains", type=float, help="mains frequency: 50 (E. Japan) or 60 (W. Japan)")
    args = ap.parse_args()

    cfg = Config()
    if args.synthetic:
        cfg.source = "synthetic"
    if args.port:
        cfg.serial_port = args.port
        cfg.source = "serial"
    if args.mains:
        cfg.mains_hz = args.mains

    try:
        source = open_source(cfg)
    except Exception as e:
        print(f"Could not open source ({cfg.source}): {e}", file=sys.stderr)
        return 1

    pipe = Pipeline(cfg)
    pipe.start(source)
    session = pipe.start_recording(args.name, args.notes)
    print(f"Recording [{source.name}] -> {session}")
    print(f"Notch: {cfg.notch_freqs()} Hz | high-pass {cfg.highpass_hz} Hz | "
          f"low-pass {cfg.lowpass_hz} Hz")

    try:
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            time.sleep(0.5)
            if pipe.error:
                print(f"\nPipeline error: {pipe.error}", file=sys.stderr)
                break
            print(f"\r  {pipe.samples_seen:6d} samples | "
                  f"~{pipe.measured_rate:5.1f} Hz | "
                  f"contact {'OK ' if pipe.contact_ok else 'POOR'} | "
                  f"level {pipe.audio_level:4.2f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        out = pipe.stop_recording()
        pipe.stop()
        print(f"\nSaved {pipe.recorder.sample_count} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
