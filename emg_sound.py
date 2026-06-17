import collections
import sys
import threading

import numpy as np
import serial
import sounddevice as sd

SERIAL_PORT     = "COM7"
BAUD_RATE       = 115200
SAMPLE_RATE     = 48000      # 48000 / 500 = 96 exactly
BLOCK_SIZE      = 96         # 2 ms per callback
EMG_MAX         = 1023.0
FREQ_MIN        = 100.0
FREQ_MAX        = 1000.0
AMP_MAX         = 1.0
DEBUG           = "--debug" in sys.argv

emg_buf = collections.deque([0.0], maxlen=1)
phase   = [0.0]
_idx    = np.arange(BLOCK_SIZE, dtype=np.float64)

def serial_reader():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE) as ser:
            ser.reset_input_buffer()   # discard any stale startup data
            buf = b""
            while True:
                n = ser.in_waiting
                if n:
                    buf += ser.read(n)
                    *lines, buf = buf.split(b"\n")   # keep incomplete tail
                    if lines:
                        raw = lines[-1].strip()      # only the most recent line
                        if raw:
                            try:
                                val = float(raw)
                                emg_buf.append(val)
                                if DEBUG:
                                    print(f"EMG: {val:.1f}")
                            except ValueError:
                                pass
    except serial.SerialException as e:
        print(f"\nSerial error: {e}", file=sys.stderr)

def audio_callback(outdata, frames, time, status):
    t = emg_buf[0] / EMG_MAX
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0

    inc     = 2.0 * np.pi * (FREQ_MIN + t * (FREQ_MAX - FREQ_MIN)) / SAMPLE_RATE
    phases  = phase[0] + inc * _idx
    outdata[:, 0] = (np.sin(phases) * (t * AMP_MAX)).astype(np.float32)
    phase[0] = (phases[-1] + inc) % (2.0 * np.pi)

threading.Thread(target=serial_reader, daemon=True).start()
print("Running. Press Enter to stop.")
with sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                     channels=1, dtype="float32", latency=0.005,
                     callback=audio_callback):
    input()
