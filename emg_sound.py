import sys
import numpy as np
import serial
import sounddevice as sd

SERIAL_PORT     = "COM7"
BAUD_RATE       = 115200
SERIAL_RATE     = 500
SAMPLE_RATE     = 48000      # 48000 / 500 = 96 exactly — no timing drift
EMG_MAX         = 1023.0
FREQ_MIN        = 100.0      # Hz at rest
FREQ_MAX        = 1000.0     # Hz at full flex
AMP_MAX         = 1.0
DEBUG           = "--debug" in sys.argv

SAMPLES_PER_EMG = SAMPLE_RATE // SERIAL_RATE   # 96
_idx            = np.arange(SAMPLES_PER_EMG, dtype=np.float64)
_silence        = np.zeros((SAMPLES_PER_EMG, 1), dtype=np.float32)

phase = 0.0
with serial.Serial(SERIAL_PORT, BAUD_RATE) as ser, \
     sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                     dtype="float32", latency="low") as stream:
    print("Running. Ctrl+C to stop.")
    while True:
        line = ser.readline()
        if not line:
            stream.write(_silence)
            continue
        try:
            val = float(line)
        except ValueError:
            stream.write(_silence)
            continue

        if DEBUG:
            print(f"EMG: {val:.1f}")

        t = val / EMG_MAX
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0

        inc     = 2.0 * np.pi * (FREQ_MIN + t * (FREQ_MAX - FREQ_MIN)) / SAMPLE_RATE
        samples = (np.sin(phase + inc * _idx) * (t * AMP_MAX)).astype(np.float32)
        stream.write(samples.reshape(-1, 1))
        phase   = (phase + inc * SAMPLES_PER_EMG) % (2.0 * np.pi)
