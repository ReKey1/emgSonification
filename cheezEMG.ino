/*
  cheezEMG firmware  —  raw acquisition only
  ------------------------------------------
  Serial:  115200 baud, 500 Hz sample rate.
  Output:  one integer per line — the raw 10-bit analogRead (0..1023) of A0.

  This sketch does NO signal processing on purpose. The board uses dry
  conductive PCB pads (noisy: mains hum + drift), and ALL filtering — high-pass,
  mains notch (50 Hz eastern Japan / 60 Hz western Japan), low-pass, envelope —
  is done on the host in Python (emg/streaming.py). Keeping the firmware to a
  bare analogRead makes the whole signal path visible and tunable in one place
  instead of hidden inside a library. No external library is required.
*/
#define SAMPLE_RATE 500        // Hz
#define BAUD_RATE   115200     // Serial baud rate
#define INPUT_PIN   A0         // Signal input (white wire)

const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE;  // 2000 us

unsigned long lastSampleMicros = 0;

void setup()
{
  Serial.begin(BAUD_RATE);
  lastSampleMicros = micros();   // seed so the first sample is one full interval out
}

void loop()
{
  // Fixed-rate sampling via a micros() accumulator. Unsigned subtraction is
  // overflow-safe, and advancing by the interval (rather than snapping to now)
  // keeps the long-term rate drift-free at exactly SAMPLE_RATE.
  unsigned long now = micros();
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US)
  {
    lastSampleMicros += SAMPLE_INTERVAL_US;
    Serial.println(analogRead(INPUT_PIN));
  }
}
