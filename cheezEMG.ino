/*
  cheezEMG firmware  —  streaming acquisition front-end
  -----------------------------------------------------
  Serial:  115200 baud, 500 Hz sample rate.

  Output (one CSV line per sample, '\n' terminated):

        raw,filtered,envelope,detect

    raw       int    0..1023   raw ADC value (A0). This is the channel the
                               host filters authoritatively (SW high-pass +
                               mains notch + low-pass + envelope).
    filtered  float            firmware Butterworth band-pass (reference only;
                               note it has NO mains notch).
    envelope  int              firmware moving-average envelope (reference).
    detect    int    0/1       wear / skin-contact flag (yellow wire, pin 2).

  Why stream raw instead of just the envelope?
    The cheezEMG board uses dry conductive PCB pads, not gel stickers, so the
    signal carries a lot of mains hum and motion drift. Filtering on the host
    (see emg/streaming.py) lets us place a 50 Hz (eastern Japan) / 60 Hz notch
    exactly where it is needed and re-process recordings offline. The firmware
    columns are kept alongside for comparison and for the contact indicator.
*/
#include "CheezsEMG.h"
#define SAMPLE_RATE 500        // Sample rate (Hz)
#define BAUD_RATE   115200     // Serial baud rate
#define INPUT_PIN   A0         // Signal input (white wire)
#define DETECT_PIN  2          // Contact / wear detection (yellow wire)

CheezsEMG sEMG(INPUT_PIN, DETECT_PIN, SAMPLE_RATE);

void setup()
{
  Serial.begin(BAUD_RATE);
  sEMG.begin();
}

void loop()
{
  if (sEMG.checkSampleInterval())
  {
    sEMG.processSignal();

    // Print fields individually to avoid String heap churn at 500 Hz.
    Serial.print(sEMG.getRawSignal());        Serial.print(',');
    Serial.print(sEMG.getFilteredSignal(), 2); Serial.print(',');
    Serial.print(sEMG.getEnvelopeSignal());   Serial.print(',');
    Serial.println(sEMG.getDetectSignal());
  }
}
