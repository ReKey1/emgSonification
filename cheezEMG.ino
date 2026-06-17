/* 
Serial output:
        Baud rate: 115200
        Sample rate: 500Hz 
Serial output content (ASCII):
        Raw data, filtered EMG signal, envelope signal, wear detection signal
*/
#include "CheezsEMG.h"
#define SAMPLE_RATE 500        // Sample rate
#define BAUD_RATE 115200       // Serial baud rate
#define INPUT_PIN A0           // Signal input (white wire)
#define DETECT_PIN 2           // Detection input (yellow wire)
 
// Using default configuration  
CheezsEMG sEMG(INPUT_PIN, DETECT_PIN, SAMPLE_RATE);  

void setup() 
{
  Serial.begin(BAUD_RATE);
  sEMG.begin();  
}  

void loop() 
{   
  if(sEMG.checkSampleInterval())
  {
    sEMG.processSignal();  
    Serial.println(
        //String(sEMG.getRawSignal()) + "," +      // Raw data
        //String(sEMG.getFilteredSignal()) + "," + // Filtered data
        String(sEMG.getEnvelopeSignal())// + "," + // Envelope data
        //String(sEMG.getDetectSignal())           // Wear status
    );
  }
}