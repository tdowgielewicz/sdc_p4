#include <Arduino.h>


#include "AccelStepper.h"

#define DIR 5
#define STEP 2
#define enabled_pin 8

// 2-wire basic config, microstepping is hardwired on the driver
AccelStepper stepper(1, STEP, DIR);

int inByte = 0;  

void setup() {


    Serial.begin(250000);
    pinMode(enabled_pin, OUTPUT);
    digitalWrite(enabled_pin, LOW);


    stepper.setMaxSpeed(2000.0);
    stepper.setAcceleration(2200.0);
    stepper.setSpeed(3000);
    stepper.moveTo(2);
    

}

void loop() {

   if (Serial.available() > 0) {
   // get incoming byte:
   inByte = Serial.read();
    inByte -= 127;
    long absPos = inByte * 10;
   
    //Serial.println(absPos, DEC);

    stepper.moveTo(absPos);
   }

    
    stepper.run();

}
