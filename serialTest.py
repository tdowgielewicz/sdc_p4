import serial
import time
from struct import *


COMportNumber = 'COM4'
COMportBaudrate = 250000
COMportBytesize = serial.EIGHTBITS
COMportParity = serial.PARITY_NONE
COMportStopbits = serial.STOPBITS_ONE
COMportTimeout = 0.05



serial_handler = serial.Serial(COMportNumber, baudrate=COMportBaudrate, bytesize=COMportBytesize,
                               parity=COMportParity, stopbits=COMportStopbits, timeout=COMportTimeout)
def MoveTo(angle):

    pos = angle + 127

    serial_handler.write(serial.to_bytes([pos]))
    print(serial_handler.readlines())







for i in range (40):

    MoveTo(-100)


    time.sleep(1)

    MoveTo(100)

    time.sleep(2)

    MoveTo(0)

    time.sleep(2)