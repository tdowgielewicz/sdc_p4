import serial
import time
from struct import *


COMportNumber = 'COM4'
COMportBaudrate = 250000
COMportBytesize = serial.EIGHTBITS
COMportParity = serial.PARITY_NONE
COMportStopbits = serial.STOPBITS_ONE
COMportTimeout = 0.01



serial_handler = serial.Serial(COMportNumber, baudrate=COMportBaudrate, bytesize=COMportBytesize,
                               parity=COMportParity, stopbits=COMportStopbits, timeout=COMportTimeout)
def MoveTo(angle):

    pos = int(angle) + 127

    serial_handler.write(serial.to_bytes([pos]))
    print(serial_handler.readlines(),pos,angle)







# for i in range (40):
#
#     MoveTo(-34)
#
#
#     time.sleep(1)
#
#     MoveTo(22)
#
#     time.sleep(2)
#
#     MoveTo(0)
#
#     time.sleep(2)


pozes = [
-32.5072682233 ,
-36.0082329505 ,
-38.9067174683 ,
-39.1785402845 ,
-36.6157447539 ,
-34.5830475709 ,
-32.4355864712 ,
-29.1618878891 ,
-25.6367559046 ,
-23.4153205231 ,
-21.3498623789 ,
-19.5273535287 ,
-19.7572019569 ,
-20.7812533533 ,
-22.4125892054 ,
-25.2168374166 ,
-29.9185609348 ,
-34.5027021108 ,
-36.5767339439 ,

]

for pos in (pozes):
    MoveTo(pos)
    time.sleep(0.5)
