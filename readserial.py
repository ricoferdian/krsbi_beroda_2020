# Import socket module
import serial
import socket
import struct

sendSerialMode=True
sendSocketMode=False
# Create a socket object
s = socket.socket()

# Define the port on which you want to connect
port = 12345

# def keyControl(keys):
#     if keys == ord("s"):
#         if sendSocketMode==False:
#             sendSocketMode=True
#             # connect to the server on local computer
#             s.connect(('127.0.0.1', port))
#             print("Mode ready to send socket communication On")
#         elif sendSocketMode==True:
#             sendSocketMode=False
#             s.close()
#             print("Mode ready to send socket communication off")
#     elif keys == ord("c"):
#         if sendSerialMode==False:
#             sendSerialMode=True
#             print("Mode ready to send serial communication On")
#         elif sendSerialMode==True:
#             sendSerialMode=False
#             print("Mode ready to send serial communication off")

def parseData(xyresgyro):
    print('xyresgyro:',xyresgyro)

_serDeclare=True
ser=serial.Serial('COM3',115200,timeout=100000)
readdata = ''
xyresgyro = [0 for i in range(4)]
i = 0
while(True):
        #print sendSerialMode
    if sendSerialMode==True:
        if sendSerialMode==True:
            msg=ser.read()
            strmsg = msg.decode("utf-8")
            if(strmsg=='*'):
                readdata = ''
            elif (strmsg == ','):
                xyresgyro[i] = float(readdata)
                i += 1
                readdata = ''
            elif(strmsg=='#'):
                xyresgyro[i] = float(readdata)
                i  = 0
                parseData(xyresgyro)
            else:
                readdata += strmsg
            # print ('strmsg:'+strmsg)

    elif sendSerialMode==False:
        ser.close()

    if sendSocketMode==True:
        print (s.recv(1024))
        s.send('kontrol')