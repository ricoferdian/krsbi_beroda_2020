import serial
import socket
import threading

def readSerialData():
    sendSerialMode = True
    sendSocketMode = False
    # Create a socket object
    s = socket.socket()

    # Define the port on which you want to connect
    port = 12345

    _serDeclare = True
    readdata = ''
    xyresgyro = [0 for i in range(6)]
    i = 0

    global ser
    ser = serial.Serial('COM3', 115200, timeout=100000)

    global myCoordX
    global myCoordY
    global myRes
    global myGyro
    while (True):
        # print sendSerialMode
        if sendSerialMode == True:
            if sendSerialMode == True:
                msg = ser.read()
                strmsg = msg.decode("utf-8")
                if (strmsg == '*'):
                    readdata = ''
                elif (strmsg == ','):
                    xyresgyro[i] = float(readdata)
                    i += 1
                    readdata = ''
                elif (strmsg == '#'):
                    xyresgyro[i] = float(readdata)
                    print('xyresgyro',xyresgyro)
                    i = 0
                    myCoordX = xyresgyro[0]
                    myCoordY = xyresgyro[1]
                    myGyro = xyresgyro[2]

                    # print('myCoordX',myCoordX)
                    # print('myCoordY',myCoordY)
                    # print('myRes',myRes)
                    # print('myGyro',myGyro)

                else:
                    readdata += strmsg
                # print ('strmsg:'+strmsg)

        elif sendSerialMode == False:
            ser.close()

        if sendSocketMode == True:
            print(s.recv(1024))
            s.send('kontrol')
