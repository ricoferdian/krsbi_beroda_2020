import threading
import numpy as np
from math import *
import time
import serial

####################
destroy = False
_key = ""
lastMode = 10
modeCam = 1  # mode 1 atau 0 untuk usb webcam
centerDisplay = (343, 238)
showBallParam = False
sendSerialMode = False
receiveSocketMode = False
setRotateOnCam = -90
global hsv, camera, camera1
###ball detection
global radiusBall, centerBall, ballMask, cntsBall
global controlDebug, debugMode, mode
tetaBall = None
rPixelBall = None
realDistanceX = 0
realDistanceY = 0
###pidTeta
errorTeta = 0
previousErrorTeta = 0
pTeta = 0
iTeta = 0
dTeta = 0
errorRealDistanceX = 0
previousErrorRealDistanceX = 0
pX = 0
iX = 0
dX = 0
errorRealDistanceY = 0
previousErrorRealDistanceY = 0
pY = 0
iY = 0
dY = 0

def constraint(x, xmin, xmax):
    return max(min(xmax, x), xmin)

def pidRobot(tetaBall, realDistanceX, realDistanceY, ser):

    lRobot = 22
    currentTime = 0.0
    ####
    KpTeta = 0.01
    KiTeta = 0.0001
    KdTeta = 0.001
    ####
    KpX = 0.88
    KiX = 0
    KdX = 0.1
    ####
    KpY = 0.88
    KiY = 0
    KdY = 0.1

    global iTeta, iX, iY, destroy, sendSerialMode, previousErrorTeta, previousErrorRealDistanceX, previousErrorRealDistanceY

    # ser = serial.Serial('COM5', 115200, timeout=1)
    # print sendSerialMode
    if sendSerialMode == True:
        try:
            ser.open()
        except:
            if ser.is_open == False:
                sendSerialMode = False
        finally:
            pass

            # sendSerialMode=False
            # print  "tidak bisa mengirim"

    #            if ser.is_open==False:

    elif sendSerialMode == False:
        try:
            if ser.is_open == True:
                ser.write(b'*0,0,0,0#')
                ser.close()
                # print msg
        except:
            pass

    if tetaBall != None:
        errorTeta = 0 - tetaBall
        pTeta = errorTeta
        iTeta = iTeta + errorTeta
        dTeta = errorTeta - previousErrorTeta
        iTeta = constraint(iTeta, -22, 22)

        pidTeta = (KpTeta * pTeta) + (KiTeta * iTeta) + (KdTeta * dTeta)
        pidTeta = constraint(pidTeta, -150, 150)
        previousErrorTeta = errorTeta
        print('pidT= ', pidTeta)

        # realDistanceX
        # pidX
        if realDistanceX != None and realDistanceY != None:
            errorRealDistanceX = 0 - realDistanceX
            if abs(errorRealDistanceX) < 100:
                errorRealDistanceX = 0
            pX = errorRealDistanceX
            iX = iX + errorRealDistanceX
            dX = errorRealDistanceX - previousErrorRealDistanceX
            iX = constraint(iX, -22, 22)

            pidX = (KpX * pX) + (KiX * iX) + (KdX * dX)
            pidX = constraint(pidX, -125, 125)
            # print "pidX=",pidX

            # pidY
            if realDistanceY != None:
                errorRealDistanceY = 0 - realDistanceY
                if abs(errorRealDistanceX) < 28:
                    errorRealDistanceX = 0

                pY = errorRealDistanceY
                iY = iY + errorRealDistanceY
                dY = errorRealDistanceY - previousErrorRealDistanceY
                iY = constraint(iY, -22, 22)

                pidY = (KpY * pY) + (KiY * iY) + (KdY * dY)
                pidY = constraint(pidY, -125, 125)
                # print "pidY=", pidY
        if realDistanceX == None and realDistanceY == None:
            pidX = 0
            pidY = 0
            print("pidY=", pidY)
            print("pidX=", pidX)
        # vMotor1 = pidX * sin(radians(30)) + pidY * cos(radians(30)) + lRobot * pidTeta
        # vMotor2 = pidX * sin(radians(30)) - pidY * cos(radians(30)) + lRobot * pidTeta
        # vMotor3 = -pidX + lRobot * pidTeta
        #
        # # out kecepatan motor
        #
        # # vMotor1=lRobot*pidTeta
        # # vMotor2=lRobot*pidTeta
        # # vMotor3=lRobot*pidTeta
        #
        # vMotor1 = round(vMotor1, 2)
        # vMotor2 = round(vMotor2, 2)
        # vMotor3 = round(vMotor3, 2)
        # msg = "*" + repr(vMotor1) + ",0" + "," + repr(vMotor2) + "," + repr(vMotor3) + "#"
        msg = "*"+repr(pidX)+","+repr(pidY)+","+repr(pidTeta)+"#"
        currentTime = time.time()
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        print('msg for PID',msg)
        # print currentTime
        ser.open()
        ser.write(msg.encode())
        # try:
            # if ser.is_open == True:
            #     print('PID WRITING')
            #     print('PID WRITING')
            #     print('PID WRITING')
            #     print('PID WRITING')
            #     print('PID WRITING')
            #     print('PID WRITING')
            #     print('PID WRITING')
            #     ser.write(msg)
            #     print(msg)
            # elif ser.is_open == False:
            #     print('PID OPENING')
            #     print('PID OPENING')
            #     print('PID OPENING')
            #     print('PID OPENING')
            #     print('PID OPENING')
            #     print('PID OPENING')
            #     print('PID OPENING')
            #     print('PID OPENING')
            #     ser.open()
            # else:
            #     print("ada masalah")
        # except:
        #     print("tidak bisa kirim")
        #     try:
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         print('PID CLOSING')
        #         ser.close()
        #         ser.open()
        #     except:
        #         sendSerialMode = False
        #     pass

    elif tetaBall == None:
        if time.time() - currentTime > 1.0:
            try:
                ser.write(b'*0,0,0,0#')
            except:
                pass

    time.sleep(0.1)
    # _keys = cv2.waitKey(1)
    # keyControl(_keys)
    # if _keys == ord("q") or destroy == True:
    #     destroy = True
    #     try:
    #         ser.close()
    #     except:
    #         pass
    #     break
