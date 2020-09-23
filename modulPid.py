from math import *
import time

####################
setRotateOnCam = -90
sendSerialMode = False
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


def nothing(x):
    pass


def rPix2real(rPix):
    rReal = (3 * (10 ** -34) * (rPix ** 6)) - (3 * (10 ** -10) * (rPix ** 5)) + (1 * (10 ** -7) * (rPix ** 4)) - (
                2 * (10 ** -5) * (rPix ** 3)) - (0.0025 * (rPix ** 2)) + (1.1661 * rPix) + 43.035
    return rReal


def constraint(x, xmin, xmax):
    return max(min(xmax, x), xmin)


def loadData():
    global _focus
    global hminBall, hmaxBall, sminBall, smaxBall, vminBall, vmaxBall, esizeBall, dsizeBall, blurBall
    global hminField, hmaxField, sminField, smaxField, vminField, vmaxField, esizeField, dsizeField, blurField

    f = open("save_setting.txt", "r")
    for line in f.readlines():
        tempData = line.split(',')
        _focus = int(tempData[0])
        hminBall = int(tempData[1])
        hmaxBall = int(tempData[2])
        sminBall = int(tempData[3])
        smaxBall = int(tempData[4])
        vminBall = int(tempData[5])
        vmaxBall = int(tempData[6])
        esizeBall = int(tempData[7])
        dsizeBall = int(tempData[8])
        blurBall = int(tempData[9])
        hminField = int(tempData[10])
        hmaxField = int(tempData[11])
        sminField = int(tempData[12])
        smaxField = int(tempData[13])
        vminField = int(tempData[14])
        vmaxField = int(tempData[15])
        esizeField = int(tempData[16])
        dsizeField = int(tempData[17])
        blurField = int(tempData[18])

def saveData():
    f = open("save_setting.txt", "w")
    data = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (
    _focus, hminBall, hmaxBall, sminBall, smaxBall, vminBall, vmaxBall, esizeBall, dsizeBall, blurBall, hminField,
    hmaxField, sminField, smaxField, vminField, vmaxField, esizeField, dsizeField, blurField)
    f.write(data)
    f.close()

# fungsi jarak
def distancePixel(tengah_x, tengah_y, titik_x, titik_y):
    delx = titik_x - float(tengah_x)
    dely = titik_y - float(tengah_y)
    j_rad = sqrt(pow(delx, 2) + pow(dely, 2))
    return j_rad

# fungsi teta
def teta(tengah_x, tengah_y, titik_x, titik_y):
    delx = -(float(tengah_x) - titik_x)
    dely = float(tengah_y) - titik_y
    oteta = atan2(delx, dely)
    oteta = oteta / pi * 180
    oteta = oteta + setRotateOnCam
    if oteta > 180:
        oteta = oteta - 360
    elif oteta < -180:
        oteta = oteta + 360
    return oteta

def pidRobot(tetaBall, realDistanceX, realDistanceY, ser):

    import serial

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
        # vMotor3 = round(vMotor3, 2)
        # msg = "*" + repr(vMotor1) + ",0" + "," + repr(vMotor2) + "," + repr(vMotor3) + "#"
        msg = "*"+repr(realDistanceX)+","+repr(realDistanceY)+","+repr(pidTeta)+"#"
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
        ser.open()
        ser.write(msg.encode())
        currentTime = time.time()
    #     # print currentTime
    #     try:
    #         if ser.is_open == True:
    #         elif ser.is_open == False:
    #             ser.open()
    #         else:
    #             print
    #             "ada masalah"
    #     except:
    #         print
    #         "tidak bisa kirim"
    #         try:
    #             ser.close()
    #             ser.open()
    #         except:
    #             sendSerialMode = False
    #         pass
    elif tetaBall == None:
        if time.time() - currentTime > 1.0:
            try:
                ser.write(b'*0,0,0,0#')
            except:
                pass
