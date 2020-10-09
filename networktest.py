import socket, threading

HOST = '192.168.43.20'
PORT = 28097

networkserial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
networkserial.connect((HOST, PORT))

def updateBaseData():
    x1 = 1.111
    y1 = 2.11
    teta1 = 90.00
    obsX1 = 137.00
    obsY1 = 100.00
    obsX2 = 120.00
    obsY2 = 130.00
    bolaX = 5.0
    bolaY = 6.0
    while(True):
        sendDataToBase(x1, y1, teta1, obsX1, obsY1, obsX2, obsY2, bolaX, bolaY)

def updateLocalDataFromBase():
    xRobot2 = 0.00
    yRobot2 = 0.00
    tetaRobot2 = 90.00
    while(True):
        receiveDataFromBase(xRobot2, yRobot2, tetaRobot2)

def sendDataToBase(x1, y1, teta1, obsX1, obsY1, obsX2, obsY2, bolaX, bolaY):
    msg = "*"+repr(x1)+","+repr(y1)+","+repr(teta1)+","+repr(obsX1)+","+\
          repr(obsY1)+","+repr(obsX2)+","+repr(obsY2)+","+repr(bolaX)+","+repr(bolaY)+"#"
    print('DATA SENT : ',msg)
    networkserial.send(msg.encode())

def receiveDataFromBase(xRobot2, yRobot2, tetaRobot2):
    data = networkserial.recv(4096)
    print('DATA RECEIVED : ',data)

def runServerThread():
    t3 = threading.Thread(target=updateBaseData)
    t4 = threading.Thread(target=updateLocalDataFromBase)

    t3.start()
    t4.start()
    t3.join()
    t4.join()


if __name__ == '__main__':
    runServerThread()
