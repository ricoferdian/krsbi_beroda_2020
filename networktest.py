import socket, threading

HOST = '192.168.43.20'
# LAPTOP DEK JUN
# PORT = 28097
# LAPTOP UCUP
PORT = 5204

networkserial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
networkserial.connect((HOST, PORT))

def updateBaseData():
    myCoordLapanganX = 0
    myCoordLapanganY = 0
    bolaLastSeenX = 0
    bolaLastSeenY = 0
    myGyro = 0
    strategyState = 3

    x1 = myCoordLapanganX
    y1 = myCoordLapanganY
    teta1 = myGyro
    bolaX = bolaLastSeenX
    bolaY = bolaLastSeenY

    while(True):
        sendDataToBase(x1, y1, teta1, bolaX, bolaY, strategyState)

def updateLocalDataFromBase():
    xRobot2 = 0.00
    yRobot2 = 0.00
    tetaRobot2 = 90.00
    while(True):
        receiveDataFromBase(xRobot2, yRobot2, tetaRobot2)

def sendDataToBase(x1, y1, teta1, bolaX, bolaY, strategyStatus):
    msg = "*"+repr(x1)+","+repr(y1)+","+repr(teta1)+","+repr(bolaX)+","+repr(bolaY)+","+repr(strategyStatus)+"#"
    # print('DATA SENT TO BASE : ',msg)
    networkserial.send(msg.encode())

def receiveDataFromBase(xRobot2, yRobot2, tetaRobot2):
    data = networkserial.recv(4096)
    data = data.decode("utf-8")
    print('DATA RECEIVED : ',data)
    if(data):
        perintahRobot(data)

def perintahRobot(command):
    global isKickOff
    print('COMMAND : ',command)
    if(command=='K'):
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        print('KICKOFF!!!!!')
        isKickOff = True
    elif(command=='r'):
        isKickOff = False
    elif(command[0]=='*'):
        print('COMMAND : ',command)

def runServerThread():
    t3 = threading.Thread(target=updateBaseData)
    t4 = threading.Thread(target=updateLocalDataFromBase)

    t3.start()
    t4.start()
    t3.join()
    t4.join()


if __name__ == '__main__':
    runServerThread()
