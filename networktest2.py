import socket, threading

HOST = '192.168.43.194'
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
    strategyState = 5

    x1 = myCoordLapanganX
    y1 = myCoordLapanganY
    teta1 = myGyro
    bolaX = bolaLastSeenX
    bolaY = bolaLastSeenY

    i = 0
    isIter = True
    while(isIter):
        i+=1
        if(i==20):
            isIter = False
        sendDataToBase(x1, y1, teta1, bolaX, bolaY, i)

def updateLocalDataFromBase():
    xRobot2 = 0.00
    yRobot2 = 0.00
    tetaRobot2 = 90.00
    while(True):
        receiveDataFromBase(xRobot2, yRobot2, tetaRobot2)

def sendDataToBase(x1, y1, teta1, bolaX, bolaY, strategyStatus):
    msg = "*"+repr(x1)+","+repr(y1)+","+repr(teta1)+","+repr(bolaX)+","+repr(bolaY)+","+repr(strategyStatus)+"#"
    print('DATA SENT TO BASE : ',msg)
    networkserial.send(msg.encode())

def receiveDataFromBase(xRobot2, yRobot2, tetaRobot2):
    data = networkserial.recv(4096)
    data = data.decode("utf-8")
    # print('DATA RECEIVED : ',data)
    if(data):
        perintahRobot(data)

def perintahRobot(command):
    global isKickOff
    print('COMMAND [0] : ',command[0])
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
        parseCommand(command)

def parseCommand(command):
    readdata = ''
    isNotEnd = True
    xystrategy = [0 for i in range(8)]
    commandIndex = 0
    i = 0
    while(isNotEnd):
        if (command[commandIndex] == '*'):
            readdata = ''
        elif (command[commandIndex] == ','):
            xystrategy[i] = float(readdata)
            i += 1
            readdata = ''
        elif (command[commandIndex] == '#'):
            isNotEnd = False
            xystrategy[i] = float(readdata)
            i = 0
        else:
            readdata += command[commandIndex]
        commandIndex += 1

def runServerThread():
    t3 = threading.Thread(target=updateBaseData)
    t4 = threading.Thread(target=updateLocalDataFromBase)

    t3.start()
    t4.start()
    t3.join()
    t4.join()


if __name__ == '__main__':
    runServerThread()
