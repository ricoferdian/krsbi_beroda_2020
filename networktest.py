import socket, threading

HOST = '192.168.43.118'
PORT = 28097

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

# while(True):
#     arr = ([1,2,3,4,5,6,7,8,9])
#     data_string = pickle.dumps(arr)
#     print("SENDING DATA")
#     print("SENDING DATA",data_string)
#     s.send(data_string)
#
#     data = s.recv(4096)
#     print("RECEIVING DATA")
#     print(data)
    # data_arr = pickle.loads(data)
    # s.close()
    # print('Received', repr(data_arr))

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
    s.send(msg.encode())

def receiveDataFromBase(xRobot2, yRobot2, tetaRobot2):
    data = s.recv(4096)
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
