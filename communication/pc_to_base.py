import socket
import time

HOST = '192.168.43.178'
PORT = 5204
arrayStrategy = [0, 3, 2, 0, 0, 5, 0, 0, 0, 1]
robotId = 2

networkserial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
networkserial.connect((HOST, PORT))


def send_data_to_base(robot_object, all_field_objects):
    x1, y1 = robot_object.get_my_trajectory()
    teta1 = robot_object.get_gyro()
    bolaX = 0
    bolaY = 0

    while (True):
        parse_data_to_base(x1, y1, teta1, bolaX, bolaY, 1)


def parse_data_to_base(x1, y1, teta1, bolaX, bolaY, strategyStatus):
    time.sleep(1)
    msg = "*" + repr(x1) + "," + repr(y1) + "," + repr(teta1) + "," + repr(bolaX) + "," + repr(bolaY) + "," + repr(
        strategyStatus) + "#"
    print('DATA SENT TO BASE : ', msg)
    networkserial.send(msg.encode())


def get_data_from_base(robot_object, all_field_objects):
    xRobot2 = 0.00
    yRobot2 = 0.00
    tetaRobot2 = 90.00
    while (True):
        data = networkserial.recv(4096)
        data = data.decode("utf-8")
        print('data DARI BASE STATION', data)
        if (data):
            if (data == 'K'):
                is_kickoff = True
            if (data == 'r'):
                is_kickoff = False
            elif(data[0]=='*'):
                parse_base_command(data)

def parse_base_command(command):
    readdata = ''
    is_not_end = True
    base_data_received = [0.0 for i in range(8)]
    commandIndex = 0
    i = 0
    while (is_not_end):
        if (command[commandIndex] == '*'):
            readdata = ''
        elif (command[commandIndex] == ','):
            base_data_received[i] = float(readdata)
            i += 1
            readdata = ''
        elif (command[commandIndex] == '#'):
            is_not_end = False
            base_data_received[i] = float(readdata)
            i = 0
        else:
            readdata += command[commandIndex]
        commandIndex += 1
