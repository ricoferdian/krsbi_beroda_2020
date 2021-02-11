import serial
import socket
from findpath import rotateMatrix
from objects.all_field_object import AllFielOjects
from objects.robot import Robot

gyro_calibration = 0

ser = serial.Serial('COM3', 115200, timeout=100000)


def send_serial_data(x, y, teta, is_tendang, is_bola_dekat, is_reset):
    msg = "*" + repr(x) + "," + repr(y) + "," + repr(teta) + "," + repr(is_tendang) + "," + repr(
        is_bola_dekat) + "," + repr(is_reset) + "#"
    ser.write(msg.encode())


def read_serial_data(robot_object: Robot, all_field_objects: AllFielOjects):
    readdata = ''
    micro_data_received = [0 for i in range(4)]
    i = 0

    while (True):
        msg = ser.read()
        strmsg = msg.decode("utf-8")
        if (strmsg == '*'):
            readdata = ''
        elif (strmsg == ','):
            micro_data_received[i] = float(readdata)
            i += 1
            readdata = ''
        elif (strmsg == '#'):
            micro_data_received[i] = float(readdata)
            i = 0

            # data yang diterima, diambil masukkan ke variabel
            my_trajectory_x = micro_data_received[0]
            my_trajectory_y = micro_data_received[1]
            my_gyro = micro_data_received[2]
            is_dribbling_bola = micro_data_received[3]

            # get absolute koordinat robot terhadap lapangan, belum dipake
            # my_absolute_x, my_absolute_y = rotateMatrix(my_trajectory_x, my_trajectory_y, my_gyro - gyro_calibration)

            # set value objek robot
            robot_object.set_my_trajectory(my_trajectory_x, my_trajectory_y)
            robot_object.set_gyro(my_gyro)
            if (is_dribbling_bola == 1):
                robot_object.set_dribble(True)
            else:
                robot_object.set_dribble(False)
        else:
            readdata += strmsg
