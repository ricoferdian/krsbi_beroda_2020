import argparse
import torch.backends.cudnn as cudnn

import serial
import socket
import threading

from communication.pc_to_base import send_data_to_base, get_data_from_base
from communication.pc_to_micro import readSerialData
from models.experimental import *
from objects.all_field_object import AllFielOjects
from objects.field_object import FieldObject
from objects.robot import Robot
from utils.datasets import *
from utils.utils import *
from findpath import *

global myCoordX
global myCoordY
global myCoordLapanganX
global myCoordLapanganY
global bolaLastSeenX
global bolaLastSeenY
# global myRes
global myGyro

# Strategi dan base station value
global is_kickoff
global strategyState
global isDribblingBola

# Serial port Arduino
global ser

# Gyro calibration sesuaikan dengan sudut gyro saat menghadap gawangs
gyroCalibration = 0

HOST = '192.168.43.178'
# LAPTOP UCUP
# PORT = 28097
# arrayStrategy = [0,1,2,3,0,5,6,7,0,0]
# robotId = 1
# LAPTOP DEK JUN
PORT = 5204
arrayStrategy = [0, 3, 2, 0, 0, 5, 0, 0, 0, 1]
robotId = 2

networkserial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
networkserial.connect((HOST, PORT))

cameraCenterX = 295
cameraCenterY = 248


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        'inference/output', '1', 'D:\\Libraries\\Project\\Python\\yolov5\\runs\\exp8\\weights\\best.pt', None, None, 640
    webcam = source == '0' or source == '1' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')

    maxXLapangan = 450
    maxYLapangan = 600
    splitSizeGrid = 50

    global is_kickoff
    global strategyState
    global isDribblingBola

    isKickOff = False

    isDribblingBola = False

    global myCoordX
    global myCoordY
    global bolaLastSeenX
    global bolaLastSeenX
    global myGyro
    myCoordX = 0
    myCoordY = 0
    bolaLastSeenX = 0
    bolaLastSeenY = 0

    global myCoordLapanganX
    global myCoordLapanganY

    if (myCoordX is not None):
        myCoordLapanganX = myCoordX
    else:
        myCoordLapanganX = 0
    if (myCoordY is not None):
        myCoordLapanganY = myCoordY
    else:
        myCoordLapanganY = 0
    myRes = 0
    myGyro = 0

    gridLapangan = gridGenerator(maxXLapangan, maxYLapangan, splitSizeGrid)

    matrix = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Iterasi object terdeteksi
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    object = FieldObject(cameraCenterX, cameraCenterY)
                    object.set_center_objects(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

                    # arr_objects.append({'label': names[int(cls)],'conf': float(conf),
                    #                     'x1': int(xyxy[0]), 'y1': int(xyxy[1]),
                    #                     'x2': int(xyxy[2]), 'y2': int(xyxy[3])})

                # arr_objects = getcoordinate(arr_objects)

                # print('arr_objects',arr_objects)
                isNemuBola = False
                start = {}
                end = {}

                # kalibrasi gyro
                # myGyro = 90 #HILANGINNNNN!!!!!!!!!!!!!!
                isEndpointInit = False
                tetaBall = None
                realDistanceX = 0
                realDistanceY = 0

                # matrix = [
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # ]

                matrix = [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]

                isTendangBola = False
                isBolaDekat = False

                if (len(arr_objects) > 0):
                    for object in arr_objects:
                        # Iterate object dan definisikan lokasinya di lapangan
                        rotationAngle = myGyro - gyroCalibration
                        # X dan Y dibalik karena kamera bacanya kebalik
                        object['x'], object['y'] = rotateMatrix(object['realDistanceY'], object['real_distance_x'],
                                                                rotationAngle)
                        # object['gridReal'] = getGridLocationFromCoord(object,splitSizeGrid)

                        # print('object location real',object)

                        if (object['label'] == 'bola'):
                            isNemuBola = True
                            if (not isDribblingBola):
                                print('AKU NYARI BOLA', object)
                                isNyariBola = True
                                isTendangBola = False
                                end = {}
                                end['x'] = object['x']
                                end['y'] = object['y']
                                bolaLastSeenX = object['x']
                                bolaLastSeenY = object['y']
                                tetaBall = object['tetaObj']
                                realDistanceX = object['real_distance_x']
                                realDistanceY = object['realDistanceY']
                                if (not isEndpointInit):
                                    isEndpointInit = True
                                if (realDistanceY < 50):
                                    print('BOLA SUDAH DEKAT')
                                    isBolaDekat = True
                            else:
                                if (not isEndpointInit):
                                    end = None
                                    isEndpointInit = True
                        elif (object['label'] == 'gawang' and isDribblingBola):
                            # Cari gawangs
                            print('AKU NYARI GAWANG', object)
                            end = {}
                            end['x'] = object['x']
                            end['y'] = object['y']
                            isEndpointInit = True
                            tetaBall = object['tetaObj']
                            realDistanceX = object['real_distance_x'] * 0.8
                            realDistanceY = object['realDistanceY'] * 0.8
                            # JIKA GAWANG DEKAT, TENDANG
                            if (isDribblingBola and realDistanceY < 400):
                                isTendangBola = True
                            # elif(realDistanceY<400):
                            #     realDistanceY = 0
                            #     real_distance_x = 0

                            else:
                                if (not isEndpointInit):
                                    end = None
                                    isEndpointInit = True
                        # elif (object['label'] == 'robot'):
                        #     if (isDribblingBola or (strategyState==arrayStrategy[2] or strategyState==arrayStrategy[5] or strategyState==arrayStrategy[9])):
                        #         # Cari gawang
                        #         print('AKU NYARI TEMENKU DIMANA', object)
                        #         end = {}
                        #         end['x'] = object['x']
                        #         end['y'] = object['y']
                        #         isEndpointInit = True
                        #         tetaBall = object['tetaObj']
                        #         real_distance_x = object['real_distance_x']*0.9
                        #         realDistanceY = object['realDistanceY']*0.9
                        #         #JIKA ROBOT DEKAT, TENDANG
                        #         if(not isDribblingBola):
                        #             realDistanceY = 0
                        #             real_distance_x = 0
                        #         if(isDribblingBola and realDistanceY<300):
                        #             realDistanceY = 0
                        #             isTendangBola = True
                        #     else:
                        #         if (not isEndpointInit):
                        #             end = None
                        #             isEndpointInit = True
                        # elif(object['label'] == 'obstacle'):
                        #     print('ADA OBSTACLE', object)
                        #     obstacle = {}
                        #     obstacle['x'] = object['x']*0.9
                        #     obstacle['y'] = object['y']*0.9
                        #     obstacleGridLoc = getGridLocationFromCoord(obstacle,splitSizeGrid)
                        #     if(obstacleGridLoc[0]>11):
                        #         obstacleGridLoc[0] = 11
                        #     elif(obstacleGridLoc[0]<0):
                        #         obstacleGridLoc[0] = 0
                        #     if(obstacleGridLoc[1]>8):
                        #         obstacleGridLoc[1] = 8
                        #     elif(obstacleGridLoc[1]<0):
                        #         obstacleGridLoc[1] = 0
                        #
                        #     # print("obstacleGridLoc",obstacleGridLoc)
                        #     matrix[obstacleGridLoc[1]][obstacleGridLoc[0]] = 0
                        #     if(not isNemuBola or not isDribblingBola):
                        #         if(not isEndpointInit):
                        #             end = None
                        if (not isNemuBola):
                            # Berputar-putar sampai melihat bola
                            print('AKU BINGUNG BOLANYA DIMANA')
                            end = {}
                            end['x'] = bolaLastSeenX
                            end['y'] = bolaLastSeenY
                            realDistanceX = bolaLastSeenX
                            realDistanceY = bolaLastSeenY
                        else:
                            if (not isEndpointInit):
                                end = None
                else:
                    end = None
                # # start
                # start['x'] = myCoordLapanganX
                # start['y'] = myCoordLapanganY

                # print('isDribblingBola : ',isDribblingBola)
                # if(strategyState == 2 or strategyState==5):
                #     isBolaDekat = True
                # if (isDribblingBola):
                #     if (strategyState == 1):
                #         # strategyState = 2
                #         #langung ke gawang
                #         strategyState = 7
                #     elif (strategyState == 3):
                #         time.sleep(3)
                #         strategyState = 5
                #     elif (strategyState == 6):
                #         strategyState = 7
                #     isBolaDekat = True

                # paths = []
                # print('start',start)
                # print('matrix',matrix)
                # if(end is not None):
                #     print('start',start)
                #     print('end',end)
                #     startGridLoc = getGridLocationFromCoord(start,splitSizeGrid)
                #     endGridLoc = getGridLocationFromCoord(end,splitSizeGrid)
                #     if(startGridLoc[0]>8):
                #         startGridLoc[0] = 8
                #     if(startGridLoc[1]>11):
                #         startGridLoc[1] = 11
                #     if(endGridLoc[0]>8):
                #         endGridLoc[0] = 8
                #     if(endGridLoc[1]>11):
                #         endGridLoc[1] = 11
                #
                #     if (startGridLoc[0] < 0):
                #         startGridLoc[0] = 0
                #     if (startGridLoc[1] < 0):
                #         startGridLoc[1] = 0
                #     if (endGridLoc[0] < 0):
                #         endGridLoc[0] = 0
                #     if (endGridLoc[1] < 0):
                #         endGridLoc[1] = 0
                #     print('startGridLoc',startGridLoc)
                #     print('endGridLoc',endGridLoc)
                #     paths = findPathRobot(startGridLoc,matrix,endGridLoc)
                #
                # newCoordX = 0
                # newCoordY = 0
                # print('paths',paths)
                #
                # if(len(paths)>1 and not isBolaDekat):
                #     print('PAKE PATHFINDING')
                #     newCoordX = gridLapangan[paths[1][0]][paths[1][1]][0]
                #     newCoordY = gridLapangan[paths[1][0]][paths[1][1]][1]
                #
                #     # Iterate balikin lagi ke relatif
                #     rotationAngle = myGyro + gyroCalibration
                #     # X dan Y dibalik karena kamera bacanya kebalik
                #     print('SEBELUM KALIBRASI GYRO COORD X',newCoordX)
                #     print('SEBELUM KALIBRASI GYRO COORD Y',newCoordY)
                #     newCoordX, newCoordY = rotateMatrix(newCoordX, newCoordY,rotationAngle)
                #     print('SETELAH KALIBRASI GYRO COORD X',newCoordX)
                #     print('SETELAH KALIBRASI GYRO COORD Y',newCoordY)
                #
                #     if(myCoordX>newCoordX):
                #         newCoordX = myCoordX - newCoordX
                #     else:
                #         newCoordX = newCoordX - myCoordX
                #     if(myCoordY>newCoordY):
                #         newCoordY = myCoordY - newCoordY
                #     else:
                #         newCoordY = newCoordY - myCoordY
                #
                #     print('ROBOT AKAN PERGI KE ',paths[1])
                #     print('ROBOT AKAN PERGI KE REAL COORD X',newCoordX)
                #     print('ROBOT AKAN PERGI KE REAL COORD Y',newCoordY)
                # # elif(end is not None):
                # #     print('NYARI TANPA PATHFINDING BERDASARKAN END')
                # #     endGridLoc = getGridLocationFromCoord(end,splitSizeGrid)
                # #     newCoordX = gridLapangan[endGridLoc[0]][endGridLoc[1]][0]
                # #     newCoordY = gridLapangan[endGridLoc[0]][endGridLoc[1]][1]
                # #
                # #     # Iterate balikin lagi ke relatif
                # #     rotationAngle = myGyro + gyroCalibration
                # #     # X dan Y dibalik karena kamera bacanya kebalik
                # #     print('SEBELUM KALIBRASI GYRO COORD X',newCoordX)
                # #     print('SEBELUM KALIBRASI GYRO COORD Y',newCoordY)
                # #     newCoordX, newCoordY = rotateMatrix(newCoordX, newCoordY,rotationAngle)
                # #     print('SETELAH KALIBRASI GYRO COORD X',newCoordX)
                # #     print('SETELAH KALIBRASI GYRO COORD Y',newCoordY)
                # #
                # #     print('ROBOT AKAN PERGI KE REAL COORD X',newCoordX)
                # #     print('ROBOT AKAN PERGI KE REAL COORD Y',newCoordY)
                # else:
                #     print('LANGSUNG NYARI TANPA PATHFINDING BERDASARKAN YANG DILIHAT')
                #     newCoordX = real_distance_x
                #     newCoordY = realDistanceY
                #     print('ROBOT AKAN PERGI KE REAL COORD X',newCoordX)
                #     print('ROBOT AKAN PERGI KE REAL COORD Y',newCoordY)

                print('LANGSUNG NYARI TANPA PATHFINDING BERDASARKAN YANG DILIHAT')
                newCoordX = realDistanceX
                newCoordY = realDistanceY
                print('ROBOT AKAN PERGI KE REAL COORD X', newCoordX)
                print('ROBOT AKAN PERGI KE REAL COORD Y', newCoordY)
                # msg = "*0,1250,0#"

                # ser.open()
                print('is_kickoff', isKickOff)
                if (isKickOff):
                    if (isBolaDekat > 0):
                        isBolaDekat = 1
                    else:
                        isBolaDekat = 0
                    # TENDANG BOLA DAN JIKA SUDAH MENGHADAP GAWANG
                    if (isTendangBola):
                        isTendangBola = 1
                    else:
                        isTendangBola = 0
                    if (isTendangBola and (tetaBall < 10 and tetaBall > -10)):
                        isTendangBola = 1
                        # if (strategyState == 2):
                        #     strategyState = 3
                        # elif (strategyState == 5):
                        #     strategyState = 6
                    else:
                        isTendangBola = 0

                    msg = "*" + repr(newCoordX) + "," + repr(newCoordY) + "," + repr(tetaBall) + "," + repr(
                        isTendangBola) + "," + repr(isBolaDekat) + "," + repr(0) + "#"
                    print('msg for PID', msg)
                    ser.write(msg.encode())
                else:
                    # strategyState = 1
                    msg = "*0,0,0,0,0,1#"
                    ser.write(msg.encode())

                # pidRobot(tetaBall, newCoordX, newCoordY, ser)

                print('REAL LOCATION x : ', myCoordX, '  y :', myCoordY, 'tetaball', tetaBall)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)


def run_multithread():
    robot_object = Robot()
    all_field_objects = AllFielOjects()

    # t1 = threading.Thread(target=detect, args=(robot_object, all_field_objects,))
    t2 = threading.Thread(target=readSerialData, args=(robot_object, all_field_objects,))
    t3 = threading.Thread(target=send_data_to_base, args=(robot_object, all_field_objects,))
    t4 = threading.Thread(target=get_data_from_base, args=(robot_object, all_field_objects,))

    # t1.start()
    t2.start()
    t3.start()
    t4.start()

    # t1.join()
    t2.join()
    t3.join()
    t4.join()


if __name__ == '__main__':
    run_multithread()
