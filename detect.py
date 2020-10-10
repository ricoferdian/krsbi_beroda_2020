import argparse
import torch.backends.cudnn as cudnn
from getcoord import getcoordinate

import serial
import socket
import threading
import time

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from findpath import *
from modulPid import pidRobot

global myCoordX
global myCoordY
global myCoordLapanganX
global myCoordLapanganY
global bolaLastSeenX
global bolaLastSeenY
# global myRes
global myGyro

#Strategi dan base station value
global isKickOff
global strategyState
global isDribblingBola

#Serial port Arduino
global ser

#Gyro calibration sesuaikan dengan sudut gyro saat menghadap gawang
gyroCalibration = 0

HOST = '192.168.43.61'
# LAPTOP UCUP
# PORT = 28097
# arrayStrategy = [0,1,2,3,0,5,6,7,0,0]
# robotId = 1
# LAPTOP DEK JUN
PORT = 5204
arrayStrategy = [0,3,2,0,0,5,0,0,0,1]
robotId = 2

networkserial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
networkserial.connect((HOST, PORT))

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        'inference/output', '1', 'D:\\Libraries\\Project\\Python\\yolov5\\runs\\exp8\\weights\\best.pt', None, None, 640
    webcam = source == '0' or source == '1' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')

    maxXLapangan = 450
    maxYLapangan = 600
    splitSizeGrid = 50

    global isKickOff
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

    if(myCoordX is not None):
        myCoordLapanganX = myCoordX
    else:
        myCoordLapanganX = 0
    if(myCoordY is not None):
        myCoordLapanganY = myCoordY
    else:
        myCoordLapanganY = 0
    myRes = 0
    myGyro = 0

    gridLapangan = gridGenerator(maxXLapangan,maxYLapangan,splitSizeGrid)

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

                # Write results
                arr_objects = []
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    arr_objects.append({'label': names[int(cls)],'conf': float(conf),
                                        'x1': int(xyxy[0]), 'y1': int(xyxy[1]),
                                        'x2': int(xyxy[2]), 'y2': int(xyxy[3])})
                arr_objects = getcoordinate(arr_objects)
                # print('arr_objects',arr_objects)
                isNemuBola = False
                start = {}
                end = {}

                #kalibrasi gyro
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

                if(len(arr_objects)>0):
                    for object in arr_objects:
                        #Iterate object dan definisikan lokasinya di lapangan
                        rotationAngle = myGyro - gyroCalibration
                        #X dan Y dibalik karena kamera bacanya kebalik
                        object['x'],object['y'] = rotateMatrix(object['realDistanceY'],object['realDistanceX'],rotationAngle)
                        # object['gridReal'] = getGridLocationFromCoord(object,splitSizeGrid)

                        # print('object location real',object)

                        if(object['label']=='bola'):
                            isNemuBola = True
                            if(not isDribblingBola and (strategyState==arrayStrategy[1] or strategyState==arrayStrategy[6])):
                                print('AKU NYARI BOLA', object)
                                isNyariBola = True
                                isTendangBola = False
                                end = {}
                                end['x'] = object['x']
                                end['y'] = object['y']
                                bolaLastSeenX = object['x']
                                bolaLastSeenY = object['y']
                                tetaBall = object['tetaObj']
                                realDistanceX = object['realDistanceX']
                                realDistanceY = object['realDistanceY']
                                if(not isEndpointInit):
                                    isEndpointInit = True
                                if(realDistanceY<160):
                                    print('BOLA SUDAH DEKAT')
                                    isBolaDekat = True
                            else:
                                if(not isEndpointInit):
                                    end = None
                                    isEndpointInit = True
                        elif (object['label']=='gawang'):
                            if (strategyState==arrayStrategy[3] or strategyState==arrayStrategy[7]):
                                #Cari gawang
                                print('AKU NYARI GAWANG', object)
                                end = {}
                                end['x'] = object['x']
                                end['y'] = object['y']
                                isEndpointInit = True
                                tetaBall = object['tetaObj']
                                realDistanceX = object['realDistanceX']
                                realDistanceY = object['realDistanceY']
                                #JIKA GAWANG DEKAT, TENDANG
                                if(isDribblingBola and realDistanceY<400):
                                    isTendangBola = True
                                elif(realDistanceY<400):
                                    realDistanceY = 0
                                    realDistanceX = 0
                                elif(strategyState==arrayStrategy[7] and not isDribblingBola):
                                    strategyState = 6

                            else:
                                if(not isEndpointInit):
                                    end = None
                                    isEndpointInit = True
                        elif (object['label'] == 'robot'):
                            if (isDribblingBola or (strategyState==arrayStrategy[2] or strategyState==arrayStrategy[5] or strategyState==arrayStrategy[9])):
                                # Cari gawang
                                print('AKU NYARI TEMENKU DIMANA', object)
                                end = {}
                                end['x'] = object['x']
                                end['y'] = object['y']
                                isEndpointInit = True
                                tetaBall = object['tetaObj']
                                realDistanceX = object['realDistanceX']
                                realDistanceY = object['realDistanceY']
                                #JIKA ROBOT DEKAT, TENDANG
                                if(not isDribblingBola):
                                    realDistanceY = 0
                                    realDistanceX = 0
                                if(isDribblingBola and realDistanceY<300):
                                    realDistanceY = 0
                                    isTendangBola = True
                            else:
                                if (not isEndpointInit):
                                    end = None
                                    isEndpointInit = True
                        elif(object['label'] == 'obstacle'):
                            print('ADA OBSTACLE', object)
                            obstacle = {}
                            obstacle['x'] = object['x']
                            obstacle['y'] = object['y']
                            obstacleGridLoc = getGridLocationFromCoord(obstacle,splitSizeGrid)
                            if(obstacleGridLoc[0]>11):
                                obstacleGridLoc[0] = 11
                            elif(obstacleGridLoc[0]<0):
                                obstacleGridLoc[0] = 0
                            if(obstacleGridLoc[1]>8):
                                obstacleGridLoc[1] = 8
                            elif(obstacleGridLoc[1]<0):
                                obstacleGridLoc[1] = 0

                            # print("obstacleGridLoc",obstacleGridLoc)
                            matrix[obstacleGridLoc[1]][obstacleGridLoc[0]] = 0
                            if(not isNemuBola or not isDribblingBola):
                                if(not isEndpointInit):
                                    end = None
                        if(not isNemuBola):
                            #Berputar-putar sampai melihat bola
                            print('AKU BINGUNG BOLANYA DIMANA')
                            end = {}
                            end['x'] = bolaLastSeenX
                            end['y'] = bolaLastSeenY
                        else:
                            if(not isEndpointInit):
                                end = None
                else:
                    end = None

                start['x'] = myCoordLapanganX
                start['y'] = myCoordLapanganY

                print('isDribblingBola : ',isDribblingBola)
                if(strategyState == 2 or strategyState==5):
                    isBolaDekat = True
                if (isDribblingBola):
                    if (strategyState == 1):
                        strategyState = 2
                    elif (strategyState == 3):
                        time.sleep(3)
                        strategyState = 5
                    elif (strategyState == 6):
                        strategyState = 7
                    isBolaDekat = True

                paths = []
                print('start',start)
                print('matrix',matrix)
                if(end is not None):
                    print('start',start)
                    print('end',end)
                    startGridLoc = getGridLocationFromCoord(start,splitSizeGrid)
                    endGridLoc = getGridLocationFromCoord(end,splitSizeGrid)
                    if(startGridLoc[0]>8):
                        startGridLoc[0] = 8
                    if(startGridLoc[1]>11):
                        startGridLoc[1] = 11
                    if(endGridLoc[0]>8):
                        endGridLoc[0] = 8
                    if(endGridLoc[1]>11):
                        endGridLoc[1] = 11

                    if (startGridLoc[0] < 0):
                        startGridLoc[0] = 0
                    if (startGridLoc[1] < 0):
                        startGridLoc[1] = 0
                    if (endGridLoc[0] < 0):
                        endGridLoc[0] = 0
                    if (endGridLoc[1] < 0):
                        endGridLoc[1] = 0
                    print('startGridLoc',startGridLoc)
                    print('endGridLoc',endGridLoc)
                    paths = findPathRobot(startGridLoc,matrix,endGridLoc)

                newCoordX = 0
                newCoordY = 0
                print('paths',paths)
                # if(len(paths)>1):
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
                # elif(end is not None):
                #     print('NYARI TANPA PATHFINDING BERDASARKAN END')
                #     endGridLoc = getGridLocationFromCoord(end,splitSizeGrid)
                #     newCoordX = gridLapangan[endGridLoc[0]][endGridLoc[1]][0]
                #     newCoordY = gridLapangan[endGridLoc[0]][endGridLoc[1]][1]
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
                #     print('ROBOT AKAN PERGI KE REAL COORD X',newCoordX)
                #     print('ROBOT AKAN PERGI KE REAL COORD Y',newCoordY)
                # else:
                #     print('LANGSUNG NYARI TANPA PATHFINDING BERDASARKAN YANG DILIHAT')
                #     newCoordX = realDistanceX
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
                print('isKickOff',isKickOff)
                if(isKickOff):
                    if(isBolaDekat):
                        isBolaDekat = 1
                    else:
                        isBolaDekat = 0
                    #TENDANG BOLA DAN JIKA SUDAH MENGHADAP GAWANG
                    if(isTendangBola):
                        isTendangBola = 1
                    else:
                        isTendangBola = 0
                    if(isTendangBola and (tetaBall<10 and tetaBall>-10)):
                        isTendangBola = 1
                        if (strategyState == 2):
                            strategyState = 3
                        elif (strategyState == 5):
                            strategyState = 6
                    else:
                        isTendangBola = 0

                    # if (robotId == 1):
                    #     if (strategyState == 1):
                    #         if (myCoordX < 180):
                    #             if (newCoordX > 20 and newCoordX < -20):
                    #                 newCoordY = 0
                    #                 tetaBall = myGyro
                    #     elif (strategyState == 3):
                    #         if (newCoordX > 20 and newCoordX < -20):
                    #             newCoordY = 0
                    #             newCoordX = 180
                    #             tetaBall = -myGyro
                    #     # elif(strategyState==3):
                    #     #     if(myCoordX>80):
                    #     #         newCoordY = 0
                    #     #         newCoordX = -100
                    #     #         tetaBall = -myGyro
                    # else:
                    #     if (strategyState == 1):
                    #         if (myCoordX > -180):
                    #             if (newCoordX > 20 and newCoordX < -20):
                    #                 newCoordY = 0
                    #                 tetaBall = -myGyro
                    #     elif (strategyState == 6):
                    #         if (newCoordX > 20 and newCoordX < -20):
                    #             newCoordY = 0
                    #             newCoordX = -180
                    #             tetaBall = -myGyro
                    #         # else:
                    #         #     if(myCoordY<100):
                    #         #         newCoordY = 100
                    #         #         newCoordX = 0
                    #     # elif(strategyState==6):
                    #     #     if(myCoordX<0):
                    #     #         newCoordY = 0
                    #     #         newCoordX = 180
                    #     #         tetaBall = -myGyro

                    msg = "*" + repr(newCoordX) + "," + repr(newCoordY) + "," + repr(tetaBall) +"," + repr(isTendangBola) + "," + repr(isBolaDekat)+ "," + repr(0) + "#"
                    print('msg for PID', msg)
                    ser.write(msg.encode())
                else:
                    strategyState = 1
                    msg = "*0,0,0,0,0,1#"
                    ser.write(msg.encode())

                # pidRobot(tetaBall, newCoordX, newCoordY, ser)

                print('REAL LOCATION x : ',myCoordX,'  y :',myCoordY, 'tetaball',tetaBall)


            # Process if no object detected
            else:
                print('no object detected')
                print('names', names)
                print('pred', pred)

            # print('myCoordX', myCoordX)
            # print('myCoordY', myCoordY)
            # print('myRes', myRes)
            # print('myGyro', myGyro)

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

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

def perintahRobot(command):
    global isKickOff
    if(command=='K'):
        isKickOff = True
    if(command=='r'):
        isKickOff = False
    elif(command[0]=='*'):
        parseCommand(command)

def parseCommand(command):
    global strategyState
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
    newStrategyState = xystrategy[7]
    if(newStrategyState>strategyState):
        strategyState = newStrategyState

def updateBaseData():
    global myCoordX
    global myCoordY
    global myCoordLapanganX
    global myCoordLapanganY
    global bolaLastSeenX
    global bolaLastSeenY
    global myGyro
    global strategyState

    myCoordLapanganX = 0
    myCoordLapanganY = 0
    bolaLastSeenX = 0
    bolaLastSeenY = 0
    myGyro = 0
    strategyState = 1

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
    time.sleep(1)
    msg = "*"+repr(x1)+","+repr(y1)+","+repr(teta1)+","+repr(bolaX)+","+repr(bolaY)+","+repr(strategyStatus)+"#"
    print('DATA SENT TO BASE : ',msg)
    networkserial.send(msg.encode())

def receiveDataFromBase(xRobot2, yRobot2, tetaRobot2):
    data = networkserial.recv(4096)
    data = data.decode("utf-8")
    print('data DARI BASE STATION',data)
    if(data):
        perintahRobot(data)

def readSerialData():
    sendSerialMode = True
    sendSocketMode = False
    # Create a socket object
    s = socket.socket()

    # Define the port on which you want to connect
    port = 12345

    _serDeclare = True
    readdata = ''
    xyresgyro = [0 for i in range(4)]
    i = 0

    global ser
    #robot 2
    # ser = serial.Serial('COM3', 115200, timeout=100000)
    #robot 1
    ser = serial.Serial('COM3', 115200, timeout=100000)

    global myCoordX
    global myCoordY
    myCoordX = 0
    myCoordY = 0

    global myCoordLapanganX
    global myCoordLapanganY
    global isDribblingBola
    myCoordLapanganX = 0
    myCoordLapanganY = 0
    isDribblingBola = False

    global myGyro
    myGyro = 0
    global isKickOff
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
                    dribblingBola = xyresgyro[3]
                    if(dribblingBola==1):
                        isDribblingBola = True
                    else:
                        isDribblingBola = False

                    print('isDribblingBola : ', isDribblingBola)
                    #myCoordY = depan robot. Terkalibrasi sebagai x positif di lapangan (menghadap gawang = 0 derajat)
                    #myCoordX = kanan robot. Terkalibrasi sebagai y positif di lapangan (menghadap kanan gawang = 90 derajat)
                    robotCoordX, robotCoordY = rotateMatrix(myCoordY,myCoordX,myGyro - gyroCalibration)
                    # print('myCoordX : ',myCoordX,' myCoordY : ',myCoordY)
                    print('robotCoordX : ',robotCoordX,' robotCoordY : ',robotCoordY)
                    if(robotCoordX<0):
                        myCoordLapanganX += robotCoordX - myCoordLapanganX
                    elif(robotCoordX>0):
                        myCoordLapanganX += robotCoordX - myCoordLapanganX
                    else:
                        myCoordLapanganX = 0
                    if(robotCoordY<0):
                        myCoordLapanganY += robotCoordY - myCoordLapanganY
                    elif(robotCoordX>0):
                        myCoordLapanganY += robotCoordY - myCoordLapanganY
                    else:
                        myCoordLapanganY = 0
                    print('myCoordLapanganX : ',myCoordLapanganX,' myCoordLapanganY : ',myCoordLapanganY)

                else:
                    readdata += strmsg
                # print ('strmsg:'+strmsg)

        elif sendSerialMode == False:
            ser.close()

        if sendSocketMode == True:
            print(s.recv(1024))
            s.send('kontrol')

def runMultiThread():
    t1 = threading.Thread(target=detect)
    t2 = threading.Thread(target=readSerialData)
    t3 = threading.Thread(target=updateBaseData)
    t4 = threading.Thread(target=updateLocalDataFromBase)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                runMultiThread()
                strip_optimizer(opt.weights)
        else:
            runMultiThread()
