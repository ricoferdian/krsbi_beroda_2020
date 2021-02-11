import argparse
import torch.backends.cudnn as cudnn

import serial
import socket
import threading

from communication.pc_to_base import send_data_to_base, get_data_from_base
from communication.pc_to_micro import read_serial_data, send_serial_data
from models.experimental import *
from objects.all_field_object import AllFielOjects
from objects.field_object import FieldObject
from objects.robot import Robot
from utils.datasets import *
from utils.utils import *
from findpath import *

cameraCenterX = 295
cameraCenterY = 248


def detect(robot_object: Robot, all_field_objects: AllFielOjects):
    out, source, weights, view_img, save_txt, imgsz = \
        'inference/output', '1', 'D:\\Libraries\\Project\\Python\\yolov5\\runs\\exp8\\weights\\best.pt', None, None, 640
    webcam = source == '0' or source == '1' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')

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

    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)

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

                    decision_making(robot_object, all_field_objects, is_dribbling, )


                # Cek sedang kickoff atau tidak
                if (robot_object.get_kickoff()):
                    # Get destination dari robot object
                    dest_x, dest_y, dest_teta, is_tendang, is_bola_dekat, is_reset = robot_object.get_destination()
                    send_serial_data(dest_x, dest_y, dest_teta, is_tendang, is_bola_dekat, is_reset)
                else:
                    msg = "*0,0,0,0,0,1#"
                    send_serial_data(0, 0, 0, 0, 0, 1)

            # show inference
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration


def run_multithread():
    robot_object = Robot()
    all_field_objects = AllFielOjects()

    t1 = threading.Thread(target=detect, args=(robot_object, all_field_objects,))
    t2 = threading.Thread(target=read_serial_data, args=(robot_object, all_field_objects,))
    t3 = threading.Thread(target=send_data_to_base, args=(robot_object, all_field_objects,))
    t4 = threading.Thread(target=get_data_from_base, args=(robot_object, all_field_objects,))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()


if __name__ == '__main__':
    run_multithread()
