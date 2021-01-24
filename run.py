import torch.backends.cudnn as cudnn

from getcoord import getcoordinate

def run():
    out, source, weights, view_img, save_txt, imgsz = \
        'inference/output', '1', 'D:\\Libraries\\Project\\Python\\yolov5\\runs\\exp8\\weights\\best.pt', None, None, 640
    webcam = source == '0' or source == '1' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')