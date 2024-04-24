import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep1.yaml')
    model.train(data='data/widerface_filter_small.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=8,
                patience=50,
                optimizer='SGD', # using SGD
                project='runs/widerface',
                name='test',
                )