import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('runs/widerface/yolov8n-face-baseline/weights/best.pt')
    # model.predict(source='/root/data_ssd/WIDER-FACE/train',
    #              imgsz=640,
    #              batch=16,
    #              save_txt=True,
    #              save_conf=True,
    #              conf=0.01,
    #              save=False,
    #              show_labels=False,
    #              project='runs/widerface-detect',
    #              name='yolov8n-face-baseline',
    #              )
    
    # model = YOLO('yolov8n-pose-face-baseline.pt')
    # model.predict(source='data/test.jpg',
    #              imgsz=1280,
    #              conf=0.25,
    #              save=True,
    #              show_labels=False,
    #              max_det=1000,
    #              project='runs/widerface-detect',
    #              name='test',
    #              )
    
    model = YOLO('yolov8n-pose-face-baseline.pt')
    model.predict(source='video.mp4',
                 imgsz=640,
                 conf=0.25,
                 save=True,
                 device='cpu',
                 project='runs/widerface-detect',
                 name='video-test',
                 )