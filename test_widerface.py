import os, tqdm, shutil
import argparse
from ultralytics import YOLO
from multiprocessing import Pool
from typing import Callable, Dict, List, Union
from widerface_evaluate.evaluation import evaluation

def parallelise(function: Callable, data: List, chunksize=100, verbose=True, num_workers=os.cpu_count()) -> List:
    num_workers = 1 if num_workers < 1 else num_workers  # Pool needs to have at least 1 worker.
    pool = Pool(processes=num_workers)
    results = list(
        tqdm.tqdm(pool.imap(function, data, chunksize), total=len(data), disable=not verbose)
    )
    pool.close()
    pool.join()
    return results

def inference(img_name):
    image_path = os.path.join(testset_folder, img_name)
    results = model.predict(source=image_path, stream=True, imgsz=opt.img_size, conf=opt.conf_thres, iou=opt.iou_thres, augment=opt.augment, device=opt.device)

    save_name = opt.save_folder + img_name[:-4] + ".txt"
    dirname = os.path.dirname(save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(save_name, "w") as fd:
        result = next(results).cpu().numpy()
        file_name = os.path.basename(save_name)[:-4] + "\n"
        bboxs_num = str(result.boxes.shape[0]) + '\n'
        fd.write(file_name)
        fd.write(bboxs_num)
        for box in result.boxes:
            conf = box.conf[0]
            cls  = box.cls[0]
            xyxy = box.xyxy[0]
            x1 = int(xyxy[0] + 0.5)
            y1 = int(xyxy[1] + 0.5)
            x2 = int(xyxy[2] + 0.5)
            y2 = int(xyxy[3] + 0.5)
            fd.write('%d %d %d %d %.03f' % (x1, y1, x2-x1, y2-y1, conf if conf <= 1 else 1) + '\n')
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/pose/yolov8n-face/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='', help='augmented inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers for inference')
    parser.add_argument('-p', '--pred', default="widerface_evaluate/widerface_txt/")
    parser.add_argument('-g', '--gt', default='widerface_evaluate/ground_truth/')
    opt = parser.parse_args()
    print(opt)

    if os.path.exists(opt.save_folder):
        shutil.rmtree(opt.save_folder)
    os.makedirs(opt.save_folder)
    
    model = YOLO(opt.weights)

    # testing dataset
    testset_folder = opt.dataset_folder
    test_dataset = []
    for i in os.listdir(testset_folder):
        base_path = os.listdir(f'{testset_folder}/{i}')
        for j in base_path:
            test_dataset.append(f'{i}/{j}')
    
    result = parallelise(inference, test_dataset, chunksize=100, num_workers=opt.num_workers)
    evaluation(opt.pred, opt.gt)