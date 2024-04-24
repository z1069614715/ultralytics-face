# Traing Parameter

### yolov8n-pose baseline

    model = YOLO('ultralytics/cfg/models/v8/yolov8n-pose.yaml')
    model.load('yolov8n-pose.pt') # loading pretrain weights
    model.train(data='data/widerface.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=8,
                patience=50,
                optimizer='SGD', # using SGD
                project='runs/widerface',
                name='yolov8n-face-baseline',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-baseline.log 2>&1 & tail -f logs/yolov8n-face-baseline.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-baseline/weights/last.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8

    ==================== Results ====================
    Easy   Val AP: 0.943678686359784
    Medium Val AP: 0.9187094157374914
    Hard   Val AP: 0.7745994664864935
    =================================================

### yolov8n-pose baseline no-pretrain

    model = YOLO('ultralytics/cfg/models/v8/yolov8n-pose.yaml')
    model.train(data='data/widerface.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=8,
                patience=50,
                optimizer='SGD', # using SGD
                project='runs/widerface',
                name='yolov8n-face-baseline-nopretrain',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-baseline-nopretrain.log 2>&1 & tail -f logs/yolov8n-face-baseline-nopretrain.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/widerface/yolov8n-face-baseline-nopretrain/weights/last.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-baseline-nopretrain/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9357554288022945
    Medium Val AP: 0.912094569248842
    Hard   Val AP: 0.7758193086868959
    =================================================

### yolov8n-pose filter x pixel lowprecision object in 640 images-size

    model = YOLO('ultralytics/cfg/models/v8/yolov8n-pose.yaml')
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
                name='yolov8n-face-filter-small-object',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object5.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object5.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8

    # 9 pixel     image_size, small_thresh, iou_thresh = 640, 9, 0.5
    ==================== Results ====================
    Easy   Val AP: 0.9425160034901028
    Medium Val AP: 0.922188577153425
    Hard   Val AP: 0.7580036635703826
    =================================================

    # 7 pixel     image_size, small_thresh, iou_thresh = 640, 7, 0.5
    ==================== Results ====================
    Easy   Val AP: 0.9408933867266708
    Medium Val AP: 0.9182572457765938
    Hard   Val AP: 0.7684845717003413
    =================================================

    # 6 pixel     image_size, small_thresh, iou_thresh = 640, 6, 0.5
    ==================== Results ====================
    Easy   Val AP: 0.9392037916294687
    Medium Val AP: 0.9179844989951516
    Hard   Val AP: 0.7743407582025861
    =================================================

    # 5 pixel     image_size, small_thresh, iou_thresh = 640, 5, 0.5
    ==================== Results ====================
    Easy   Val AP: 0.9383685477522108
    Medium Val AP: 0.9169525915250611
    Hard   Val AP: 0.7794487291942509
    =================================================

    # 4 pixel      image_size, small_thresh, iou_thresh = 640, 4, 0.5
    ==================== Results ====================
    Easy   Val AP: 0.9341455226581391
    Medium Val AP: 0.9111548287057303
    Hard   Val AP: 0.7745843715424623
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + FaceRandomCrop

    model = YOLO('ultralytics/cfg/models/v8/yolov8n-pose.yaml')
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
                name='yolov8n-face-filter-small-object-FaceRandomCrop',
                )
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-FaceRandomCrop-5.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-FaceRandomCrop-5.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-FaceRandomCrop5/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    
    # FaceRandomCrop(max_crop_ratio=0.5, p=0.2)
    ==================== Results ====================
    Easy   Val AP: 0.9340345080128105
    Medium Val AP: 0.9088230591031317
    Hard   Val AP: 0.7644932298164093
    =================================================

    # FaceRandomCrop(max_crop_ratio=0.2, p=0.5)
    ==================== Results ====================
    Easy   Val AP: 0.9294032908095071
    Medium Val AP: 0.9024018857067466
    Hard   Val AP: 0.7410233830985312
    =================================================
    
    # FaceRandomCrop(max_crop_ratio=0.2, p=0.2)
    ==================== Results ====================
    Easy   Val AP: 0.9328363222763076
    Medium Val AP: 0.9093156120152489
    Hard   Val AP: 0.7659403333843626
    =================================================

    # FaceRandomCrop(max_crop_ratio=0.1, p=0.2)
    ==================== Results ====================
    Easy   Val AP: 0.9347491736828661
    Medium Val AP: 0.9101764295848018
    Hard   Val AP: 0.7669252759793007
    =================================================

    # FaceRandomCrop(max_crop_ratio=0.1, p=0.1)
    ==================== Results ====================
    Easy   Val AP: 0.9357439003883163
    Medium Val AP: 0.9148017297714196
    Hard   Val AP: 0.7730805888976859
    =================================================


### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL

    model = YOLO('ultralytics/cfg/models/v8/yolov8n-pose.yaml')
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
                name='yolov8n-face-filter-small-object-TAL',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-TAL-4.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-TAL-4.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-TAL4/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8

    # TaskAlignedAssigner(topk=7, num_classes=self.nc, alpha=0.5, beta=6.0)
    ==================== Results ====================
    Easy   Val AP: 0.9337907174770645
    Medium Val AP: 0.914811616615522
    Hard   Val AP: 0.7820088430693088
    =================================================

    # TaskAlignedAssigner(topk=5, num_classes=self.nc, alpha=0.5, beta=6.0)
    ==================== Results ====================
    Easy   Val AP: 0.9349248542040435
    Medium Val AP: 0.9168321887811575
    Hard   Val AP: 0.7865576698425549
    =================================================

    # TaskAlignedAssigner(topk=3, num_classes=self.nc, alpha=0.5, beta=6.0)
    ==================== Results ====================
    Easy   Val AP: 0.9308420340982722
    Medium Val AP: 0.9152875519744202
    Hard   Val AP: 0.7872064919703559
    =================================================

    # TaskAlignedAssigner(topk=3, num_classes=self.nc, alpha=0.5, beta=9.0)
    ==================== Results ====================
    Easy   Val AP: 0.9256510306305094
    Medium Val AP: 0.907209934965147
    Hard   Val AP: 0.7887162574525065
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + sppf-3

    model = YOLO('ultralytics/cfg/models/v8/yolov8n-pose-sppf-3.yaml')
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
                name='yolov8n-face-filter-small-object-sppf-3',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-sppf-3.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-sppf-3.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-sppf-3/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-sppf-3/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9322859487121787
    Medium Val AP: 0.9147386945529493
    Hard   Val AP: 0.7872254343617164
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6.yaml')
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
                name='yolov8n-face-filter-small-object-p6',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9355275260313918
    Medium Val AP: 0.9190118329407484
    Hard   Val AP: 0.7891111955535621
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6-C2f

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-C2f.yaml')
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
                name='yolov8n-face-filter-small-object-p6-C2f',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-C2f.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-C2f.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-C2f/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-C2f/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9333051410243601
    Medium Val AP: 0.9155042175860136
    Hard   Val AP: 0.7856901170745433
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + ADown

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-ADown.yaml')
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
                name='yolov8n-face-filter-small-object-p6-ADown',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-adown 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-adown
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-ADown/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-ADown/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9336307684774567
    Medium Val AP: 0.9150758849612703
    Hard   Val AP: 0.7704155402613113
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + V7Down

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-V7Down.yaml')
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
                name='yolov8n-face-filter-small-object-p6-V7Down',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-v7down.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-v7down.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-V7Down/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-V7Down/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9354881085278206
    Medium Val AP: 0.9183207035388947
    Hard   Val AP: 0.7886194672127571
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9412578200706179
    Medium Val AP: 0.9283130060394659
    Hard   Val AP: 0.8100328894210891
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + LSCD

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-LSCD.yaml')
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
                name='yolov8n-face-filter-small-object-p6-LSCD',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-LSCD.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-LSCD.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-LSCD/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-LSCD/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9368110776925447
    Medium Val AP: 0.9195816086070555
    Hard   Val AP: 0.7886834195267196
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9418271455988033
    Medium Val AP: 0.9277675495505393
    Hard   Val AP: 0.8074121227491434
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9419754472002038
    Medium Val AP: 0.9289276822471543
    Hard   Val AP: 0.8114986117385863
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9433160891530674
    Medium Val AP: 0.9290921984906393
    Hard   Val AP: 0.8138169523694521
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep2

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep2.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep2',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep2.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep2.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep2/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep2/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9408516037267558
    Medium Val AP: 0.9252712864711324
    Hard   Val AP: 0.8122676179924786
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2f-Rep1

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep3.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep3',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep3.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep3.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep3/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep3/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.939846211901032
    Medium Val AP: 0.9259344936643292
    Hard   Val AP: 0.8140264599860008
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2f-Rep2

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep4.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep4',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep4.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep4.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep4/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep4/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9406601626930166
    Medium Val AP: 0.9273828515670495
    Hard   Val AP: 0.8131337352513474
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + EMBC

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep1-EMBC.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EMBC',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EMBC.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EMBC.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EMBC/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EMBC/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9394848987291206
    Medium Val AP: 0.926233831699091
    Hard   Val AP: 0.8126722406318776
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + Faster

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep1-Faster.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-Faster',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-Faster.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-Faster.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-Faster/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-Faster/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9345014790651598
    Medium Val AP: 0.9221657647425019
    Hard   Val AP: 0.8080684245270735
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + DWR

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep1-DWR.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DWR',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DWR.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DWR.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DWR/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DWR/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9394352865023253
    Medium Val AP: 0.9244816995249258
    Hard   Val AP: 0.8129908412221023
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + RVB

    model = YOLO('ultralytics/cfg/models/v8/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep1-RVB.yaml')
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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-RVB',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-RVB.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-RVB.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-RVB/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-RVB/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9378505784693182
    Medium Val AP: 0.9241313419381917
    Hard   Val AP: 0.8077182638272813
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + SlideLoss

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SlideLoss',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SlideLoss.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SlideLoss.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SlideLoss/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SlideLoss/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.940715344792219
    Medium Val AP: 0.9271731494217756
    Hard   Val AP: 0.8148772783749215
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + NWD

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-NWD',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-NWD-4.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-NWD-4.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-NWD4/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-NWD/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    # loss_iou = loss_iou * 0.5 +  nwd_loss * 0.5 constant=1
    ==================== Results ====================
    Easy   Val AP: 0.9397042017735961
    Medium Val AP: 0.9253030190745148
    Hard   Val AP: 0.8099157584178454
    =================================================

    # loss_iou = loss_iou * 0.7 +  nwd_loss * 0.3 constant=1
    ==================== Results ====================
    Easy   Val AP: 0.9406092379560294
    Medium Val AP: 0.9254115876516003
    Hard   Val AP: 0.8112080844776342
    =================================================

    # loss_iou = loss_iou * 0.5 +  nwd_loss * 0.5 constant=24.4
    ==================== Results ====================
    Easy   Val AP: 0.9420547336968625
    Medium Val AP: 0.9286762424528503
    Hard   Val AP: 0.8129085779476847
    =================================================

    # loss_iou = loss_iou * 0.7 +  nwd_loss * 0.3 constant=24.4
    ==================== Results ====================
    Easy   Val AP: 0.94135936450502
    Medium Val AP: 0.9273645226671535
    Hard   Val AP: 0.8134814005305742
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + Inner-CIoU

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-ICIOU',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-ICIOU-5.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-ICIOU-5.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-ICIOU5/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-ICIOU/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    # ratio=1.1
    ==================== Results ====================
    Easy   Val AP: 0.9343282928901158
    Medium Val AP: 0.9225116258067672
    Hard   Val AP: 0.8102031414037021
    =================================================

    # ratio=1.2
    ==================== Results ====================
    Easy   Val AP: 0.9416813212109524
    Medium Val AP: 0.927365835672002
    Hard   Val AP: 0.8149905688241516
    =================================================

    # ratio=1.25
    ==================== Results ====================
    Easy   Val AP: 0.9395995378097283
    Medium Val AP: 0.9263771264726179
    Hard   Val AP: 0.8114411097562714
    =================================================

    # ratio=1.3
    ==================== Results ====================
    Easy   Val AP: 0.9415624990171303
    Medium Val AP: 0.9278672888915376
    Hard   Val AP: 0.8140773386689295
    =================================================

    # ratio=1.4
    ==================== Results ====================
    Easy   Val AP: 0.9406602558331447
    Medium Val AP: 0.9264259047094887
    Hard   Val AP: 0.813158773299653
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + EIoU

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EIOU',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EIOU.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EIOU.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EIOU/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-EIOU/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.938925322068154
    Medium Val AP: 0.9252590814055168
    Hard   Val AP: 0.8141028637058665
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + DIoU

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DIoU',
                )
    
    CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DIoU.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DIoU.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DIoU/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-DIoU/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9404474113281884
    Medium Val AP: 0.926322152304119
    Hard   Val AP: 0.8130585201896894
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + SIoU

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SIoU',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SIoU.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SIoU.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SIoU/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-SIoU/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9351575678655131
    Medium Val AP: 0.9230559384871734
    Hard   Val AP: 0.8136913168712472
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + MPDIoU

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
                name='yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-MPDIOU',
                )
    
    CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-MPDIOU.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-MPDIOU.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-MPDIOU/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-MPDIOU/weights/last.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9399200558110872
    Medium Val AP: 0.9266252161682317
    Hard   Val AP: 0.8127270740129233
    =================================================

### yolov8n-pose filter 5 pixel lowprecision object in 640 images-size + TAL + P6 + HGStem + LSCD + BIFPN + C2-Rep1 + LAMP

    param_dict = {
        # origin
        'model': 'runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1/weights/best.pt',
        'data':'data/widerface_filter_small.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 16,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp1',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 2.0,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }

    CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp1.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/prune/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp1-finetune/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/prune/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp1-finetune/weights/best.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9194412080266768
    Medium Val AP: 0.9071539007783896
    Hard   Val AP: 0.79200399592807
    =================================================

    param_dict = {
        # origin
        'model': 'runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1/weights/best.pt',
        'data':'data/widerface_filter_small.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 16,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp2',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 2.5,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }

    CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp2.log
    CUDA_VISIBLE_DEVICES=1 python test_widerface.py --weights runs/prune/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp2-finetune/weights/best.pt --dataset_folder /root/data_ssd/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/prune/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp2-finetune/weights/best.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9044631175948342
    Medium Val AP: 0.8918939752260204
    Hard   Val AP: 0.7741422413815722
    =================================================

    param_dict = {
        # origin
        'model': 'runs/widerface/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1/weights/best.pt',
        'data':'data/widerface_filter_small.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 16,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp3',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 1.5,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }

    CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp3.log 2>&1 & tail -f logs/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp3.log
    CUDA_VISIBLE_DEVICES=0 python test_widerface.py --weights runs/prune/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp3-finetune/weights/best.pt --dataset_folder /home/hjj/Desktop/dataset/WIDER-FACE/WIDER_val/images --num_workers 8
    python get_inference_time.py --weights runs/prune/yolov8n-face-filter-small-object-p6-HGStem-LSCD-BIFPN-Rep1-lamp-exp3-finetune/weights/best.pt --warmup 100 --testtime 300 --batch 32 --device 0

    ==================== Results ====================
    Easy   Val AP: 0.9366055056967946
    Medium Val AP: 0.9220540638848397
    Hard   Val AP: 0.808854208767393
    =================================================