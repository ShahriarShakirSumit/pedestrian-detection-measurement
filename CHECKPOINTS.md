# Detector checkpoints

The 46 detectors, each identified by the checkpoint it was loaded from. Names match Table 7 of the paper. All are public COCO-pretrained releases and none is redistributed here. Torchvision weights download on first use. Ultralytics checkpoints are fetched by file name from the Ultralytics releases and placed in `$ULTRA_WEIGHTS`. Input is the size each detector is run at, which is its own released default.

| Family | Detector | Checkpoint | Input (px) |
|---|---|---|---|
| Anchor-free | FCOS R50-FPN | torchvision `FCOS_ResNet50_FPN_Weights.COCO_V1` | 800 |
| One-stage dense | RetinaNet R50-FPN v2 | torchvision `RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1` | 800 |
| One-stage dense | SSD300-VGG16 | torchvision `SSD300_VGG16_Weights.COCO_V1` | 300 |
| One-stage dense | SSDLite320-MNv3 | torchvision `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1` | 320 |
| Query-based | RT-DETR-L | Ultralytics `rtdetr-l.pt` | 640 |
| Query-based | RT-DETR-X | Ultralytics `rtdetr-x.pt` | 640 |
| Two-stage | Faster R-CNN R50-FPN v2 | torchvision `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` | 800 |
| YOLO series | YOLO11-l | Ultralytics `yolo11l.pt` | 640 |
| YOLO series | YOLO11-m | Ultralytics `yolo11m.pt` | 640 |
| YOLO series | YOLO11-n | Ultralytics `yolo11n.pt` | 640 |
| YOLO series | YOLO11-s | Ultralytics `yolo11s.pt` | 640 |
| YOLO series | YOLO11-x | Ultralytics `yolo11x.pt` | 640 |
| YOLO series | YOLO12-l | Ultralytics `yolo12l.pt` | 640 |
| YOLO series | YOLO12-m | Ultralytics `yolo12m.pt` | 640 |
| YOLO series | YOLO12-n | Ultralytics `yolo12n.pt` | 640 |
| YOLO series | YOLO12-s | Ultralytics `yolo12s.pt` | 640 |
| YOLO series | YOLO12-x | Ultralytics `yolo12x.pt` | 640 |
| YOLO series | YOLO26-l | Ultralytics `yolo26l.pt` | 640 |
| YOLO series | YOLO26-m | Ultralytics `yolo26m.pt` | 640 |
| YOLO series | YOLO26-n | Ultralytics `yolo26n.pt` | 640 |
| YOLO series | YOLO26-s | Ultralytics `yolo26s.pt` | 640 |
| YOLO series | YOLO26-x | Ultralytics `yolo26x.pt` | 640 |
| YOLO series | YOLOv10-b | Ultralytics `yolov10b.pt` | 640 |
| YOLO series | YOLOv10-l | Ultralytics `yolov10l.pt` | 640 |
| YOLO series | YOLOv10-m | Ultralytics `yolov10m.pt` | 640 |
| YOLO series | YOLOv10-n | Ultralytics `yolov10n.pt` | 640 |
| YOLO series | YOLOv10-s | Ultralytics `yolov10s.pt` | 640 |
| YOLO series | YOLOv10-x | Ultralytics `yolov10x.pt` | 640 |
| YOLO series | YOLOv3-SPP-u | Ultralytics `yolov3-sppu.pt` | 640 |
| YOLO series | YOLOv3-tiny-u | Ultralytics `yolov3-tinyu.pt` | 640 |
| YOLO series | YOLOv3-u | Ultralytics `yolov3u.pt` | 640 |
| YOLO series | YOLOv5-lu | Ultralytics `yolov5lu.pt` | 640 |
| YOLO series | YOLOv5-mu | Ultralytics `yolov5mu.pt` | 640 |
| YOLO series | YOLOv5-nu | Ultralytics `yolov5nu.pt` | 640 |
| YOLO series | YOLOv5-su | Ultralytics `yolov5su.pt` | 640 |
| YOLO series | YOLOv5-xu | Ultralytics `yolov5xu.pt` | 640 |
| YOLO series | YOLOv8-l | Ultralytics `yolov8l.pt` | 640 |
| YOLO series | YOLOv8-m | Ultralytics `yolov8m.pt` | 640 |
| YOLO series | YOLOv8-n | Ultralytics `yolov8n.pt` | 640 |
| YOLO series | YOLOv8-s | Ultralytics `yolov8s.pt` | 640 |
| YOLO series | YOLOv8-x | Ultralytics `yolov8x.pt` | 640 |
| YOLO series | YOLOv9-c | Ultralytics `yolov9c.pt` | 640 |
| YOLO series | YOLOv9-e | Ultralytics `yolov9e.pt` | 640 |
| YOLO series | YOLOv9-m | Ultralytics `yolov9m.pt` | 640 |
| YOLO series | YOLOv9-s | Ultralytics `yolov9s.pt` | 640 |
| YOLO series | YOLOv9-t | Ultralytics `yolov9t.pt` | 640 |
