"""Parameter count and multiply-accumulate cost for every detector.

Latency is a property of one machine, whereas the FLOP count is not. It is
reported beside the timing so that readers on other hardware can use it, and
so that the relationship between compute and measured latency can be examined
rather than assumed.

Each detector is counted at the input resolution it is evaluated at, because a
FLOP count quoted at a resolution the model was not evaluated at describes a
different experiment. Ultralytics models use a 640 x 640 input. torchvision
models use a square input at their short side: 800 px for Faster R-CNN,
RetinaNet and FCOS, 300 px for SSD300 and 320 px for SSDlite320. MACs are
counted with thop, and GFLOPs are reported as twice the MAC count.
"""
import argparse, json, os, sys
import torch
from thop import profile
from torchvision.models import detection as det

TV = {
    "faster_rcnn_r50_fpn_v2": (det.fasterrcnn_resnet50_fpn_v2,
                               det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, 800),
    "retinanet_r50_fpn_v2": (det.retinanet_resnet50_fpn_v2,
                             det.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, 800),
    "fcos_r50_fpn": (det.fcos_resnet50_fpn,
                     det.FCOS_ResNet50_FPN_Weights.COCO_V1, 800),
    "ssd300_vgg16": (det.ssd300_vgg16, det.SSD300_VGG16_Weights.COCO_V1, 300),
    "ssdlite320_mnv3": (det.ssdlite320_mobilenet_v3_large,
                        det.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1, 320),
}


def ultra_flops(name, weight_dir, imgsz=640):
    from ultralytics import RTDETR, YOLO
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detect_coco_ultralytics import MODELS
    m = cls(os.path.join(weight_dir, MODELS[name][1]))
    net = m.model.float().eval()
    x = torch.zeros(1, 3, imgsz, imgsz)
    macs, params = profile(net, inputs=(x,), verbose=False)
    return macs, params, imgsz


def tv_flops(name):
    builder, weights, short = TV[name]
    m = builder(weights=weights).eval()
    # A square input at the short side keeps the count comparable across families.
    x = [torch.zeros(3, short, short)]
    macs, params = profile(m, inputs=(x,), verbose=False)
    return macs, params, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weightDir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detect_coco_ultralytics import MODELS as UM

    rec, names = {}, sorted(set(UM) | set(TV))
    for n in names:
        try:
            macs, params, imgsz = (tv_flops(n) if n in TV
                                   else ultra_flops(n, a.weightDir))
            rec[n] = {"model": n, "input_px": imgsz,
                      "gmacs": round(macs / 1e9, 2),
                      "gflops": round(2 * macs / 1e9, 2),
                      "params_M": round(params / 1e6, 2)}
            print(f"  {n:<24} {imgsz:>5} px  {rec[n]['gflops']:>9.2f} GFLOPs  "
                  f"{rec[n]['params_M']:>7.2f} M", flush=True)
        except Exception as e:
            print(f"  {n:<24} FAILED: {type(e).__name__}: {e}", flush=True)
    json.dump({"note": "GFLOPs = 2 x MACs, counted at the input resolution each "
                       "detector is evaluated at; torchvision models use a square "
                       "input at their short side",
               "results": rec}, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}  ({len(rec)} detectors)")


if __name__ == "__main__":
    main()
