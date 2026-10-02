"""Run the detector set over the NightOwls validation split.

NightOwls is a night-time pedestrian benchmark. It supplies what the COCO person
evaluation cannot: pedestrian target semantics, night-time illumination, and
person identities that persist across frames, which make a temporal analysis
possible.

The output format matches the COCO detection files exactly, so the same
evaluators, statistics and oracle code consume them without special-casing.
Detections are written for the person class only.
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from torchvision.models import detection as det
from torchvision.transforms.functional import to_tensor

PERSON_TV = 1        # torchvision COCO indexing
PERSON_ULTRA = 0     # Ultralytics 80-class indexing

TV_MODELS = {
    "faster_rcnn_r50_fpn_v2": (det.fasterrcnn_resnet50_fpn_v2,
                               det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    "retinanet_r50_fpn_v2":   (det.retinanet_resnet50_fpn_v2,
                               det.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
    "fcos_r50_fpn":           (det.fcos_resnet50_fpn,
                               det.FCOS_ResNet50_FPN_Weights.COCO_V1),
    "ssdlite320_mnv3":        (det.ssdlite320_mobilenet_v3_large,
                               det.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
    "ssd300_vgg16":           (det.ssd300_vgg16, det.SSD300_VGG16_Weights.COCO_V1),
}
ULTRA_MODELS = {
    "yolov3u": "yolov3u.pt",
    "yolov3_tinyu": "yolov3-tinyu.pt",
    "yolov3_sppu": "yolov3-sppu.pt",
    "yolov5nu": "yolov5nu.pt",
    "yolov5su": "yolov5su.pt",
    "yolov5mu": "yolov5mu.pt",
    "yolov5lu": "yolov5lu.pt",
    "yolov5xu": "yolov5xu.pt",
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolov9t": "yolov9t.pt",
    "yolov9s": "yolov9s.pt",
    "yolov9m": "yolov9m.pt",
    "yolov9c": "yolov9c.pt",
    "yolov9e": "yolov9e.pt",
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10b": "yolov10b.pt",
    "yolov10l": "yolov10l.pt",
    "yolov10x": "yolov10x.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt",
    "yolo12x": "yolo12x.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    "rtdetr_l": "rtdetr-l.pt",
    "rtdetr_x": "rtdetr-x.pt",
}


def batch_for(weight_name):
    """Pick a batch size from the checkpoint scale. 8 GB of VRAM will not hold
    the x- and e-scale checkpoints at batch 16."""
    w = weight_name.lower()
    if "yolov9e" in w or w.endswith(("x.pt", "xu.pt")):
        return 2
    if w.endswith(("l.pt", "lu.pt", "b.pt", "c.pt")) or "rtdetr" in w:
        return 4
    if w.endswith(("m.pt", "mu.pt")):
        return 8
    return 16


def run_torchvision(name, images, root, out, score_min):
    builder, weights = TV_MODELS[name]
    kw = {"box_score_thresh": score_min} if name.startswith(
        ("faster", "retina", "fcos")) else {"score_thresh": score_min}
    m = builder(weights=weights, **kw).eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    dets, t0 = [], time.time()
    with torch.inference_mode():
        for n, im in enumerate(images):
            x = to_tensor(Image.open(os.path.join(root, im["file_name"]))
                          .convert("RGB")).cuda()
            o = m([x])[0]
            keep = o["labels"] == PERSON_TV
            for (x1, y1, x2, y2), sc in zip(o["boxes"][keep].cpu().numpy(),
                                            o["scores"][keep].cpu().numpy()):
                dets.append({"image_id": im["id"], "category_id": 1,
                             "bbox": [round(float(x1), 2), round(float(y1), 2),
                                      round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                             "score": round(float(sc), 5)})
            if (n + 1) % 5000 == 0:
                el = time.time() - t0
                print(f"   {n+1}/{len(images)} {el:.0f}s {(n+1)/el:.1f} img/s "
                      f"{len(dets)} dets", flush=True)
    el = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    del m; torch.cuda.empty_cache()
    return dets, el, peak


def run_ultralytics(name, images, root, out, score_min):
    from ultralytics import RTDETR, YOLO
    wt = ULTRA_MODELS[name]
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    torch.cuda.empty_cache()
    m = cls(os.path.join(os.environ["ULTRA_WEIGHTS"], wt))
    torch.cuda.reset_peak_memory_stats()
    B = batch_for(wt)
    print(f"   batch {B}", flush=True)
    dets, t0 = [], time.time()
    for s in range(0, len(images), B):
        batch = images[s:s + B]
        paths = [os.path.join(root, i["file_name"]) for i in batch]
        try:
            res = m.predict(paths, conf=score_min, verbose=False, device=0)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            res = []
            for p1 in paths:
                res.extend(m.predict([p1], conf=score_min, verbose=False, device=0))
        for im, r in zip(batch, res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            keep = b.cls.int() == PERSON_ULTRA
            for (x1, y1, x2, y2), sc in zip(b.xyxy[keep].cpu().numpy(),
                                            b.conf[keep].cpu().numpy()):
                dets.append({"image_id": im["id"], "category_id": 1,
                             "bbox": [round(float(x1), 2), round(float(y1), 2),
                                      round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                             "score": round(float(sc), 5)})
        if (s + B) % 5000 < B:
            el = time.time() - t0
            print(f"   {min(s+B,len(images))}/{len(images)} {el:.0f}s "
                  f"{len(dets)} dets", flush=True)
    el = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    del m; torch.cuda.empty_cache()
    return dets, el, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annFile", required=True)
    ap.add_argument("--imageRoot", required=True)
    ap.add_argument("--outDir", required=True)
    ap.add_argument("--weightDir", required=True)
    ap.add_argument("--models", nargs="+",
                    default=list(TV_MODELS) + list(ULTRA_MODELS))
    ap.add_argument("--scoreMin", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ["ULTRA_WEIGHTS"] = a.weightDir

    ann = json.load(open(a.annFile))
    images = sorted(ann["images"], key=lambda i: i["id"])
    if a.limit:
        images = images[:a.limit]
    os.makedirs(a.outDir, exist_ok=True)
    print(f"{len(images)} NightOwls validation images")

    for name in a.models:
        out = os.path.join(a.outDir, f"{name}.json")
        if os.path.exists(out):
            print(f"  skip {name}"); continue
        print(f"\n== {name} ==", flush=True)
        fn = run_torchvision if name in TV_MODELS else run_ultralytics
        dets, el, peak = fn(name, images, a.imageRoot, out, a.scoreMin)
        json.dump(dets, open(out, "w"))
        json.dump({"model": name, "dataset": "nightowls_validation",
                   "n_images": len(images), "n_detections": len(dets),
                   "score_min": a.scoreMin, "seconds": round(el, 1),
                   "img_per_s": round(len(images) / el, 2),
                   "peak_alloc_MiB": round(peak)},
                  open(out.replace(".json", "_meta.json"), "w"), indent=2)
        print(f"   -> {out} ({len(dets)} dets, {el:.0f}s, peak {peak:.0f} MiB)")


if __name__ == "__main__":
    main()
