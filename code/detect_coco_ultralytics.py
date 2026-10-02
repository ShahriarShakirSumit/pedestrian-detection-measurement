"""Run the Ultralytics detectors over COCO val2017 and write person detections.

The output format matches `detect_coco_person.py` exactly, so both detector
families are scored by the same evaluators without special-casing downstream.

Class indexing: Ultralytics models use the 80-class COCO indexing, in which
person is 0, while torchvision uses the 91-class indexing, in which person is 1.
Detections are written with category_id 1 so that both families share one
namespace.
"""
import argparse, json, os, time
import torch
from ultralytics import RTDETR, YOLO

PERSON_ULTRA = 0
PERSON_COCO = 1

MODELS = {
    "yolov3u": (YOLO, "yolov3u.pt"),
    "yolov3_tinyu": (YOLO, "yolov3-tinyu.pt"),
    "yolov3_sppu": (YOLO, "yolov3-sppu.pt"),
    "yolov5nu": (YOLO, "yolov5nu.pt"),
    "yolov5su": (YOLO, "yolov5su.pt"),
    "yolov5mu": (YOLO, "yolov5mu.pt"),
    "yolov5lu": (YOLO, "yolov5lu.pt"),
    "yolov5xu": (YOLO, "yolov5xu.pt"),
    "yolov8n": (YOLO, "yolov8n.pt"),
    "yolov8s": (YOLO, "yolov8s.pt"),
    "yolov8m": (YOLO, "yolov8m.pt"),
    "yolov8l": (YOLO, "yolov8l.pt"),
    "yolov8x": (YOLO, "yolov8x.pt"),
    "yolov9t": (YOLO, "yolov9t.pt"),
    "yolov9s": (YOLO, "yolov9s.pt"),
    "yolov9m": (YOLO, "yolov9m.pt"),
    "yolov9c": (YOLO, "yolov9c.pt"),
    "yolov9e": (YOLO, "yolov9e.pt"),
    "yolov10n": (YOLO, "yolov10n.pt"),
    "yolov10s": (YOLO, "yolov10s.pt"),
    "yolov10m": (YOLO, "yolov10m.pt"),
    "yolov10b": (YOLO, "yolov10b.pt"),
    "yolov10l": (YOLO, "yolov10l.pt"),
    "yolov10x": (YOLO, "yolov10x.pt"),
    "yolo11n": (YOLO, "yolo11n.pt"),
    "yolo11s": (YOLO, "yolo11s.pt"),
    "yolo11m": (YOLO, "yolo11m.pt"),
    "yolo11l": (YOLO, "yolo11l.pt"),
    "yolo11x": (YOLO, "yolo11x.pt"),
    "yolo12n": (YOLO, "yolo12n.pt"),
    "yolo12s": (YOLO, "yolo12s.pt"),
    "yolo12m": (YOLO, "yolo12m.pt"),
    "yolo12l": (YOLO, "yolo12l.pt"),
    "yolo12x": (YOLO, "yolo12x.pt"),
    "yolo26n": (YOLO, "yolo26n.pt"),
    "yolo26s": (YOLO, "yolo26s.pt"),
    "yolo26m": (YOLO, "yolo26m.pt"),
    "yolo26l": (YOLO, "yolo26l.pt"),
    "yolo26x": (YOLO, "yolo26x.pt"),
    "rtdetr_l": (RTDETR, "rtdetr-l.pt"),
    "rtdetr_x": (RTDETR, "rtdetr-x.pt"),
}


def batch_for(weight_name):
    """Pick a batch size from the checkpoint scale. 8 GB of VRAM will not hold
    the x- and e-scale models at batch 16, and yolov9e will not hold batch 8."""
    w = weight_name.lower()
    if "yolov9e" in w or w.endswith(("x.pt", "xu.pt")):
        return 2
    if w.endswith(("l.pt", "lu.pt", "b.pt", "c.pt")) or "rtdetr" in w:
        return 4
    if w.endswith(("m.pt", "mu.pt")):
        return 8
    return 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cocoRoot", required=True)
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--outDir", required=True)
    ap.add_argument("--weightDir", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--scoreMin", type=float, default=0.01)
    a = ap.parse_args()

    ann = json.load(open(f"{a.cocoRoot}/annotations/instances_{a.split}.json"))
    images = sorted(ann["images"], key=lambda i: i["id"])
    os.makedirs(a.outDir, exist_ok=True)
    print(f"{len(images)} images")

    for name in a.models:
        cls, wt = MODELS[name]
        out = os.path.join(a.outDir, f"{name}.json")
        if os.path.exists(out):
            print(f"  skip {name}"); continue
        print(f"\n== {name} ==", flush=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m = cls(os.path.join(a.weightDir, wt))

        dets, t0 = [], time.time()
        B = batch_for(wt)
        print(f"   batch {B}", flush=True)
        for s in range(0, len(images), B):
            batch = images[s:s + B]
            paths = [os.path.join(a.cocoRoot, a.split, i["file_name"]) for i in batch]
            try:
                res = m.predict(paths, conf=a.scoreMin, verbose=False, device=0)
            except torch.OutOfMemoryError:
                # If a batch does not fit in GPU memory, run its images one at a
                # time instead.
                torch.cuda.empty_cache()
                res = []
                for p1 in paths:
                    res.extend(m.predict([p1], conf=a.scoreMin, verbose=False, device=0))
            for im, r in zip(batch, res):
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                keep = b.cls.int() == PERSON_ULTRA
                xy = b.xyxy[keep].cpu().numpy()
                sc = b.conf[keep].cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xy, sc):
                    dets.append({"image_id": im["id"], "category_id": PERSON_COCO,
                                 "bbox": [round(float(x1), 2), round(float(y1), 2),
                                          round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                 "score": round(float(c), 5)})
            if (s + B) % 1000 < B:
                el = time.time() - t0
                print(f"   {min(s+B,len(images))}/{len(images)}  {el:.0f}s  "
                      f"{len(dets)} dets", flush=True)

        el = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        json.dump(dets, open(out, "w"))
        json.dump({"model": name, "weights": wt, "split": a.split,
                   "n_images": len(images), "n_detections": len(dets),
                   "score_min": a.scoreMin, "seconds": round(el, 1),
                   "img_per_s": round(len(images) / el, 2),
                   "peak_alloc_MiB": round(peak),
                   "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
                   "note": "batched inference pass; not a deployment timing measurement"},
                  open(out.replace(".json", "_meta.json"), "w"), indent=2)
        print(f"   -> {out} ({len(dets)} dets, {el:.0f}s, peak {peak:.0f} MiB)")
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
