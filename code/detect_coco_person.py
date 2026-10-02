"""Run the torchvision COCO-pretrained detectors over COCO val2017 and write
person detections.

Produces one COCO-format detection file per model. These files are then scored
under several evaluation protocols without re-running inference, so the
evaluator is the only thing that varies in the protocol-sensitivity analysis.

Detections are written unfiltered by score, apart from a low floor (--scoreMin)
passed to each model's built-in score threshold, because different protocols
need different operating points. Thresholding at evaluation time is a protocol
choice, not an inference choice.
"""
import argparse, json, os, sys, time
import torch
from PIL import Image
from torchvision.models import detection as det
from torchvision.transforms.functional import to_tensor

PERSON = 1  # COCO category id, identical to torchvision's label index

MODELS = {
    "faster_rcnn_r50_fpn_v2": (det.fasterrcnn_resnet50_fpn_v2,
                               det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    "retinanet_r50_fpn_v2":   (det.retinanet_resnet50_fpn_v2,
                               det.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
    "fcos_r50_fpn":           (det.fcos_resnet50_fpn,
                               det.FCOS_ResNet50_FPN_Weights.COCO_V1),
    "ssdlite320_mnv3":        (det.ssdlite320_mobilenet_v3_large,
                               det.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
    "ssd300_vgg16":           (det.ssd300_vgg16,
                               det.SSD300_VGG16_Weights.COCO_V1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cocoRoot", required=True)
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--outDir", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--scoreMin", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ann = json.load(open(f"{a.cocoRoot}/annotations/instances_{a.split}.json"))
    images = sorted(ann["images"], key=lambda i: i["id"])
    if a.limit:
        images = images[:a.limit]
    print(f"{len(images)} images in {a.split}")

    os.makedirs(a.outDir, exist_ok=True)
    dev = torch.device("cuda")

    for name in a.models:
        builder, weights = MODELS[name]
        out = os.path.join(a.outDir, f"{name}.json")
        if os.path.exists(out):
            print(f"  skip {name} (exists)"); continue
        print(f"\n== {name} ==")
        w = weights
        m = builder(weights=w, box_score_thresh=a.scoreMin) if "rcnn" in name or "retina" in name or "fcos" in name \
            else builder(weights=w, score_thresh=a.scoreMin)
        m.eval().to(dev)
        torch.cuda.reset_peak_memory_stats()

        dets, t0 = [], time.time()
        with torch.inference_mode():
            for n, im in enumerate(images):
                p = os.path.join(a.cocoRoot, a.split, im["file_name"])
                x = to_tensor(Image.open(p).convert("RGB")).to(dev)
                o = m([x])[0]
                keep = o["labels"] == PERSON
                b = o["boxes"][keep].cpu().numpy()
                s = o["scores"][keep].cpu().numpy()
                for (x1, y1, x2, y2), sc in zip(b, s):
                    dets.append({"image_id": im["id"], "category_id": PERSON,
                                 "bbox": [round(float(x1), 2), round(float(y1), 2),
                                          round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                 "score": round(float(sc), 5)})
                if (n + 1) % 500 == 0:
                    el = time.time() - t0
                    print(f"   {n+1}/{len(images)}  {el:.0f}s  {(n+1)/el:.1f} img/s  "
                          f"{len(dets)} dets", flush=True)
        el = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        json.dump(dets, open(out, "w"))
        meta = {"model": name, "weights": str(w), "split": a.split,
                "n_images": len(images), "n_detections": len(dets),
                "score_min": a.scoreMin, "seconds": round(el, 1),
                "img_per_s": round(len(images) / el, 2),
                "peak_alloc_MiB": round(peak),
                "note": "inference-only pass; not a deployment timing measurement"}
        json.dump(meta, open(out.replace(".json", "_meta.json"), "w"), indent=2)
        print(f"   -> {out}  ({len(dets)} dets, {el:.0f}s, peak {peak:.0f} MiB)")
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
