"""Evaluate visible-trained detectors on LLVIP visible and infrared imagery.

LLVIP provides 15,488 registered visible/infrared pairs of the same scenes with
one shared annotation per pair. Because the pairing is exact and the annotation
is shared, the visible and infrared evaluations differ in one respect only: the
input modality. This isolates the modality effect from the differences in
architecture, training data and annotation that separate published
multispectral results from one another.

Every detector here is COCO-pretrained on visible imagery, so this is a
domain-shift measurement rather than a multispectral detector comparison. It
measures how much of a visible-trained detector's performance transfers to
thermal input, and whether the ranking of detectors is preserved across the
change of modality (Kendall tau between the visible and infrared rankings).
Confidence intervals come from a bootstrap that resamples image pairs.

Scope: LLVIP is low-light throughout and its people are large, with a median box
height of 222 px in train and 247 px in test against 83 px in CityPersons. It
therefore cannot support conclusions about scale or occlusion and serves as a
secondary modality study only.
"""
import argparse, json, os, sys, time, glob
import xml.etree.ElementTree as ET
import numpy as np
import torch
from collections import defaultdict
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P
import statistics as ST


def load_llvip_gt(root, split):
    """Load the person boxes shared by each visible/infrared pair.

    Annotations are Pascal VOC XML files keyed by image stem. Boxes are
    returned as [x, y, w, h] with no ignore regions."""
    vis = sorted(glob.glob(os.path.join(root, "visible", split, "*.jpg")))
    gt, ids, stems = {}, [], []
    for p in vis:
        stem = os.path.splitext(os.path.basename(p))[0]
        xml = os.path.join(root, "Annotations", stem + ".xml")
        if not os.path.exists(xml):
            continue
        boxes = []
        for o in ET.parse(xml).getroot().findall("object"):
            if o.findtext("name") != "person":
                continue
            b = o.find("bndbox")
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        gt[stem] = (boxes, [0] * len(boxes))
        ids.append(stem); stems.append(stem)
    return gt, ids


def detect(model_name, root, split, modality, stems, weight_dir, score_min=0.01):
    from torchvision.models import detection as det
    TV = {
        "faster_rcnn_r50_fpn_v2": (det.fasterrcnn_resnet50_fpn_v2,
                                   det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
        "retinanet_r50_fpn_v2": (det.retinanet_resnet50_fpn_v2,
                                 det.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
        "fcos_r50_fpn": (det.fcos_resnet50_fpn, det.FCOS_ResNet50_FPN_Weights.COCO_V1),
        "ssdlite320_mnv3": (det.ssdlite320_mobilenet_v3_large,
                            det.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
        "ssd300_vgg16": (det.ssd300_vgg16, det.SSD300_VGG16_Weights.COCO_V1),
    }
    ULTRA = {
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
    out = defaultdict(lambda: ([], []))
    d = os.path.join(root, modality, split)

    if model_name in TV:
        builder, w = TV[model_name]
        kw = {"box_score_thresh": score_min} if model_name.startswith(
            ("faster", "retina", "fcos")) else {"score_thresh": score_min}
        m = builder(weights=w, **kw).eval().cuda()
        with torch.inference_mode():
            for stem in stems:
                x = to_tensor(Image.open(os.path.join(d, stem + ".jpg"))
                              .convert("RGB")).cuda()
                o = m([x])[0]
                keep = o["labels"] == 1
                for bb, sc in zip(o["boxes"][keep].cpu().numpy(),
                                  o["scores"][keep].cpu().numpy()):
                    out[stem][0].append([float(bb[0]), float(bb[1]),
                                         float(bb[2] - bb[0]), float(bb[3] - bb[1])])
                    out[stem][1].append(float(sc))
        del m
    else:
        from ultralytics import RTDETR, YOLO
        cls = RTDETR if model_name.startswith("rtdetr") else YOLO
        torch.cuda.empty_cache()
        m = cls(os.path.join(weight_dir, ULTRA[model_name]))
        w = ULTRA[model_name].lower()
        if "yolov9e" in w or w.endswith(("x.pt", "xu.pt")):
            B = 2
        elif w.endswith(("l.pt", "lu.pt", "b.pt", "c.pt")) or "rtdetr" in w:
            B = 4
        elif w.endswith(("m.pt", "mu.pt")):
            B = 8
        else:
            B = 16
        for s in range(0, len(stems), B):
            batch = stems[s:s + B]
            _paths = [os.path.join(d, st + ".jpg") for st in batch]
            try:
                res = m.predict(_paths, conf=score_min, verbose=False, device=0)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                res = []
                for _p in _paths:
                    res.extend(m.predict([_p], conf=score_min, verbose=False, device=0))
            for st, r in zip(batch, res):
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                keep = b.cls.int() == 0
                for bb, sc in zip(b.xyxy[keep].cpu().numpy(), b.conf[keep].cpu().numpy()):
                    out[st][0].append([float(bb[0]), float(bb[1]),
                                       float(bb[2] - bb[0]), float(bb[3] - bb[1])])
                    out[st][1].append(float(sc))
        del m
    torch.cuda.empty_cache()
    return out


def score(gt, dets, ids):
    npos = [len(gt[i][0]) for i in ids]
    recs = []
    for i in ids:
        gb, gi = gt[i]
        db, ds = dets.get(i, ([], []))
        if not db:
            recs.append(np.zeros((0, 2))); continue
        labels, order = P._match_image(db, ds, gb, gi, 0.5, 1000)
        s = np.asarray(ds, float)[order]
        keep = labels >= 0
        recs.append(np.column_stack([s[keep], labels[keep]]))
    return ST.BootstrapSet(recs, npos, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--weightDir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=[
        "yolov3u",
        "yolov3_tinyu",
        "yolov3_sppu",
        "yolov5nu",
        "yolov5su",
        "yolov5mu",
        "yolov5lu",
        "yolov5xu",
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
        "yolov9t",
        "yolov9s",
        "yolov9m",
        "yolov9c",
        "yolov9e",
        "yolov10n",
        "yolov10s",
        "yolov10m",
        "yolov10b",
        "yolov10l",
        "yolov10x",
        "yolo11n",
        "yolo11s",
        "yolo11m",
        "yolo11l",
        "yolo11x",
        "yolo12n",
        "yolo12s",
        "yolo12m",
        "yolo12l",
        "yolo12x",
        "yolo26n",
        "yolo26s",
        "yolo26m",
        "yolo26l",
        "yolo26x",
        "rtdetr_l",
        "rtdetr_x",
        "faster_rcnn_r50_fpn_v2",
        "retinanet_r50_fpn_v2",
        "fcos_r50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mnv3",
    ])
    ap.add_argument("--nBoot", type=int, default=500)
    a = ap.parse_args()

    gt, ids = load_llvip_gt(a.root, a.split)
    print(f"LLVIP {a.split}: {len(ids)} pairs, "
          f"{sum(len(gt[i][0]) for i in ids)} annotated people")

    rec = {"dataset": "LLVIP", "split": a.split, "n_pairs": len(ids),
           "n_people": sum(len(gt[i][0]) for i in ids),
           "design": ("registered visible/infrared pairs with one shared "
                      "annotation; the two evaluations differ only in input "
                      "modality"),
           "caveat": ("all detectors are COCO-pretrained on visible imagery, so "
                      "the infrared column is a domain-shift measurement, not a "
                      "multispectral detector comparison"),
           "results": {}}

    sets = {}
    for m in a.models:
        row = {}
        for modality in ("visible", "infrared"):
            t0 = time.time()
            dets = detect(m, a.root, a.split, modality, ids, a.weightDir)
            bs = score(gt, dets, ids)
            sets[f"{m}::{modality}"] = bs
            mt = bs.metrics()
            row[modality] = {"MR": round(mt["MR"], 2), "AP50": round(mt["AP50"], 2),
                             "recall": round(mt["recall"], 2),
                             "seconds": round(time.time() - t0, 1)}
        row["modality_gap_MR"] = round(row["infrared"]["MR"] - row["visible"]["MR"], 2)
        rec["results"][m] = row
        print(f"  {m:<26} visible MR {row['visible']['MR']:>6.2f}   "
              f"infrared MR {row['infrared']['MR']:>6.2f}   "
              f"gap {row['modality_gap_MR']:>+7.2f}")

    print(f"\nbootstrap ({a.nBoot} replicates) ...")
    boot = ST.bootstrap(sets, n_boot=a.nBoot, seed=20260822)
    boot.pop("replicates", None)
    rec["bootstrap"] = {"per_model": boot["per_model"], "n_boot": a.nBoot,
                        "resampling_unit": "image pair"}

    vis_rank = sorted(a.models, key=lambda m: rec["results"][m]["visible"]["MR"])
    ir_rank = sorted(a.models, key=lambda m: rec["results"][m]["infrared"]["MR"])
    tau, _, _ = ST.kendall_tau({m: i + 1 for i, m in enumerate(vis_rank)},
                               {m: i + 1 for i, m in enumerate(ir_rank)})
    rec["ranking"] = {"visible": vis_rank, "infrared": ir_rank,
                      "kendall_tau_visible_vs_infrared": round(tau, 4)}
    print(f"\nranking agreement across modality: Kendall tau = {tau:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
