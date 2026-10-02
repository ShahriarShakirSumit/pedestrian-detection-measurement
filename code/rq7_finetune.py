"""Measure what fine-tuning on the target domain adds to COCO-pretrained weights.

The other controlled results are inference from released COCO-pretrained
weights under a common protocol. That design cannot show whether the
differences observed between detectors survive adaptation to the deployment
domain, or how much of the gap to a domain-specific system is closed by
fine-tuning alone. This script evaluates one torchvision detector on LLVIP
before and after fine-tuning on LLVIP training images.

Data. LLVIP is used because it is the one pedestrian dataset in this study with
a usable training split: 12,025 training pairs and 3,463 test pairs, disjoint by
construction in the released layout. Either image of the registered pair can be
used (--modality, visible by default). A validation split is carved from the
training images only, so the test split plays no part in choosing the
configuration. The test split is evaluated once per condition, with the schedule
set in advance.

Controls. Everything that could differ between the compared conditions is held
fixed: the same images, annotations, evaluator, operating point and seed. The
only change is whether the detector was adapted.

Schedule. The defaults are SGD with momentum 0.9 and weight decay 1e-4, a
one-cycle learning-rate policy with a peak of 5e-3, batch 2 with gradient
accumulation over 4 steps (effective batch 8), mixed precision, two epochs,
and horizontal flipping as the only augmentation. Peak memory is recorded
because the 8 GB GPU budget is the constraint that shaped this schedule, and
reporting the schedule without that budget would make it look like a free
choice.
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import detection as det
from torchvision.transforms.functional import to_tensor, hflip

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocols as P
import statistics as ST

BUILDERS = {
    "faster_rcnn_r50_fpn_v2": (det.fasterrcnn_resnet50_fpn_v2,
                               det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    "retinanet_r50_fpn_v2": (det.retinanet_resnet50_fpn_v2,
                             det.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
    "fcos_r50_fpn": (det.fcos_resnet50_fpn, det.FCOS_ResNet50_FPN_Weights.COCO_V1),
    "ssd300_vgg16": (det.ssd300_vgg16, det.SSD300_VGG16_Weights.COCO_V1),
    "ssdlite320_mnv3": (det.ssdlite320_mobilenet_v3_large,
                        det.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
}

# The SSD builders name the score threshold differently from the other builders.
SCORE_KW = {"ssd300_vgg16": "score_thresh", "ssdlite320_mnv3": "score_thresh"}


def read_boxes(xml):
    """Parse person boxes as [x1, y1, x2, y2].

    LLVIP contains 5 zero-width annotations out of 42,437 (images 091180,
    020118, 020236, 170350 and 090437). torchvision asserts on any box without
    positive height and width, so such boxes are dropped here."""
    out = []
    for o in ET.parse(xml).getroot().findall("object"):
        if o.findtext("name") != "person":
            continue
        b = o.find("bndbox")
        x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
        x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            continue
        out.append([x1, y1, x2, y2])
    return out


class LLVIP(Dataset):
    def __init__(self, root, stems, split, train=False, modality="visible"):
        self.root, self.stems, self.split, self.train = root, stems, split, train
        self.modality = modality

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(os.path.join(self.root, self.modality, self.split,
                                      stem + ".jpg")).convert("RGB")
        boxes = read_boxes(os.path.join(self.root, "Annotations", stem + ".xml"))
        x = to_tensor(img)
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.train and random.random() < 0.5:      # the only augmentation
            x = hflip(x)
            if len(b):
                w = x.shape[-1]
                b = b.clone()
                b[:, [0, 2]] = w - b[:, [2, 0]]
        if len(b):
            keep = (b[:, 2] - b[:, 0] > 1e-3) & (b[:, 3] - b[:, 1] > 1e-3)
            b = b[keep]
        return x, {"boxes": b,
                   "labels": torch.ones(len(b), dtype=torch.int64)}, stem


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def evaluate(model, root, split, stems, device, score_min=0.01,
             modality="visible"):
    model.eval()
    gt, dets = {}, defaultdict(lambda: ([], []))
    with torch.inference_mode():
        for stem in stems:
            img = Image.open(os.path.join(root, modality, split, stem + ".jpg")).convert("RGB")
            xyxy = read_boxes(os.path.join(root, "Annotations", stem + ".xml"))
            gt[stem] = ([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in xyxy],
                        [0] * len(xyxy))
            o = model([to_tensor(img).to(device)])[0]
            keep = (o["labels"] == 1) & (o["scores"] >= score_min)
            for bb, sc in zip(o["boxes"][keep].cpu().numpy(), o["scores"][keep].cpu().numpy()):
                dets[stem][0].append([float(bb[0]), float(bb[1]),
                                      float(bb[2] - bb[0]), float(bb[3] - bb[1])])
                dets[stem][1].append(float(sc))
    npos = [len(gt[s][0]) for s in stems]
    recs = []
    for s in stems:
        gb, gi = gt[s]
        db, ds = dets.get(s, ([], []))
        if not db:
            recs.append(np.zeros((0, 2))); continue
        labels, order = P._match_image(db, ds, gb, gi, 0.5, 1000)
        sc = np.asarray(ds, float)[order]
        keep = labels >= 0
        recs.append(np.column_stack([sc[keep], labels[keep]]))
    return ST.BootstrapSet(recs, npos, stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="faster_rcnn_r50_fpn_v2")
    ap.add_argument("--modality", default="visible",
                    choices=["visible", "infrared"],
                    help="which channel of the registered pair to train and test on")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--valFraction", type=float, default=0.1)
    ap.add_argument("--trainLimit", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda")

    import glob
    tr = sorted(os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(a.root, a.modality, "train", "*.jpg")))
    te = sorted(os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(a.root, a.modality, "test", "*.jpg")))
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(tr))
    n_val = int(len(tr) * a.valFraction)
    val_stems = [tr[i] for i in perm[:n_val]]
    train_stems = [tr[i] for i in perm[n_val:]]
    if a.trainLimit:
        train_stems = train_stems[:a.trainLimit]
    print(f"train {len(train_stems)}  val {len(val_stems)}  test {len(te)}")

    builder, weights = BUILDERS[a.model]
    rec = {"model": a.model, "dataset": f"LLVIP {a.modality}",
           "seed": a.seed,
           "schedule": {"epochs": a.epochs, "batch": a.batch,
                        "grad_accum": a.accum,
                        "effective_batch": a.batch * a.accum, "lr": a.lr,
                        "optimiser": "SGD momentum 0.9, weight decay 1e-4",
                        "amp": True, "augmentation": "horizontal flip only"},
           "splits": {"train": len(train_stems), "val": len(val_stems),
                      "test": len(te)},
           "note": ("the test split is evaluated once, after the schedule was "
                    "fixed; validation is carved from training images only")}

    # Baseline: released COCO weights, no adaptation.
    print("\n== baseline: COCO-pretrained, no adaptation ==")
    kw = {SCORE_KW.get(a.model, "box_score_thresh"): 0.01}
    base = builder(weights=weights, **kw).to(device)
    b_val = evaluate(base, a.root, "train", val_stems, device, modality=a.modality).metrics()
    b_test = evaluate(base, a.root, "test", te, device, modality=a.modality).metrics()
    rec["baseline"] = {"val": {k: round(v, 2) for k, v in b_val.items()},
                       "test": {k: round(v, 2) for k, v in b_test.items()}}
    print(f"   val  MR {b_val['MR']:.2f}   test MR {b_test['MR']:.2f}")
    del base; torch.cuda.empty_cache()

    # Fine-tuning.
    print("\n== fine-tuning ==")
    model = builder(weights=weights, **kw).to(device)
    if hasattr(model, "roi_heads"):
        pass       # person keeps COCO label 1, so the COCO heads are reused as is
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(LLVIP(a.root, train_stems, "train", train=True, modality=a.modality),
                    batch_size=a.batch, shuffle=True, num_workers=4,
                    collate_fn=collate, pin_memory=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, a.epochs * len(dl) // a.accum + 1))
    torch.cuda.reset_peak_memory_stats()
    model.train()
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        for i, (imgs, tgts, _) in enumerate(dl):
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            if all(len(t["boxes"]) == 0 for t in tgts):
                continue
            with torch.autocast("cuda", torch.float16):
                loss = sum(model(imgs, tgts).values())
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                try: sched.step()
                except Exception: pass
                step += 1
            if (i + 1) % 500 == 0:
                print(f"   ep{ep} it{i+1}/{len(dl)} loss {loss.item():.3f} "
                      f"{time.time()-t0:.0f}s peak "
                      f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    rec["training"] = {"seconds": round(train_s, 1), "optimiser_steps": step,
                       "peak_alloc_MiB": round(peak)}
    print(f"   trained {train_s:.0f}s, {step} steps, peak {peak:.0f} MiB")

    f_val = evaluate(model, a.root, "train", val_stems, device, modality=a.modality).metrics()
    f_test = evaluate(model, a.root, "test", te, device, modality=a.modality).metrics()
    rec["finetuned"] = {"val": {k: round(v, 2) for k, v in f_val.items()},
                        "test": {k: round(v, 2) for k, v in f_test.items()}}
    rec["improvement"] = {
        "val_MR_delta": round(b_val["MR"] - f_val["MR"], 2),
        "test_MR_delta": round(b_test["MR"] - f_test["MR"], 2),
        "test_recall_delta": round(f_test["recall"] - b_test["recall"], 2)}

    print(f"\n   baseline  test MR {b_test['MR']:>6.2f}  recall {b_test['recall']:>6.2f}")
    print(f"   finetuned test MR {f_test['MR']:>6.2f}  recall {f_test['recall']:>6.2f}")
    print(f"   improvement {rec['improvement']['test_MR_delta']:>+6.2f} miss-rate points")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
