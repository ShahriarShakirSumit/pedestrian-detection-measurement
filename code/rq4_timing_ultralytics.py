"""Deployment timing for the Ultralytics detectors.

Results are merged into the torchvision timing record written by
`rq4_timing.py` (passed as --merge). The run settings match that script: batch
size 1, FP32 and FP16, the same default warm-up and iteration counts, the same
score threshold, and CUDA-synchronised timing. Run it on the same device as
`rq4_timing.py`, because the merged record carries that script's hardware and
software description.

The timing boundary is the same as in `rq4_timing.py`. The image is read and
decoded once, before timing, and every timed call receives the decoded array.
The timed interval covers pre-processing and transfer to the device, the
forward pass including NMS, and copying the detections to the host.

Ultralytics performs pre-processing, the forward pass and post-processing inside
a single predict() call and does not expose the stages separately. Only the
end-to-end latency is therefore measured. It is stored as both the model-stage
and the total time, the pre-processing and post-processing fields are left as
None, and each record is marked stage_split_available = False.
"""
import argparse, csv, json, os, platform, sys, time
import numpy as np
import torch
from PIL import Image
from ultralytics import RTDETR, YOLO
from ultralytics.utils.patches import imread

# The detector list is taken from the detection sweep rather than repeated
# here, so a model can never be timed under a name it was not evaluated under.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_coco_ultralytics import MODELS as _SWEEP_MODELS  # noqa: E402

MODELS = dict(_SWEEP_MODELS)   # already {name: (loader_class, weight_file)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weightDir", required=True)
    ap.add_argument("--merge", required=True, help="existing rq4 json to extend")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scoreThresh", type=float, default=0.05)
    a = ap.parse_args()

    rec = json.load(open(a.merge))
    rec["ultralytics_note"] = (
        "The image is decoded once, outside the timed interval, as for torchvision. "
        "Ultralytics performs pre-processing, the forward pass and NMS inside "
        "predict(), and the timed interval also copies the detections to the host, "
        "so the end-to-end total covers the same boundary as the torchvision total. "
        "The internal split is not available and is attributed to the model stage.")
    # Decoded once with the reader predict() itself uses (BGR, as from a path).
    img = imread(a.image)
    rec["settings"]["input_size"] = list(img.shape[:2])
    raw = open(a.out.rsplit(".", 1)[0] + "_ultralytics_raw.csv", "w", newline="")
    w = csv.writer(raw); w.writerow(["model", "precision", "iter", "total_ms"])

    for name, (cls, wt) in MODELS.items():
        for prec in ("fp32", "fp16"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = cls(os.path.join(a.weightDir, wt))
            half = prec == "fp16"
            kw = dict(conf=a.scoreThresh, verbose=False, device=0, half=half)
            def one():
                r = m.predict(img, **kw)[0]
                _ = r.boxes.data.cpu()
            for _ in range(a.warmup):
                one()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ts = []
            for i in range(a.iters):
                t0 = time.perf_counter()
                one()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e3
                ts.append(dt); w.writerow([name, prec, i, f"{dt:.4f}"])
            ts = np.array(ts)
            nparam = sum(p.numel() for p in m.model.parameters())
            rec["results"][f"{name}::{prec}"] = {
                "model": name, "precision": prec,
                "params_M": round(nparam / 1e6, 2),
                "pre_ms_median": None, "model_ms_median": round(float(np.median(ts)), 2),
                "post_ms_median": None,
                "total_ms_median": round(float(np.median(ts)), 2),
                "total_ms_p95": round(float(np.percentile(ts, 95)), 2),
                "model_ms_p95": round(float(np.percentile(ts, 95)), 2),
                "throughput_img_s": round(1000.0 / float(np.median(ts)), 2),
                "peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20),
                "stage_split_available": False,
            }
            r = rec["results"][f"{name}::{prec}"]
            print(f"  {name:<12}{prec}  total={r['total_ms_median']:>7.2f} ms "
                  f"(p95 {r['total_ms_p95']:>7.2f})  {r['throughput_img_s']:>6.2f} img/s  "
                  f"peak={r['peak_alloc_MiB']:>5} MiB  params={r['params_M']}M")
            del m; torch.cuda.empty_cache()

    raw.close()
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
