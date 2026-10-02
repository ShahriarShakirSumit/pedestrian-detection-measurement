"""Deployment timing for the torchvision detectors on one controlled device.

This is a dedicated measurement run and is never derived from training or
evaluation logs.

Reported per model and precision (FP32, and FP16 under autocast), at batch
size 1:
  - preprocessing, model forward and post-processing, separately and end to end
  - CUDA-synchronised median and 95th-percentile latency
  - throughput
  - peak allocated and reserved GPU memory
  - parameter count
Raw per-iteration measurements are written to CSV so that intervals can be
recomputed without re-running.

Timing boundary: the input image is decoded once, before timing. Preprocessing
converts the decoded image to a tensor and copies it to the GPU. The model
stage is the forward pass, which includes the model's own resizing,
normalisation and box post-processing such as non-maximum suppression.
Post-processing copies the outputs back to the host. Each stage ends with a
CUDA synchronisation. Warm-up iterations are discarded, and the peak memory
counters are reset after warm-up.

Results describe the device they were measured on, which is recorded in the
output, and must not be ranked against latency published for other hardware.
"""
import argparse, csv, json, os, platform, subprocess, sys, time
import numpy as np
import torch
from torchvision.models import detection as det
from torchvision.transforms.functional import to_tensor
from PIL import Image

MODELS = {
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


def gpu_state():
    try:
        q = ("clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,"
             "utilization.gpu,persistence_mode")
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            text=True, timeout=15).strip()
        return dict(zip(q.split(","), [v.strip() for v in out.split(",")]))
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="one fixed input image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scoreThresh", type=float, default=0.05)
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    pil = Image.open(a.image).convert("RGB")

    rec = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {"gpu": torch.cuda.get_device_name(0),
                     "total_MiB": round(torch.cuda.mem_get_info()[1] / 2**20),
                     "driver_state_before": gpu_state()},
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(),
                     "python": sys.version.split()[0], "os": platform.platform()},
        "settings": {"batch_size": 1, "input_size": list(pil.size)[::-1],
                     "warmup_iters": a.warmup, "measured_iters": a.iters,
                     "score_threshold": a.scoreThresh,
                     "timing_boundary": "decode = once, outside the timed interval; "
                                        "preprocess = image->tensor->device; "
                                        "model = forward incl. NMS; "
                                        "post = move results to host",
                     "cudnn_benchmark": True},
        "results": {},
    }
    print(json.dumps({**rec["hardware"], **rec["software"]}, indent=2))

    raw_path = a.out.rsplit(".", 1)[0] + "_raw.csv"
    raw = open(raw_path, "w", newline="")
    wtr = csv.writer(raw)
    wtr.writerow(["model", "precision", "iter", "pre_ms", "model_ms", "post_ms", "total_ms"])

    for name in a.models:
        builder, weights = MODELS[name]
        for prec in ("fp32", "fp16"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            kw = {"box_score_thresh": a.scoreThresh} if name.startswith(
                ("faster", "retina", "fcos")) else {"score_thresh": a.scoreThresh}
            m = builder(weights=weights, **kw).eval().to(dev)
            nparam = sum(p.numel() for p in m.parameters())
            amp = prec == "fp16"

            def one():
                t0 = time.perf_counter()
                x = to_tensor(pil).to(dev, non_blocking=True)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=amp):
                    o = m([x])[0]
                torch.cuda.synchronize(); t2 = time.perf_counter()
                _ = {k: v.detach().cpu() for k, v in o.items()}
                torch.cuda.synchronize(); t3 = time.perf_counter()
                return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3

            for _ in range(a.warmup):
                one()
            torch.cuda.reset_peak_memory_stats()

            pre, mod, post = [], [], []
            for i in range(a.iters):
                p_, m_, q_ = one()
                pre.append(p_); mod.append(m_); post.append(q_)
                wtr.writerow([name, prec, i, f"{p_:.4f}", f"{m_:.4f}",
                              f"{q_:.4f}", f"{p_+m_+q_:.4f}"])

            tot = np.array(pre) + np.array(mod) + np.array(post)
            rec["results"][f"{name}::{prec}"] = {
                "model": name, "precision": prec, "params_M": round(nparam / 1e6, 2),
                "pre_ms_median": round(float(np.median(pre)), 2),
                "model_ms_median": round(float(np.median(mod)), 2),
                "post_ms_median": round(float(np.median(post)), 2),
                "total_ms_median": round(float(np.median(tot)), 2),
                "total_ms_p95": round(float(np.percentile(tot, 95)), 2),
                "model_ms_p95": round(float(np.percentile(mod, 95)), 2),
                "throughput_img_s": round(1000.0 / float(np.median(tot)), 2),
                "peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20),
            }
            r = rec["results"][f"{name}::{prec}"]
            print(f"  {name:<24}{prec}  total={r['total_ms_median']:>7.2f} ms "
                  f"(p95 {r['total_ms_p95']:>7.2f})  model={r['model_ms_median']:>7.2f}  "
                  f"{r['throughput_img_s']:>6.2f} img/s  peak={r['peak_alloc_MiB']:>5} MiB")
            del m; torch.cuda.empty_cache()

    raw.close()
    rec["hardware"]["driver_state_after"] = gpu_state()
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}\n-> {raw_path}")


if __name__ == "__main__":
    main()
