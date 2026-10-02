"""Re-evaluate published KAIST detection files under one fixed protocol.

The detections are the authors' released outputs, but the evaluator (recorded
by checksum), annotation version and subset definitions are the same for every
method. The resulting miss rates are therefore comparable with one another in a
way that the original papers' tables are not.

They are not comparable with the controlled runs in this repository. Each
method was trained and run by its own authors, with its own architecture,
training data and inference settings, so this re-scoring ranks complete
published systems rather than detectors under controlled conditions. It must
not be merged into a controlled-experiment ranking.

Beyond the Reasonable subset reported by the MLPD evaluation script's own
command line, this also reports the evaluator's other subsets (small, heavy
occlusion and All), each over all images and split by day and night.

Writes JSON and CSV so that no metric is typed into a table by hand.
"""
import argparse, copy, hashlib, io, json, os, platform, sys, time
from contextlib import redirect_stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalKit", required=True, help="dir holding evaluation_script.py")
    ap.add_argument("--annFile", required=True)
    ap.add_argument("--rstFiles", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kit = os.path.abspath(a.evalKit)
    ann = os.path.abspath(a.annFile)
    rsts = [os.path.abspath(r) for r in a.rstFiles]
    out = os.path.abspath(a.out)
    sys.path.insert(0, kit)
    os.chdir(kit)
    import evaluation_script as ev

    with redirect_stdout(io.StringIO()):
        gt = ev.KAIST(ann)
    imgIds = sorted(gt.getImgIds())
    names = {i["id"]: i["im_name"] for i in gt.dataset["images"]}

    # The MLPD evaluation script splits day and night at a hard-coded image
    # index. Check that index against the KAIST set numbering (set06 to set08
    # are day, set09 to set11 are night) rather than assuming it is correct.
    SPLIT = 1455
    day_sets = {f"set{n:02d}" for n in (6, 7, 8)}
    day_ok = all(names[i].split("/")[0] in day_sets for i in imgIds[:SPLIT])
    night_ok = all(names[i].split("/")[0] not in day_sets for i in imgIds[SPLIT:])
    if not (day_ok and night_ok):
        raise SystemExit("day/night split index 1455 does not match set numbering")

    p = ev.KAISTParams(iouType="bbox")
    rec = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_layer": "published detections re-evaluated under one local protocol",
        "comparable_with_controlled_experiments": False,
        "evaluator": {
            "source": "https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection",
            "path": kit,
            "evaluation_script_sha256": sha256(os.path.join(kit, "evaluation_script.py")),
            "cocoeval_sha256": sha256(os.path.join(kit, "cocoeval.py")),
            "coco_sha256": sha256(os.path.join(kit, "coco.py")),
        },
        "protocol": {
            "metric": "log-average miss rate, percent, lower is better",
            "iou_threshold": float(p.iouThrs[0]),
            "fppi_sampling_points": [float(x) for x in p.fppiThrs],
            "max_detections": int(max(p.maxDets)),
            "category_ids_evaluated": [1],
            "subsets": {lbl: {"height_px": [float(h[0]), float(h[1])],
                              "occlusion_levels": [int(o) for o in occ]}
                        for lbl, h, occ in zip(p.SetupLbl, p.HtRng, p.OccRng)},
            "day_night_split": {
                "day_image_index_range": [0, SPLIT], "night_image_index_range": [SPLIT, len(imgIds)],
                "verified_against_set_numbering": True,
                "day_sets": sorted(day_sets), "n_day": SPLIT, "n_night": len(imgIds) - SPLIT,
            },
        },
        "annotation_file": {
            "path": ann, "sha256": sha256(ann),
            "n_images": len(imgIds), "n_annotations": len(gt.dataset["annotations"]),
        },
        "environment": {"python": sys.version.split()[0], "os": platform.platform(),
                        "numpy": __import__("numpy").__version__},
        "results": {},
    }

    for rst in rsts:
        method = os.path.basename(rst).split("_")[0]
        row = {"detection_file": rst, "detection_file_sha256": sha256(rst)}
        for cond, ids in (("all", imgIds), ("day", imgIds[:SPLIT]), ("night", imgIds[SPLIT:])):
            for sid, lbl in enumerate(p.SetupLbl):
                # KAISTPedEval._prepare() writes `ignore` flags onto the ground-truth
                # objects for the current height/occlusion setup and never resets
                # them, so a reused KAIST object carries those filters into every
                # later evaluation. The ground truth is therefore reloaded for
                # each evaluation, as the upstream command-line tool does.
                with redirect_stdout(io.StringIO()):
                    g = ev.KAIST(ann)
                    d = g.loadRes(rst)
                    e = ev.KAISTPedEval(g, d, "bbox", method)
                    e.params.catIds = [1]
                    e.params.imgIds = ids
                    e.evaluate(sid); e.accumulate()
                    mr = e.summarize(sid)
                row[f"MR_{cond}_{lbl}"] = round(float(mr) * 100, 2)
                if cond == "all" and sid == 0:
                    assert e.eval["yy"], "evaluator produced no recall curve"
                    row["recall_all_Reasonable"] = round(
                        float(1 - e.eval["yy"][0][-1]) * 100, 2)
        row["day_night_gap_Reasonable"] = round(
            row["MR_night_Reasonable"] - row["MR_day_Reasonable"], 2)
        rec["results"][method] = row
        print(f"  evaluated {method}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=2)

    cols = ["MR_all_Reasonable", "MR_day_Reasonable", "MR_night_Reasonable",
            "day_night_gap_Reasonable", "MR_all_Reasonable_small",
            "MR_all_Reasonable_occ=heavy", "MR_all_All", "recall_all_Reasonable"]
    csv = out.rsplit(".", 1)[0] + ".csv"
    order = sorted(rec["results"].items(), key=lambda kv: kv[1]["MR_all_Reasonable"])
    with open(csv, "w") as f:
        f.write("method," + ",".join(cols) + "\n")
        for k, v in order:
            f.write(k + "," + ",".join(str(v[c]) for c in cols) + "\n")

    hdr = ["method", "Reas", "Day", "Night", "gap", "small", "heavyOcc", "All", "recall"]
    print("\n" + "".join(f"{h:>12}" if i else f"{h:<12}" for i, h in enumerate(hdr)))
    for k, v in order:
        print(f"{k:<12}" + "".join(f"{v[c]:>12.2f}" for c in cols))
    print(f"\n-> {out}\n-> {csv}")


if __name__ == "__main__":
    main()
