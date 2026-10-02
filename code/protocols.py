"""Three incompatible detection-evaluation protocols, implemented explicitly.

One fixed set of detections can be scored under each protocol, so that the
evaluator, not the detector, is the variable. The VOC and miss-rate protocols
are written out in full rather than delegated to a library, because the
differences between protocols are exactly what is being measured.

Protocol summary
----------------
COCO   AP averaged over IoU 0.50:0.05:0.95, 101-point recall interpolation.
       Crowd regions permit repeated matches and never count as false
       positives. Delegated to pycocotools, which is the reference
       implementation.
VOC    AP at a single IoU 0.50. `difficult` ground truth is excluded: a
       detection matched to it is discarded rather than scored either way.
       Two interpolations, the VOC2007 11-point sample and the VOC2010+
       all-point area under the precision envelope.
MR     Log-average miss rate sampled at nine log-spaced false-positives-per-
       image points from 0.01 to 1.0, at IoU 0.50, lower is better. Ignore
       regions are matched by intersection over detection area rather than
       IoU, following the Caltech and CityPersons convention, and absorb
       detections without penalty.

The VOC and MR protocols share one greedy, score-ordered matching stage, the
same rule that COCOeval applies, so that any difference in outcome is
attributable to the protocol definition and not to a different matching rule.
"""
import numpy as np

FPPI_POINTS = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000,
                        0.1778, 0.3162, 0.5623, 1.0000])


def iou_matrix(dt, gt):
    """IoU between detections and ground truth. Boxes are xywh."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    d = np.asarray(dt, float); g = np.asarray(gt, float)
    dx1, dy1 = d[:, 0], d[:, 1]; dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx1, gy1 = g[:, 0], g[:, 1]; gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.maximum(0, np.minimum(dx2[:, None], gx2[None]) - np.maximum(dx1[:, None], gx1[None]))
    ih = np.maximum(0, np.minimum(dy2[:, None], gy2[None]) - np.maximum(dy1[:, None], gy1[None]))
    inter = iw * ih
    union = (d[:, 2] * d[:, 3])[:, None] + (g[:, 2] * g[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def ioa_matrix(dt, gt):
    """Intersection over DETECTION area. Used for ignore regions, where a small
    detection fully inside a large ignore region must count as ignored even
    though its IoU is tiny."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    d = np.asarray(dt, float); g = np.asarray(gt, float)
    dx1, dy1 = d[:, 0], d[:, 1]; dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx1, gy1 = g[:, 0], g[:, 1]; gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.maximum(0, np.minimum(dx2[:, None], gx2[None]) - np.maximum(dx1[:, None], gx1[None]))
    ih = np.maximum(0, np.minimum(dy2[:, None], gy2[None]) - np.maximum(dy1[:, None], gy1[None]))
    inter = iw * ih
    area = (d[:, 2] * d[:, 3])[:, None]
    return np.where(area > 0, inter / np.maximum(area, 1e-12), 0.0)


def overlap_matrix(dt, gt, gt_ignore):
    """Overlap between detections and ground truth, xywh boxes.

    Real objects use intersection over union. Ignore objects use intersection
    over DETECTION area, which is the COCO `iscrowd` convention that the KAIST
    and CityPersons evaluators inherit: a small detection lying wholly inside a
    large ignore region must be absorbed by it even though their IoU is tiny.
    Mixing the two definitions in one matrix is deliberate and is what the
    reference implementations do.
    """
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    d = np.asarray(dt, float); g = np.asarray(gt, float)
    ig = np.asarray(gt_ignore, dtype=bool)
    dx1, dy1 = d[:, 0], d[:, 1]; dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx1, gy1 = g[:, 0], g[:, 1]; gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.minimum(dx2[:, None], gx2[None]) - np.maximum(dx1[:, None], gx1[None])
    ih = np.minimum(dy2[:, None], gy2[None]) - np.maximum(dy1[:, None], gy1[None])
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    darea = (d[:, 2] * d[:, 3])[:, None]
    garea = (g[:, 2] * g[:, 3])[None]
    denom = np.where(ig[None], np.broadcast_to(darea, inter.shape),
                     darea + garea - inter)
    return np.where(denom > 0, inter / np.maximum(denom, 1e-12), 0.0)


def _match_image(dets, scores, gt_boxes, gt_ignore, iou_thr, max_det):
    """Greedy score-ordered matching within one image, following the COCOeval
    convention that the KAIST and CityPersons evaluators both inherit.

    Ground truth flagged `ignore` stays in the matching pool but is sorted last,
    so a detection prefers a real object and falls back to an ignore object only
    when no real object matches. A detection matched to ignore ground truth is
    neither a true nor a false positive, and ignore objects absorb any number of
    detections.

    Returns per-detection labels in score order: 1 true positive, 0 false
    positive, -1 ignored.
    """
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")[:max_det]
    dets = [dets[i] for i in order]
    n = len(dets)
    labels = np.zeros(n, dtype=int)
    if n == 0 or len(gt_boxes) == 0:
        return labels, order

    gt_ignore = np.asarray(gt_ignore, dtype=int)
    gidx = np.argsort(gt_ignore, kind="mergesort")      # real objects first
    gts = [gt_boxes[i] for i in gidx]
    gig = gt_ignore[gidx]

    ov = overlap_matrix(dets, gts, gig)
    matched = np.zeros(len(gts), dtype=bool)

    for di in range(n):
        best, bg, bm = iou_thr, -1, -2
        for gj in range(len(gts)):
            if matched[gj]:
                continue
            if bm != -2 and gig[gj] == 1:
                break              # already matched, and ignores are sorted last
            if ov[di, gj] < best:
                continue
            best, bg = ov[di, gj], gj
            bm = 1 if gig[gj] == 0 else -1
        if bg < 0:
            continue
        if gig[bg] == 0:
            matched[bg] = True     # real objects match at most once
            labels[di] = 1
        else:
            labels[di] = -1        # absorbed by an ignore object

    return labels, order


def evaluate(gt_by_image, dt_by_image, iou_thr=0.5, max_det=1000):
    """Shared matching stage across all protocols.

    `gt_by_image[img]` is `(boxes, ignore_flags)`. Ignore regions are supplied
    as ordinary entries carrying an ignore flag, which is how the reference
    evaluators represent them.
    `dt_by_image[img]` is `(boxes, scores)`.

    Returns cumulative TP/FP arrays in global score order, the number of
    matchable ground-truth objects, and the image count used for FPPI.
    """
    recs, npos = [], 0
    for img in gt_by_image:
        gt_boxes, gt_ign = gt_by_image[img]
        dts, scs = dt_by_image.get(img, ([], []))
        npos += int(np.sum(np.asarray(gt_ign, dtype=int) == 0))
        if not len(dts):
            continue
        labels, order = _match_image(dts, scs, gt_boxes, gt_ign, iou_thr, max_det)
        s = np.asarray(scs, float)[order]
        recs.extend(zip(s.tolist(), labels.tolist()))

    if not recs:
        return np.array([]), np.array([]), npos, len(gt_by_image)
    recs.sort(key=lambda r: -r[0])
    labs = np.array([r[1] for r in recs])
    labs = labs[labs >= 0]                  # ignored detections leave the ranking
    return np.cumsum(labs == 1), np.cumsum(labs == 0), npos, len(gt_by_image)


def voc_ap(tp, fp, npos, mode="all"):
    """VOC average precision at a single IoU. `mode` is '11point' for VOC2007
    or 'all' for the VOC2010+ all-point envelope."""
    if npos == 0 or len(tp) == 0:
        return 0.0
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-12)
    if mode == "11point":
        ap = 0.0
        for t in np.arange(0, 1.01, 0.1):
            p = prec[rec >= t].max() if (rec >= t).any() else 0.0
            ap += p / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):          # precision envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def log_average_miss_rate(tp, fp, npos, n_images, fppi_points=FPPI_POINTS):
    """Log-average miss rate over the given FPPI sampling points."""
    if npos == 0:
        return float("nan"), None
    if len(tp) == 0:
        return 1.0, {float(f): 1.0 for f in fppi_points}
    rec = tp / npos
    fppi = fp / n_images
    # tp is cumulative, so recall is non-decreasing and the miss rate is
    # non-increasing in score order, which lets it be sampled directly
    mr = 1 - rec
    sampled = {}
    vals = []
    for f in fppi_points:
        idx = np.searchsorted(fppi, f, side="right") - 1
        v = float(mr[idx]) if idx >= 0 else 1.0
        v = min(max(v, 1e-10), 1.0)
        sampled[float(f)] = v
        vals.append(v)
    return float(np.exp(np.mean(np.log(vals)))), sampled
