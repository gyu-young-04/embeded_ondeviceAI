"""KITTI 2D Object Detection 공식 채점 방식(AP40)을 파이썬으로 옮긴 코드.

KITTI devkit(cpp/evaluate_object.cpp)과 널리 쓰이는 파이썬 이식본
(kitti-object-eval-python)의 2D bbox 채점 규칙을 그대로 따른다.

- 클래스별 IoU 기준: Car 0.7, Pedestrian 0.5, Cyclist 0.5
- 난이도: Easy / Moderate / Hard (박스 높이, 가려짐, 잘림 기준)
- Van 은 Car 의, Person_sitting 은 Pedestrian 의 이웃 클래스로 취급해 감점하지 않는다.
- DontCare 영역 안의 탐지는 오답으로 세지 않는다.
- AP40: recall 1/40 ~ 40/40 지점의 precision 평균

예측 파일은 KITTI 라벨과 같은 형식(이미지마다 txt 하나, 마지막 열이 점수)이어야 한다.

사용법:
    python scripts/kitti_eval.py --pred preds/ --label /content/kitti/training/label_2 --ids eval_val.txt
"""

import argparse
from pathlib import Path

import numpy as np

CLASSES = ["Car", "Pedestrian", "Cyclist"]
MIN_OVERLAP = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
DIFFICULTIES = ["Easy", "Moderate", "Hard"]
MIN_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]
N_SAMPLE_PTS = 41


def read_kitti_file(path, with_score):
    objs = []
    if not path.exists():
        return objs
    for row in path.read_text().splitlines():
        p = row.split()
        if not p:
            continue
        objs.append({
            "name": p[0],
            "truncated": float(p[1]),
            "occluded": int(float(p[2])),
            "bbox": [float(v) for v in p[4:8]],
            "score": float(p[15]) if with_score and len(p) > 15 else 0.0,
        })
    return objs


def box_overlap(a, b, criterion):
    """criterion -1: IoU, 0: 교집합 / a 의 넓이"""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    if iw <= 0:
        return 0.0
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if criterion == 0:
        return inter / area_a if area_a > 0 else 0.0
    union = area_a + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clean_data(gts, dets, cls, diff):
    cls_l = cls.lower()
    ignored_gt, ignored_dt, dc_boxes = [], [], []
    num_valid_gt = 0
    for g in gts:
        name = g["name"].lower()
        height = g["bbox"][3] - g["bbox"][1]
        if name == cls_l:
            valid = 1
        elif (cls_l == "pedestrian" and name == "person_sitting") or (cls_l == "car" and name == "van"):
            valid = 0
        else:
            valid = -1
        ignore = (g["occluded"] > MAX_OCCLUSION[diff] or g["truncated"] > MAX_TRUNCATION[diff]
                  or height <= MIN_HEIGHT[diff])
        if valid == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid == 0 or (ignore and valid == 1):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if g["name"] == "DontCare":
            dc_boxes.append(g["bbox"])
    for d in dets:
        height = abs(d["bbox"][3] - d["bbox"][1])
        if height < MIN_HEIGHT[diff]:
            ignored_dt.append(1)
        elif d["name"].lower() == cls_l:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    return num_valid_gt, ignored_gt, ignored_dt, dc_boxes


def compute_statistics(overlaps, gts, dets, ignored_gt, ignored_dt, dc_boxes, min_overlap, thresh, compute_fp):
    no_det = -10000000
    n_det = len(dets)
    scores = [d["score"] for d in dets]
    assigned = [False] * n_det
    below = [compute_fp and s < thresh for s in scores]
    tp = fp = fn = 0
    tp_scores = []

    for i in range(len(gts)):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_det = no_det
        max_overlap = 0.0
        assigned_ignored = False
        for j in range(n_det):
            if ignored_dt[j] == -1 or assigned[j] or below[j]:
                continue
            ov = overlaps[j][i]
            if not compute_fp and ov > min_overlap and scores[j] > valid_det:
                det_idx, valid_det = j, scores[j]
            elif compute_fp and ov > min_overlap and (ov > max_overlap or assigned_ignored) and ignored_dt[j] == 0:
                max_overlap, det_idx, valid_det, assigned_ignored = ov, j, 1, False
            elif compute_fp and ov > min_overlap and valid_det == no_det and ignored_dt[j] == 1:
                det_idx, valid_det, assigned_ignored = j, 1, True

        if valid_det == no_det and ignored_gt[i] == 0:
            fn += 1
        elif valid_det != no_det and (ignored_gt[i] == 1 or ignored_dt[det_idx] == 1):
            assigned[det_idx] = True
        elif valid_det != no_det:
            tp += 1
            tp_scores.append(scores[det_idx])
            assigned[det_idx] = True

    if compute_fp:
        for j in range(n_det):
            if not (assigned[j] or ignored_dt[j] in (-1, 1) or below[j]):
                fp += 1
        nstuff = 0
        for dc in dc_boxes:
            for j in range(n_det):
                if assigned[j] or ignored_dt[j] in (-1, 1) or below[j]:
                    continue
                if box_overlap(dets[j]["bbox"], dc, 0) > min_overlap:
                    assigned[j] = True
                    nstuff += 1
        fp -= nstuff
    return tp, fp, fn, tp_scores


def get_thresholds(scores, num_gt):
    scores = sorted(scores, reverse=True)
    current_recall = 0.0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < len(scores) - 1 else l_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < len(scores) - 1:
            continue
        thresholds.append(score)
        current_recall += 1.0 / (N_SAMPLE_PTS - 1.0)
    return thresholds


def eval_class(all_gts, all_dets, cls, diff):
    min_overlap = MIN_OVERLAP[cls]
    per_image = []
    total_valid_gt = 0
    all_tp_scores = []
    for gts, dets in zip(all_gts, all_dets):
        num_valid, ig_gt, ig_dt, dc = clean_data(gts, dets, cls, diff)
        overlaps = [[box_overlap(d["bbox"], g["bbox"], -1) for g in gts] for d in dets]
        total_valid_gt += num_valid
        _, _, _, tp_scores = compute_statistics(overlaps, gts, dets, ig_gt, ig_dt, dc, min_overlap, 0.0, False)
        all_tp_scores += tp_scores
        per_image.append((overlaps, gts, dets, ig_gt, ig_dt, dc))

    if total_valid_gt == 0:
        return float("nan")
    thresholds = get_thresholds(all_tp_scores, total_valid_gt)
    precision = np.zeros(N_SAMPLE_PTS)
    for t_idx, t in enumerate(thresholds):
        tp = fp = 0
        for overlaps, gts, dets, ig_gt, ig_dt, dc in per_image:
            a, b, _, _ = compute_statistics(overlaps, gts, dets, ig_gt, ig_dt, dc, min_overlap, t, True)
            tp += a
            fp += b
        precision[t_idx] = tp / (tp + fp) if tp + fp > 0 else 0.0
    for i in range(len(thresholds)):
        precision[i] = precision[i:].max()
    return precision[1:].sum() / 40 * 100


def evaluate(pred_dir, label_dir, ids):
    """반환: {클래스: [Easy, Moderate, Hard] AP40}, 그리고 Moderate mAP"""
    pred_dir, label_dir = Path(pred_dir), Path(label_dir)
    all_gts = [read_kitti_file(label_dir / f"{i}.txt", False) for i in ids]
    all_dets = [read_kitti_file(pred_dir / f"{i}.txt", True) for i in ids]
    result = {cls: [eval_class(all_gts, all_dets, cls, d) for d in range(3)] for cls in CLASSES}
    moderate_map = float(np.mean([result[c][1] for c in CLASSES]))
    return result, moderate_map


def format_result(result, moderate_map):
    lines = [f"{'AP40':<11}" + "".join(f"{d:>10}" for d in DIFFICULTIES)]
    for cls in CLASSES:
        lines.append(f"{cls:<11}" + "".join(f"{v:>10.2f}" for v in result[cls]))
    lines.append(f"Moderate mAP (3클래스 평균): {moderate_map:.2f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="예측 txt 폴더 (KITTI 형식, 마지막 열이 점수)")
    parser.add_argument("--label", required=True, help="KITTI training/label_2 폴더")
    parser.add_argument("--ids", default=str(Path(__file__).resolve().parent.parent / "eval_val.txt"))
    args = parser.parse_args()
    ids = [l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()]
    print(format_result(*evaluate(args.pred, args.label, ids)))


if __name__ == "__main__":
    main()
