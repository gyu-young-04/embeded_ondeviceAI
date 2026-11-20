"""사람(Pedestrian) 오류를 원인별로 나눠서 다음에 무엇을 고칠지 정한다. dev 사진만 쓴다.

- 채점 대상: KITTI Moderate 사람 (높이 25px 초과, 가림 1 이하, 잘림 0.3 이하)
- 기준 점수: dev 에서 F1 이 가장 높은 점수 하나를 골라 그 지점의 오류를 센다.
- 매칭은 점수 높은 탐지부터 차례로 하는 단순한 방식이라 공식 AP 계산과 완전히 같지는 않다.

사용법:
    python scripts/analyze_ped.py --model int8_qat.onnx --images /content/kitti/training/image_2 \
        --labels /content/kitti/training/label_2 --dev-list /content/kitti_yolo/dev.txt \
        --work /content/ped_analysis --out ped_analysis.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import kitti_eval
from kitti_eval import MAX_OCCLUSION, MAX_TRUNCATION, MIN_HEIGHT, box_overlap, read_kitti_file

MODERATE = 1
MATCH_IOU = 0.5
NEAR_IOU = 0.2
TOUCH_IOU = 0.3
HEIGHT_BINS = [(25, 40), (40, 60), (60, 100), (100, 10000)]
MISS_CAUSES = ["점수가 낮음", "겹친 사람에게 뺏김", "박스 어긋남", "자전거로 착각", "차로 착각", "후보 없음"]
FP_CAUSES = ["같은 사람 중복", "자전거를 사람으로", "차를 사람으로", "다른 물체",
             "찾은 사람 옆 박스", "놓친 사람 옆 박스", "채점 제외 사람 옆 박스", "배경"]


def height(box):
    return box[3] - box[1]


def iou(a, b):
    return box_overlap(a, b, -1)


def is_target(g):
    return (g["name"] == "Pedestrian" and g["occluded"] <= MAX_OCCLUSION[MODERATE]
            and g["truncated"] <= MAX_TRUNCATION[MODERATE] and height(g["bbox"]) > MIN_HEIGHT[MODERATE])


def best_iou(box, others):
    return max((iou(box, o) for o in others), default=0.0)


def match_image(gts, dets, thr, with_fp=False):
    """한 사진에서 thr 이상인 사람 탐지를 정답에 짝지어 (놓친 사람 목록, 오탐 원인 목록, 찾은 사람 목록)을 돌려준다."""
    targets = [g for g in gts if is_target(g)]
    ignored = [g["bbox"] for g in gts if g["name"] in ("Pedestrian", "Person_sitting") and not is_target(g)]
    dontcare = [g["bbox"] for g in gts if g["name"] == "DontCare"]
    cyclists = [g["bbox"] for g in gts if g["name"] == "Cyclist"]
    cars = [g["bbox"] for g in gts if g["name"] in ("Car", "Van", "Truck")]
    misc = [g["bbox"] for g in gts if g["name"] in ("Tram", "Misc")]

    peds = sorted((d for d in dets if d["name"] == "Pedestrian" and height(d["bbox"]) >= MIN_HEIGHT[MODERATE]),
                  key=lambda d: -d["score"])
    kept = [d for d in peds if d["score"] >= thr]
    owner, used_ignored, fps, near = {}, set(), [], []
    for k, d in enumerate(kept):
        box = d["bbox"]
        free = [(iou(box, g["bbox"]), i) for i, g in enumerate(targets) if i not in owner]
        ov, i = max(free, default=(0.0, -1))
        if ov > MATCH_IOU:
            owner[i] = k
            continue
        free_ign = [(iou(box, b), j) for j, b in enumerate(ignored) if j not in used_ignored]
        ov, j = max(free_ign, default=(0.0, -1))
        if ov > MATCH_IOU:
            used_ignored.add(j)
            continue
        if any(box_overlap(box, b, 0) > MATCH_IOU for b in dontcare):
            continue
        all_people = [g["bbox"] for g in targets] + ignored
        if best_iou(box, [targets[i]["bbox"] for i in owner]) > MATCH_IOU:
            fps.append(("같은 사람 중복", d))
        elif best_iou(box, cyclists) > TOUCH_IOU:
            fps.append(("자전거를 사람으로", d))
        elif best_iou(box, cars) > TOUCH_IOU:
            fps.append(("차를 사람으로", d))
        elif best_iou(box, misc) > TOUCH_IOU:
            fps.append(("다른 물체", d))
        elif best_iou(box, all_people) > NEAR_IOU:
            near.append(d)
        else:
            fps.append(("배경", d))

    for d in near:   # 사람 근처의 어긋난 박스: 가장 많이 겹치는 사람이 누구인지로 나눈다
        box = d["bbox"]
        ov, is_found = max(((iou(box, g["bbox"]), i in owner) for i, g in enumerate(targets)), default=(0.0, False))
        if best_iou(box, ignored) >= ov:
            fps.append(("채점 제외 사람 옆 박스", d))
        else:
            fps.append(("찾은 사람 옆 박스" if is_found else "놓친 사람 옆 박스", d))

    missed, found = [], [targets[i] for i in owner]
    used = [kept[k]["bbox"] for k in owner.values()]
    for i, g in enumerate(targets):
        if i in owner:
            continue
        box = g["bbox"]
        by_name = lambda name: [d["bbox"] for d in dets if d["score"] >= thr and d["name"] == name]
        kept_boxes = [d["bbox"] for d in kept]
        if best_iou(box, used) > MATCH_IOU:
            cause = "겹친 사람에게 뺏김"
        elif best_iou(box, [d["bbox"] for d in peds if d["score"] < thr]) > MATCH_IOU:
            cause = "점수가 낮음"
        elif best_iou(box, kept_boxes) > NEAR_IOU:
            cause = "박스 어긋남"
        elif best_iou(box, by_name("Cyclist")) > MATCH_IOU:
            cause = "자전거로 착각"
        elif best_iou(box, by_name("Car")) > MATCH_IOU:
            cause = "차로 착각"
        else:
            cause = "후보 없음"
        missed.append((g, cause))
    if not with_fp:   # with_fp=True 면 (원인, 탐지) 를 돌려준다
        fps = [c for c, _ in fps]
    return missed, fps, found


def f1_at(all_gts, all_dets, thr):
    tp = fp = fn = 0
    for gts, dets in zip(all_gts, all_dets):
        missed, fps, found = match_image(gts, dets, thr)
        tp += len(found); fp += len(fps); fn += len(missed)
    return 2 * tp / max(2 * tp + fp + fn, 1), tp, fp, fn


def analyze(all_gts, all_dets, thresholds=None):
    thresholds = np.arange(0.05, 0.91, 0.05) if thresholds is None else thresholds
    thr = max((round(float(t), 2) for t in thresholds), key=lambda t: f1_at(all_gts, all_dets, t)[0])
    f1, tp, fp, fn = f1_at(all_gts, all_dets, thr)

    rows, fp_causes = [], Counter()
    for gts, dets in zip(all_gts, all_dets):
        missed, fps, found = match_image(gts, dets, thr)
        fp_causes.update(fps)
        rows += [(g, "찾음") for g in found] + missed

    def row(label, pick):
        part = [c for g, c in rows if pick(g)]
        cnt = Counter(part)
        return {"구분": label, "사람 수": len(part), "찾은 비율(%)": round(100 * cnt["찾음"] / max(len(part), 1), 1),
                **{c: cnt[c] for c in MISS_CAUSES}}

    table = [row("전체", lambda g: True)]
    table += [row(f"높이 {lo}~{hi}px" if hi < 10000 else f"높이 {lo}px 이상",
                  lambda g, lo=lo, hi=hi: lo < height(g["bbox"]) <= hi) for lo, hi in HEIGHT_BINS]
    table += [row("가림 없음", lambda g: g["occluded"] == 0), row("일부 가림", lambda g: g["occluded"] == 1)]
    table += [row("잘림 적음 (0.15 이하)", lambda g: g["truncated"] <= 0.15),
              row("잘림 있음 (0.15 초과)", lambda g: g["truncated"] > 0.15)]
    return {"threshold": thr, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "table": table,
            "fp_causes": {c: fp_causes[c] for c in FP_CAUSES}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dev-list", required=True)
    parser.add_argument("--work", required=True, help="예측 파일을 둘 폴더")
    parser.add_argument("--out", required=True, help="결과 json")
    parser.add_argument("--ped-iou", type=float, default=0.5, help="사람 클래스 NMS IoU")
    args = parser.parse_args()

    import onnx_infer   # 추론할 때만 필요 (mine_hard.py 는 안 씀)

    ids = [Path(l.strip()).stem for l in Path(args.dev_list).read_text().splitlines() if l.strip()]
    work, labels = Path(args.work), Path(args.labels)
    print(f"dev {len(ids)}장 추론")
    onnx_infer.predict_folder(args.model, args.images, ids, work, iou_thr=(0.5, args.ped_iou, 0.5))
    result, _ = kitti_eval.evaluate(work, labels, ids)
    print(f"dev 사람 Moderate AP: {result['Pedestrian'][1]:.2f}")

    all_gts = [read_kitti_file(labels / f"{i}.txt", False) for i in ids]
    all_dets = [read_kitti_file(work / f"{i}.txt", True) for i in ids]
    r = analyze(all_gts, all_dets)
    r["dev_ped_ap"] = result["Pedestrian"][1]

    print(f"\n기준 점수 {r['threshold']} | F1 {r['f1']:.3f} | 찾음 {r['tp']} | 오탐 {r['fp']} | 놓침 {r['fn']}")
    print("\n[놓친 사람: 어떤 사람을, 왜 놓쳤나]")
    print(pd.DataFrame(r["table"]).to_string(index=False))
    print("\n[오탐: 사람이 아닌데 사람이라고 한 것]")
    print(pd.Series(r["fp_causes"]).to_string())
    Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장:", args.out)


if __name__ == "__main__":
    main()
