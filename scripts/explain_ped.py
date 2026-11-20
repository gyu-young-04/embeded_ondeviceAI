"""사람(Pedestrian) 오류를 두 가지로 자세히 본다. dev 사진만 쓴다.

1. 점수 손실 나누기: 오류 종류 하나씩을 '완벽하게 고쳤다고 치고' dev 사람 Moderate AP 를 다시 계산한다.
   많이 오르는 종류일수록 고쳤을 때 이득이 크다. (실제로 다 고칠 수는 없으므로 '최대 이득'이다)
2. 틀린 사진 모아 보기: 놓친 사람과 오탐 주변을 잘라 한 장의 그림으로 모은다.
   초록 = 채점 대상 사람, 회색 = 채점 제외(앉은 사람·많이 가려진 사람 등), 빨강 = 기준 점수 이상 탐지, 주황 = 기준 점수 미만 탐지

사용법:
    python scripts/explain_ped.py --model fp32.onnx --images /content/kitti/training/image_2 \
        --labels /content/kitti/training/label_2 --dev-list /content/kitti_yolo/dev.txt \
        --work /content/ped_explain --out-dir <결과 폴더>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import kitti_eval
from analyze_ped import analyze, best_iou, height, iou, is_target, match_image
from kitti_eval import box_overlap, read_kitti_file

FIXES = [
    ("loc", "박스 위치를 정확하게"),
    ("rank", "점수 순서를 정확하게 (맞는 박스가 틀린 박스보다 위)"),
    ("dup", "같은 사람 중복 박스 지우기"),
    ("near_ign", "채점 제외 사람 근처 박스 지우기"),
    ("other_cls", "다른 물체를 사람으로 본 박스 지우기"),
    ("bg", "배경 오탐 지우기"),
    ("missed", "완전히 놓친 사람 추가"),
]
MISS_CODE = {"점수가 낮음": "LOW", "겹친 사람에게 뺏김": "TAKEN", "박스 어긋남": "LOC",
             "자전거로 착각": "AS-CYC", "차로 착각": "AS-CAR", "후보 없음": "NONE"}
FP_CODE = {"같은 사람 중복": "DUP", "자전거를 사람으로": "CYC", "차를 사람으로": "CAR", "다른 물체": "MISC",
           "찾은 사람 옆 박스": "NEAR-FOUND", "놓친 사람 옆 박스": "NEAR-MISS", "채점 제외 사람 옆 박스": "NEAR-IGN", "배경": "BG"}


def classify(gts, dets):
    """사람 탐지 하나하나를 (종류, 정답 번호)로 나눈다. 점수 높은 탐지부터 정답을 차지한다."""
    targets = [g for g in gts if is_target(g)]
    ignored = [g["bbox"] for g in gts if g["name"] in ("Pedestrian", "Person_sitting") and not is_target(g)]
    dontcare = [g["bbox"] for g in gts if g["name"] == "DontCare"]
    others = [g["bbox"] for g in gts if g["name"] in ("Cyclist", "Car", "Van", "Truck", "Tram", "Misc")]
    order = sorted((j for j, d in enumerate(dets) if d["name"] == "Pedestrian"), key=lambda j: -dets[j]["score"])
    labels, taken = {}, set()
    for j in order:
        box = dets[j]["bbox"]
        if height(box) < kitti_eval.MIN_HEIGHT[1]:
            continue   # 채점에서 빠지는 작은 박스
        ov, i = max(((iou(box, g["bbox"]), i) for i, g in enumerate(targets)), default=(0.0, -1))
        if ov > 0.5:
            labels[j] = ("dup", i) if i in taken else ("tp", i)
            taken.add(i)
        elif best_iou(box, ignored) > 0.5 or any(box_overlap(box, b, 0) > 0.5 for b in dontcare):
            continue   # 채점에서 무시되는 박스
        elif ov >= 0.1:
            labels[j] = ("loc", i)
        elif best_iou(box, ignored) >= 0.1:
            labels[j] = ("near_ign", -1)
        elif best_iou(box, others) >= 0.3:
            labels[j] = ("other_cls", -1)
        else:
            labels[j] = ("bg", -1)
    near_any = lambda g: any(iou(dets[j]["bbox"], g["bbox"]) >= 0.1 for j in order)
    missed = [g["bbox"] for i, g in enumerate(targets) if i not in taken and not near_any(g)]
    return targets, labels, taken, missed


def apply_fixes(all_gts, all_dets, kinds):
    out = []
    for gts, dets in zip(all_gts, all_dets):
        targets, labels, taken, missed = classify(gts, dets)
        new, snapped = [], set()
        for j, d in enumerate(dets):
            kind, i = labels.get(j, ("keep", -1))
            if kind == "loc" and "loc" in kinds:
                if i in taken or i in snapped:
                    continue   # 이미 맞는 박스가 있는 사람이면 남는 박스는 지운다
                snapped.add(i)
                d = {**d, "bbox": list(targets[i]["bbox"])}
            elif kind == "tp" and "rank" in kinds:
                d = {**d, "score": d["score"] + 1.0}
            elif kind in kinds and kind in ("dup", "near_ign", "other_cls", "bg"):
                continue
            new.append(d)
        if "missed" in kinds:
            new += [{"name": "Pedestrian", "bbox": list(b), "score": 2.0, "truncated": -1, "occluded": -1} for b in missed]
        out.append(new)
    return out


def ped_ap(all_gts, all_dets):
    return kitti_eval.eval_class(all_gts, all_dets, "Pedestrian", 1)


def breakdown(all_gts, all_dets):
    base = ped_ap(all_gts, all_dets)
    rows = [{"고친 것": text, "사람 AP": round(ap, 2), "오르는 점수": round(ap - base, 2)}
            for kind, text in FIXES for ap in [ped_ap(all_gts, apply_fixes(all_gts, all_dets, {kind}))]]
    every = ped_ap(all_gts, apply_fixes(all_gts, all_dets, {k for k, _ in FIXES}))
    rows.append({"고친 것": "전부", "사람 AP": round(every, 2), "오르는 점수": round(every - base, 2)})
    return base, rows


TILE_W, TILE_H, HEAD = 128, 192, 18


def tile(img, focus, gts, dets, thr, text):
    """focus 박스 주변을 잘라 박스들을 그리고 위에 글자를 붙인다."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = focus
    rh = max((y2 - y1) * 1.8, 120.0)
    rw = max(rh * TILE_W / TILE_H, (x2 - x1) * 1.5)
    rh = rw * TILE_H / TILE_W
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(np.clip(cx - rw / 2, 0, max(w - rw, 0)))
    top = int(np.clip(cy - rh / 2, 0, max(h - rh, 0)))
    right, bottom = int(min(left + rw, w)), int(min(top + rh, h))
    crop = img[top:bottom, left:right]
    sx, sy = TILE_W / max(crop.shape[1], 1), TILE_H / max(crop.shape[0], 1)
    crop = cv2.resize(crop, (TILE_W, TILE_H))

    def draw(box, color, thick=1):
        p1 = (int((box[0] - left) * sx), int((box[1] - top) * sy))
        p2 = (int((box[2] - left) * sx), int((box[3] - top) * sy))
        cv2.rectangle(crop, p1, p2, color, thick)

    for g in gts:
        if g["name"] in ("Pedestrian", "Person_sitting"):
            draw(g["bbox"], (0, 200, 0) if is_target(g) else (160, 160, 160))
    for d in dets:
        if d["name"] == "Pedestrian" and d["score"] >= 0.05 and iou(d["bbox"], focus) > 0.05:
            draw(d["bbox"], (0, 0, 255) if d["score"] >= thr else (0, 165, 255))
    head = np.zeros((HEAD, TILE_W, 3), np.uint8)
    cv2.putText(head, text, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, crop])


def grid(tiles, cols=10):
    if not tiles:
        return None
    blank = np.full_like(tiles[0], 255)
    tiles = tiles + [blank] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def gallery(ids, images, all_gts, all_dets, thr, out_dir, limit=80):
    miss_tiles, fp_tiles = [], []
    for image_id, gts, dets in zip(ids, all_gts, all_dets):
        missed, fps, _ = match_image(gts, dets, thr, with_fp=True)
        if not missed and not fps:
            continue
        img = cv2.imread(str(Path(images) / f"{image_id}.png"))
        for g, cause in missed:
            b = g["bbox"]
            text = f"{MISS_CODE[cause]} h{int(height(b))} o{g['occluded']}"
            miss_tiles.append((list(MISS_CODE).index(cause), height(b), tile(img, b, gts, dets, thr, text)))
        for cause, d in fps:
            fp_tiles.append((list(FP_CODE).index(cause), -d["score"], tile(img, d["bbox"], gts, dets, thr, f"{FP_CODE[cause]} {d['score']:.2f}")))
    paths = {}
    for name, items in (("missed", miss_tiles), ("fp", fp_tiles)):
        items.sort(key=lambda t: (t[0], t[1]))
        g = grid([t[2] for t in items[:limit]])
        if g is not None:
            paths[name] = Path(out_dir) / f"ped_errors_{name}.png"
            cv2.imwrite(str(paths[name]), g)
    return paths, len(miss_tiles), len(fp_tiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dev-list", required=True)
    parser.add_argument("--work", required=True, help="예측 파일을 둘 폴더")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ped-iou", type=float, default=0.5, help="사람 클래스 NMS IoU")
    args = parser.parse_args()

    ids = [Path(l.strip()).stem for l in Path(args.dev_list).read_text().splitlines() if l.strip()]
    work, labels, out_dir = Path(args.work), Path(args.labels), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(work.glob("*.txt"))) != len(ids):
        import onnx_infer
        print(f"dev {len(ids)}장 추론")
        onnx_infer.predict_folder(args.model, args.images, ids, work, iou_thr=(0.5, args.ped_iou, 0.5))
    all_gts = [read_kitti_file(labels / f"{i}.txt", False) for i in ids]
    all_dets = [read_kitti_file(work / f"{i}.txt", True) for i in ids]

    base, rows = breakdown(all_gts, all_dets)
    print(f"\n[점수 손실 나누기] 지금 dev 사람 AP {base:.2f}")
    print("하나씩 '완벽하게 고쳤다고 치면' 사람 AP 가 얼마나 오르는지예요. 클수록 고칠 가치가 커요.")
    width = max(len(r["고친 것"]) for r in rows)
    for r in rows:
        print(f"  {r['고친 것']:<{width}}  {r['사람 AP']:7.2f}  {r['오르는 점수']:+6.2f}")

    thr = analyze(all_gts, all_dets)["threshold"]
    paths, n_miss, n_fp = gallery(ids, args.images, all_gts, all_dets, thr, out_dir)
    print(f"\n[틀린 사진 모아 보기] 기준 점수 {thr} | 놓친 사람 {n_miss}명 | 오탐 {n_fp}개 (그림에는 각각 최대 80개)")
    for k, p in paths.items():
        print(" ", k, p)
    (out_dir / "ped_explain.json").write_text(json.dumps({"base": base, "fixes": rows, "threshold": thr},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
