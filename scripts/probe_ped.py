"""학습하기 전에 사람 점수의 병목을 가늠한다. dev 사진만 쓴다.

1. 맞힌 사람의 박스 정확도: 찾은 사람의 IoU 가 0.5~0.6 으로 아슬아슬한지, 0.4~0.5 로 아깝게 놓친 박스가 몇 개인지.
   아슬아슬한 사람이 많으면 박스 손실(box)을 키우는 실험이 유망하다.
2. 입력 크기만 키워 보기: 같은 모델(재학습 없음)을 더 큰 입력으로 돌려 작은 사람을 더 찾는지 본다.
   많이 오르면 해상도가 병목이라 큰 입력으로 학습하는 실험이 유망하다.
   (학습하지 않은 크기라 점수 자체보다 '방향'만 본다)

사용법:
    python scripts/probe_ped.py --best-pt best.pt --base-onnx fp32.onnx --base-size 1280 --sizes 1280 1408 1536 \
        --images /content/kitti/training/image_2 --labels /content/kitti/training/label_2 \
        --dev-list /content/kitti_yolo/dev.txt --work /content/ped_probe --out-dir <결과 폴더>
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import kitti_eval
import onnx_infer
from analyze_ped import HEIGHT_BINS, analyze, height, iou
from explain_ped import classify
from kitti_eval import read_kitti_file

IOU_BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def bin_name(lo, hi):
    return f"높이 {lo}~{hi}px" if hi < 10000 else f"높이 {lo}px 이상"


def iou_table(all_gts, all_dets, thr):
    """기준 점수 이상으로 맞힌 사람의 IoU 분포와, 0.4~0.5 로 아깝게 놓친 박스 수."""
    names = ["전체"] + [bin_name(*b) for b in HEIGHT_BINS]
    rows = {k: {"찾은 사람": 0, **{f"IoU {lo}~{min(hi, 1.0)}": 0 for lo, hi in IOU_BINS}, "평균 IoU": [],
                           "아깝게 놓침 (0.4~0.5)": 0} for k in names}
    first = "전체"
    for gts, dets in zip(all_gts, all_dets):
        targets, labels, taken, _ = classify(gts, dets)
        for j, (kind, i) in labels.items():
            if dets[j]["score"] < thr or i < 0:
                continue
            g = targets[i]["bbox"]
            v = iou(dets[j]["bbox"], g)
            keys = [first] + [bin_name(lo, hi) for lo, hi in HEIGHT_BINS if lo < height(g) <= hi]
            for k in keys:
                if kind == "tp":
                    rows[k]["찾은 사람"] += 1
                    rows[k]["평균 IoU"].append(v)
                    for lo, hi in IOU_BINS:
                        if lo < v <= hi:
                            rows[k][f"IoU {lo}~{min(hi, 1.0)}"] += 1
                elif kind == "loc" and v >= 0.4 and i not in taken:
                    rows[k]["아깝게 놓침 (0.4~0.5)"] += 1
    for r in rows.values():
        r["평균 IoU"] = round(sum(r["평균 IoU"]) / len(r["평균 IoU"]), 3) if r["평균 IoU"] else None
    return rows


def onnx_for(size, args):
    if size == args.base_size:
        return Path(args.base_onnx)
    out = Path(args.out_dir) / f"fp32_{size}.onnx"
    if not out.exists():
        from ultralytics import YOLO
        h = round(size * 375 / 1242 / 32) * 32   # 03 노트북과 같은 비율
        f = YOLO(args.best_pt).export(format="onnx", imgsz=[h, size], opset=13, simplify=True, dynamic=False)
        shutil.copy(f, out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-pt", required=True)
    parser.add_argument("--base-onnx", required=True, help="학습한 크기의 fp32.onnx")
    parser.add_argument("--base-size", type=int, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dev-list", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ids = [Path(l.strip()).stem for l in Path(args.dev_list).read_text().splitlines() if l.strip()]
    labels = Path(args.labels)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_gts = [read_kitti_file(labels / f"{i}.txt", False) for i in ids]

    report, size_rows = {}, []
    for size in sorted(set(args.sizes) | {args.base_size}):
        work = Path(args.work) / str(size)
        if len(list(work.glob("*.txt"))) != len(ids):
            print(f"입력 {size} 로 dev {len(ids)}장 추론")
            onnx_infer.predict_folder(onnx_for(size, args), args.images, ids, work)
        result, mod_map = kitti_eval.evaluate(work, labels, ids)
        all_dets = [read_kitti_file(work / f"{i}.txt", True) for i in ids]
        a = analyze(all_gts, all_dets)
        found = {t["구분"]: t["찾은 비율(%)"] for t in a["table"]}
        size_rows.append({"입력 크기": size, "Car": round(result["Car"][1], 2), "Pedestrian": round(result["Pedestrian"][1], 2),
                          "Cyclist": round(result["Cyclist"][1], 2), "mAP": round(mod_map, 2),
                          **{f"사람 {k} 찾은 %": found[k] for k in [bin_name(*b) for b in HEIGHT_BINS]}})
        if size == args.base_size:
            rows = iou_table(all_gts, all_dets, a["threshold"])
            print(f"\n[1. 맞힌 사람의 박스 정확도] 입력 {size}, 기준 점수 {a['threshold']}")
            print("IoU 0.5~0.6 은 조금만 어긋나도 놓침이 되는 사람이에요. '아깝게 놓침'은 IoU 0.4~0.5 로 아깝게 틀린 박스예요.")
            print(pd.DataFrame(rows).T.to_string())
            report["iou"] = rows

    print("\n[2. 입력 크기만 키워 보기] dev Moderate AP (재학습 없음, 방향만 보세요)")
    df = pd.DataFrame(size_rows).set_index("입력 크기").T
    print(df.to_string())
    report["sizes"] = size_rows
    (Path(args.out_dir) / "ped_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
