"""클래스마다 NMS IoU 를 따로 골라서 점수가 오르는지 본다. 재학습 없이 CPU 로 돌아간다.

- 값은 dev 로만 고른다. eval(시험지)은 고른 값을 마지막에 한 번 채점할 때만 쓴다.
- NMS 와 AP 는 클래스끼리 서로 영향을 주지 않아서 클래스마다 따로 고를 수 있다.
- dev 에서 --min-gain 보다 덜 오르면 우연일 수 있으므로 기본값(0.5)을 그대로 둔다.

사용법:
    python scripts/tune_nms.py --model int8_qat.onnx --images /content/kitti/training/image_2 \
        --labels /content/kitti/training/label_2 --dev-list /content/kitti_yolo/dev.txt \
        --work /content/nms --out nms_tuned.json
"""

import argparse
import json
from pathlib import Path

import kitti_eval
import onnx_infer
from onnx_infer import CLASSES, NMS_IOU


def read_ids(path):
    return [Path(l.strip()).stem for l in Path(path).read_text().splitlines() if l.strip()]


def score(pred_dir, labels, ids):
    result, mod_map = kitti_eval.evaluate(pred_dir, labels, ids)
    return {c: result[c][1] for c in CLASSES}, mod_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dev-list", required=True)
    parser.add_argument("--eval-list", default=str(Path(__file__).resolve().parent.parent / "eval_val.txt"))
    parser.add_argument("--work", required=True, help="예측 파일을 임시로 둘 폴더")
    parser.add_argument("--out", required=True, help="결과 json")
    parser.add_argument("--ious", type=float, nargs="+", default=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    parser.add_argument("--min-gain", type=float, default=0.05, help="dev AP 가 이만큼은 올라야 값을 바꾼다")
    args = parser.parse_args()

    work = Path(args.work)
    ious = sorted(set(args.ious) | {NMS_IOU})
    dev_ids, eval_ids = read_ids(args.dev_list), read_ids(args.eval_list)

    print(f"dev {len(dev_ids)}장 추론 (IoU {ious})")
    dirs = {iou: work / "dev" / f"{iou:.2f}" for iou in ious}
    onnx_infer.predict_folder_nms(args.model, args.images, dev_ids, dirs)
    dev = {iou: score(d, args.labels, dev_ids)[0] for iou, d in dirs.items()}

    print("\n[dev Moderate AP]\nIoU   " + "".join(f"{c:>12}" for c in CLASSES))
    for iou in ious:
        print(f"{iou:.2f}  " + "".join(f"{dev[iou][c]:12.2f}" for c in CLASSES))

    chosen = []
    for c in CLASSES:
        best = max(ious, key=lambda iou: dev[iou][c])
        chosen.append(best if dev[best][c] - dev[NMS_IOU][c] >= args.min_gain else NMS_IOU)
    chosen = tuple(chosen)
    print("\n고른 IoU:", dict(zip(CLASSES, chosen)))

    dev_before = sum(dev[NMS_IOU].values()) / len(CLASSES)
    dev_after = sum(dev[iou][c] for c, iou in zip(CLASSES, chosen)) / len(CLASSES)
    print(f"dev mAP: {dev_before:.2f} → {dev_after:.2f}")

    report = {"nms_iou": dict(zip(CLASSES, chosen)), "dev_before": dev_before, "dev_after": dev_after}
    if chosen == (NMS_IOU,) * len(CLASSES):
        print("dev 에서 의미 있게 오른 클래스가 없어서 기본값을 그대로 써요. 시험지는 채점하지 않았어요.")
    else:
        dirs = {NMS_IOU: work / "eval" / "base", chosen: work / "eval" / "tuned"}
        onnx_infer.predict_folder_nms(args.model, args.images, eval_ids, dirs)
        before, before_map = score(dirs[NMS_IOU], args.labels, eval_ids)
        after, after_map = score(dirs[chosen], args.labels, eval_ids)
        print("\n[시험지 eval Moderate AP]\n          " + "".join(f"{c:>12}" for c in CLASSES) + f"{'mAP':>10}")
        print("기본 0.5  " + "".join(f"{before[c]:12.2f}" for c in CLASSES) + f"{before_map:10.2f}")
        print("클래스별  " + "".join(f"{after[c]:12.2f}" for c in CLASSES) + f"{after_map:10.2f}")
        report.update(eval_before=before_map, eval_after=after_map, eval_ap_before=before, eval_ap_after=after)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", args.out)


if __name__ == "__main__":
    main()
