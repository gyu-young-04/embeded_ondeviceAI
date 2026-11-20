"""학습된 모델을 train 사진에 돌려서, 모델이 실제로 놓치는 사람을 찾는다. dev·eval 은 쓰지 않는다.

찾은 결과는 make_ped_data.py --mined 로 넘겨서 그 사진과 확대 사진을 학습에 더 넣는다.
train 사진은 이미 학습에 쓴 사진이라 점수가 높게 나오므로, 기준 점수를 조금 높게(기본 0.5) 잡는다.

사용법:
    python scripts/mine_hard.py --weights best.pt --list /tmp/kitti_yolo/train.txt \
        --labels /tmp/kitti/training/label_2 --out mined.json --imgsz 1280 --device 0
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from analyze_ped import match_image
from kitti_eval import read_kitti_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--list", required=True, help="train 사진 목록")
    parser.add_argument("--labels", required=True, help="KITTI label_2 폴더")
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--thr", type=float, default=0.5, help="이 점수 미만이면 놓친 것으로 본다")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    paths = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip()]
    labels = Path(args.labels)
    model = YOLO(args.weights)
    mined, causes, n_people = {}, Counter(), 0
    # 목록(list)을 넘기면 Ultralytics 가 사진을 전부 메모리에 올려서 터져요. 목록 파일을 넘겨 한 장씩 읽게 해요
    for r in model.predict(source=args.list, stream=True, imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=100,
                           device=args.device, verbose=False):
        image_id = Path(r.path).stem
        dets = [{"name": r.names[int(c)], "bbox": [float(v) for v in box], "score": float(s)}
                for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        missed, _, found = match_image(read_kitti_file(labels / f"{image_id}.txt", False), dets, args.thr)
        n_people += len(missed) + len(found)
        if missed:
            mined[image_id] = [g["bbox"] for g, _ in missed]
            causes.update(c for _, c in missed)

    n_missed = sum(len(v) for v in mined.values())
    print(f"train {len(paths)}장 | 채점 대상 사람 {n_people}명 | 놓친 사람 {n_missed}명 | 해당 사진 {len(mined)}장")
    print("원인:", dict(causes))
    assert mined, "놓친 사람이 없어요. --thr 를 높여 보세요."
    Path(args.out).write_text(json.dumps({"thr": args.thr, "images": mined}), encoding="utf-8")


if __name__ == "__main__":
    main()
