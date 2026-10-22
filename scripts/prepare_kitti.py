"""KITTI 2D Object 라벨을 YOLO 형식으로 바꾸고 train / dev / eval 목록을 만든다.

- 클래스 정리: Van -> Car, Person_sitting -> Pedestrian 로 합치고
  Truck, Tram, Misc, DontCare 는 버린다.
  (KITTI 공식 평가에서도 Van 은 Car 의, Person_sitting 은 Pedestrian 의
  '이웃 클래스'로 취급되어 오답으로 세지 않는다.)
- eval_val.txt 의 1,000장은 학습(train)과 학습 중 검증(dev) 어디에도 넣지 않는다.
- 학습 중 검증용 dev 셋은 나머지 사진에서 따로 떼어낸다.

사용법:
    python scripts/prepare_kitti.py --kitti /content/kitti --out /content/kitti_yolo
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from PIL import Image

CLASS_MAP = {
    "Car": 0,
    "Van": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def convert_label(label_path, width, height):
    lines = []
    for row in label_path.read_text().splitlines():
        parts = row.split()
        if not parts or parts[0] not in CLASS_MAP:
            continue
        x1, y1, x2, y2 = map(float, parts[4:8])
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{CLASS_MAP[parts[0]]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def link_or_copy_dir(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti", required=True, help="training/ 폴더가 들어 있는 KITTI 루트")
    parser.add_argument("--out", required=True, help="YOLO 데이터셋을 만들 폴더")
    parser.add_argument("--eval-list", default=str(Path(__file__).resolve().parent.parent / "eval_val.txt"))
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="학습 중 검증(dev)으로 뗄 비율")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kitti = Path(args.kitti)
    image_dir = kitti / "training" / "image_2"
    label_dir = kitti / "training" / "label_2"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Ultralytics 는 이미지 경로의 /images/ 를 /labels/ 로 바꿔서 라벨을 찾는다.
    images_out = out / "images"
    labels_out = out / "labels"
    link_or_copy_dir(image_dir, images_out)
    labels_out.mkdir(exist_ok=True)

    eval_ids = {line.strip() for line in Path(args.eval_list).read_text().splitlines() if line.strip()}
    all_ids = sorted(p.stem for p in image_dir.glob("*.png"))
    missing = eval_ids - set(all_ids)
    if missing:
        raise SystemExit(f"eval_val.txt 의 {len(missing)}개 이미지가 KITTI 폴더에 없습니다. 예: {sorted(missing)[:5]}")

    counts = [0] * len(CLASS_NAMES)
    for image_id in all_ids:
        with Image.open(image_dir / f"{image_id}.png") as img:
            width, height = img.size
        lines = convert_label(label_dir / f"{image_id}.txt", width, height)
        for line in lines:
            counts[int(line.split()[0])] += 1
        (labels_out / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    rest = [i for i in all_ids if i not in eval_ids]
    random.Random(args.seed).shuffle(rest)
    n_dev = int(len(rest) * args.dev_ratio)
    dev_ids, train_ids = sorted(rest[:n_dev]), sorted(rest[n_dev:])
    eval_sorted = sorted(eval_ids)

    # 학습 데이터에 시험 문제(eval_val)가 섞이지 않았는지 한 번 더 확인
    assert not (set(train_ids) | set(dev_ids)) & eval_ids

    for name, ids in (("train", train_ids), ("dev", dev_ids), ("eval", eval_sorted)):
        (out / f"{name}.txt").write_text("".join(f"{images_out / f'{i}.png'}\n" for i in ids))

    (out / "kitti.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: train.txt\n"
        "val: dev.txt\n"
        "test: eval.txt\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )

    print(f"train {len(train_ids)} / dev {len(dev_ids)} / eval {len(eval_sorted)} (eval 은 학습에서 제외됨)")
    print("박스 개수: " + ", ".join(f"{n} {c}" for n, c in zip(CLASS_NAMES, counts)))
    print(f"데이터셋 설정 파일: {out / 'kitti.yaml'}")


if __name__ == "__main__":
    main()
