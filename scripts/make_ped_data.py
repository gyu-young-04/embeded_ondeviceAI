"""사람(Pedestrian) 점수를 올리기 위한 학습 데이터 변형본을 만든다. 원본 데이터는 그대로 둔다.

1. DontCare 가리기
   KITTI 에는 '너무 멀거나 작아서 라벨을 안 단' DontCare 영역이 있다. 그 안에 작은 사람·차가 많은데,
   그대로 학습하면 모델이 '이건 물체가 아니다'라고 잘못 배운다. 채점에서도 DontCare 안의 탐지는 오답으로
   세지 않으므로, 학습 사진에서 이 영역을 회색(114, YOLO 여백색)으로 칠한다.
   단, 라벨이 있는 물체 박스와 겹치는 부분은 원래 픽셀로 되돌려서 물체를 지우지 않는다.
2. 사람 사진 오버샘플링
   사람이 한 명이라도 있는 학습 사진은 '<id>_dup1', '<id>_dup2' ... 이름으로 더 넣어서
   한 바퀴에 --dup 번(기본 2번) 보게 한다.

- train 사진만 바꾼다. dev(학습 중 검증)·eval(채점)은 원본 그대로 쓴다.
- 결과: <out>/images, <out>/labels, <out>/train.txt, <out>/kitti_dc.yaml

사용법:
    python scripts/make_ped_data.py --kitti /content/kitti --src /content/kitti_yolo --out /content/kitti_yolo_dc
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_kitti import CLASS_MAP, CLASS_NAMES

GRAY = 114
PED = CLASS_NAMES.index("Pedestrian")


def read_boxes(label_path):
    """KITTI 라벨에서 (DontCare 박스들, 라벨 있는 물체 박스들)을 픽셀 좌표로 읽는다."""
    dontcare, objects = [], []
    for row in label_path.read_text().splitlines():
        parts = row.split()
        if not parts:
            continue
        box = [float(v) for v in parts[4:8]]
        if parts[0] == "DontCare":
            dontcare.append(box)
        elif parts[0] in CLASS_MAP:
            objects.append(box)
    return dontcare, objects


def to_slice(box, w, h):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    return slice(y1, y2), slice(x1, x2)


def mask_dontcare(img, dontcare, objects):
    """DontCare 영역을 회색으로 칠하고, 물체 박스 부분은 원래 픽셀로 되돌린다."""
    h, w = img.shape[:2]
    out = img.copy()
    for box in dontcare:
        out[to_slice(box, w, h)] = GRAY
    for box in objects:
        s = to_slice(box, w, h)
        out[s] = img[s]
    return out


def link(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti", required=True, help="training/ 폴더가 들어 있는 KITTI 루트")
    parser.add_argument("--src", required=True, help="prepare_kitti.py 로 만든 폴더 (train.txt, dev.txt, labels/)")
    parser.add_argument("--out", required=True, help="변형 데이터를 만들 폴더")
    parser.add_argument("--no-mask", action="store_true", help="DontCare 가리기 끄기")
    parser.add_argument("--dup", type=int, default=2, help="사람 사진을 한 바퀴에 몇 번 볼지 (1 = 오버샘플링 끄기)")
    parser.add_argument("--no-dup", action="store_true", help="--dup 1 과 같음")
    args = parser.parse_args()
    if args.no_dup:
        args.dup = 1
    assert args.dup >= 1

    kitti_label = Path(args.kitti) / "training" / "label_2"
    src = Path(args.src)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    train_imgs = [Path(l.strip()) for l in (src / "train.txt").read_text().splitlines() if l.strip()]
    eval_ids = {Path(l.strip()).stem for l in (src / "eval.txt").read_text().splitlines() if l.strip()}
    assert not {p.stem for p in train_imgs} & eval_ids, "train 에 eval 사진이 섞여 있어요"

    lines, n_masked, n_dup = [], 0, 0
    for img_path in train_imgs:
        image_id = img_path.stem
        yolo_label = src / "labels" / f"{image_id}.txt"
        dst_img = out / "images" / f"{image_id}.png"
        dst_label = out / "labels" / f"{image_id}.txt"

        dontcare, objects = read_boxes(kitti_label / f"{image_id}.txt")
        if dontcare and not args.no_mask:
            img = np.array(Image.open(img_path).convert("RGB"))
            Image.fromarray(mask_dontcare(img, dontcare, objects)).save(dst_img)
            n_masked += 1
        else:
            link(img_path, dst_img)
        shutil.copy(yolo_label, dst_label)
        lines.append(dst_img)

        classes = {int(r.split()[0]) for r in yolo_label.read_text().splitlines() if r.strip()}
        if PED in classes:
            for k in range(1, args.dup):
                dup_img = out / "images" / f"{image_id}_dup{k}.png"
                link(dst_img, dup_img)
                shutil.copy(yolo_label, out / "labels" / f"{image_id}_dup{k}.txt")
                lines.append(dup_img)
                n_dup += 1

    (out / "train.txt").write_text("".join(f"{p}\n" for p in lines))
    # dev(검증)는 원본 그대로: 채점 조건과 같게
    (out / "kitti_dc.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: train.txt\n"
        f"val: {(src / 'dev.txt').resolve()}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    print(f"train 원본 {len(train_imgs)}장 → {len(lines)}장 (사람 사진 {n_dup}장 추가, 사람 사진은 한 바퀴에 {args.dup}번), DontCare 가린 사진 {n_masked}장")
    print(f"데이터셋 설정 파일: {out / 'kitti_dc.yaml'}")


if __name__ == "__main__":
    main()
