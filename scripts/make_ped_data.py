"""사람(Pedestrian) 점수를 올리기 위한 학습 데이터 변형본을 만든다. 원본 데이터는 그대로 둔다.

1. DontCare 가리기
   KITTI 에는 '너무 멀거나 작아서 라벨을 안 단' DontCare 영역이 있다. 그 안에 작은 사람·차가 많은데,
   그대로 학습하면 모델이 '이건 물체가 아니다'라고 잘못 배운다. 채점에서도 DontCare 안의 탐지는 오답으로
   세지 않으므로, 학습 사진에서 이 영역을 회색(114, YOLO 여백색)으로 칠한다.
   단, 라벨이 있는 물체 박스와 겹치는 부분은 원래 픽셀로 되돌려서 물체를 지우지 않는다.
2. 사람 사진 오버샘플링
   사람이 한 명이라도 있는 학습 사진은 '<id>_dup1', '<id>_dup2' ... 이름으로 더 넣어서
   한 바퀴에 --dup 번(기본 2번) 보게 한다.
   --hard-dup N 을 주면 '어려운 사람'이 있는 사진은 N번 본다.
   어려운 사람 = 대회 채점(Moderate)에는 들어가지만 Easy 에는 못 들어가는 사람 (작거나, 가려졌거나, 잘린 사람).
3. 사람 주변 확대 사진 (--crop N)
   어려운 사람이 있는 사진마다 그 사람 주변을 잘라낸 사진을 N장 더 넣는다. 학습 때 입력 크기로 커지므로
   작은 사람이 --crop-zoom 배(기본 2배) 크게 보인다. 잘려서 절반도 안 남은 물체는 회색으로 칠한다.
4. --full 을 주면 dev 사진도 학습에 넣는다 (설정을 다 정한 뒤 마지막 재학습용). 이때 dev 점수는 믿을 수 없다.

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

from kitti_eval import MAX_OCCLUSION, MAX_TRUNCATION, MIN_HEIGHT
from prepare_kitti import CLASS_MAP, CLASS_NAMES

GRAY = 114
PED = CLASS_NAMES.index("Pedestrian")
EASY, MODERATE = 0, 1


def in_difficulty(parts, diff):
    """KITTI 라벨 한 줄이 해당 난이도의 채점 대상인지 (kitti_eval.py 와 같은 기준)."""
    height = float(parts[7]) - float(parts[5])
    return (int(float(parts[2])) <= MAX_OCCLUSION[diff] and float(parts[1]) <= MAX_TRUNCATION[diff]
            and height > MIN_HEIGHT[diff])


def read_boxes(label_path):
    """KITTI 라벨에서 (DontCare 박스들, 물체 박스들, 물체 클래스들, 어려운 사람 박스들)을 읽는다."""
    dontcare, objects, classes, hard = [], [], [], []
    for row in label_path.read_text().splitlines():
        parts = row.split()
        if not parts:
            continue
        box = [float(v) for v in parts[4:8]]
        if parts[0] == "DontCare":
            dontcare.append(box)
        elif parts[0] in CLASS_MAP:
            objects.append(box)
            classes.append(CLASS_MAP[parts[0]])
            if parts[0] == "Pedestrian" and in_difficulty(parts, MODERATE) and not in_difficulty(parts, EASY):
                hard.append(box)
    return dontcare, objects, classes, hard


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


def make_crop(img, objects, classes, target, zoom, rng):
    """target 사람이 들어가도록 사진을 잘라 (잘린 사진, YOLO 라벨 줄들)을 돌려준다. 못 자르면 None."""
    h, w = img.shape[:2]
    cw, ch = round(w / zoom), round(h / zoom)
    x1, y1, x2, y2 = target
    if x2 - x1 > 0.7 * cw or y2 - y1 > 0.7 * ch:   # 이미 충분히 큰 사람은 확대할 필요 없음
        return None
    left_lo, left_hi = max(0, int(np.ceil(x2)) - cw), min(w - cw, int(np.floor(x1)))
    top_lo, top_hi = max(0, int(np.ceil(y2)) - ch), min(h - ch, int(np.floor(y1)))
    if left_lo > left_hi or top_lo > top_hi:
        return None
    left, top = int(rng.integers(left_lo, left_hi + 1)), int(rng.integers(top_lo, top_hi + 1))

    crop = img[top:top + ch, left:left + cw].copy()
    lines = []
    for box, cls in zip(objects, classes):
        bx1, by1, bx2, by2 = max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)
        ix1, iy1, ix2, iy2 = max(bx1, left), max(by1, top), min(bx2, left + cw), min(by2, top + ch)
        area = (bx2 - bx1) * (by2 - by1)
        if ix2 <= ix1 or iy2 <= iy1 or area <= 0:
            continue
        if (ix2 - ix1) * (iy2 - iy1) / area >= 0.5:
            lines.append(f"{cls} {((ix1 + ix2) / 2 - left) / cw:.6f} {((iy1 + iy2) / 2 - top) / ch:.6f} "
                         f"{(ix2 - ix1) / cw:.6f} {(iy2 - iy1) / ch:.6f}\n")
        else:   # 조금만 보이는 물체는 라벨 없이 남기지 않고 가린다
            crop[to_slice((ix1 - left, iy1 - top, ix2 - left, iy2 - top), cw, ch)] = GRAY
    return crop, lines


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
    parser.add_argument("--hard-dup", type=int, default=0, help="어려운 사람이 있는 사진을 몇 번 볼지 (0 = 구분 안 함)")
    parser.add_argument("--crop", type=int, default=0, help="어려운 사람이 있는 사진마다 확대 사진 몇 장 (0 = 안 함)")
    parser.add_argument("--crop-zoom", type=float, default=2.0, help="확대 배율")
    parser.add_argument("--full", action="store_true", help="dev 사진도 학습에 넣기 (마지막 재학습용)")
    parser.add_argument("--no-dup", action="store_true", help="--dup 1 과 같음")
    args = parser.parse_args()
    if args.no_dup:
        args.dup = 1
    assert args.dup >= 1 and args.hard_dup >= 0 and args.crop >= 0 and args.crop_zoom > 1

    kitti_label = Path(args.kitti) / "training" / "label_2"
    src = Path(args.src)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    list_files = ["train.txt", "dev.txt"] if args.full else ["train.txt"]
    train_imgs = [Path(l.strip()) for f in list_files for l in (src / f).read_text().splitlines() if l.strip()]
    eval_ids = {Path(l.strip()).stem for l in (src / "eval.txt").read_text().splitlines() if l.strip()}
    assert not {p.stem for p in train_imgs} & eval_ids, "train 에 eval 사진이 섞여 있어요"

    lines, n_masked, n_dup, n_hard, n_crop = [], 0, 0, 0, 0
    rng = np.random.default_rng(0)
    for img_path in train_imgs:
        image_id = img_path.stem
        yolo_label = src / "labels" / f"{image_id}.txt"
        dst_img = out / "images" / f"{image_id}.png"
        dst_label = out / "labels" / f"{image_id}.txt"

        dontcare, objects, box_classes, hard = read_boxes(kitti_label / f"{image_id}.txt")
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
            n_hard += bool(hard)
            times = args.hard_dup if hard and args.hard_dup else args.dup
            for k in range(1, times):
                dup_img = out / "images" / f"{image_id}_dup{k}.png"
                link(dst_img, dup_img)
                shutil.copy(yolo_label, out / "labels" / f"{image_id}_dup{k}.txt")
                lines.append(dup_img)
                n_dup += 1

        if hard and args.crop:
            img = np.array(Image.open(dst_img).convert("RGB"))
            for k in range(args.crop):
                target = hard[int(rng.integers(len(hard)))]
                made = make_crop(img, objects, box_classes, target, args.crop_zoom, rng)
                if made is None:
                    continue
                crop_img = out / "images" / f"{image_id}_crop{k}.png"
                Image.fromarray(made[0]).save(crop_img)
                (out / "labels" / f"{image_id}_crop{k}.txt").write_text("".join(made[1]))
                lines.append(crop_img)
                n_crop += 1

    (out / "train.txt").write_text("".join(f"{p}\n" for p in lines))
    # dev(검증)는 원본 그대로: 채점 조건과 같게
    (out / "kitti_dc.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: train.txt\n"
        f"val: {(src / 'dev.txt').resolve()}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    print(f"train 원본 {len(train_imgs)}장 → {len(lines)}장 (복사본 {n_dup}장 추가, 사람 사진 {args.dup}번, 어려운 사람 사진 {n_hard}장은 {args.hard_dup or args.dup}번, 확대 사진 {n_crop}장), DontCare 가린 사진 {n_masked}장")
    print(f"데이터셋 설정 파일: {out / 'kitti_dc.yaml'}")


if __name__ == "__main__":
    main()
