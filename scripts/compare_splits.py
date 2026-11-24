"""train / dev / eval 에 어떤 물체가 얼마나 있는지 비교한다. 정답 라벨만 쓴다 (모델 필요 없음).

dev 점수로 설정을 고르는데, dev 가 eval 과 많이 다르면 dev 에서 좋은 설정이 eval 에서 안 좋을 수 있다.
예: dev 자전거 98.6 / eval 자전거 94.3. 자전거가 dev 에 적거나 쉬우면 이런 차이가 난다.

세는 것 (클래스마다, 채점 대상 = KITTI Moderate 기준과 같게):
- 채점 대상 수, 사진 1장당 수, '1개 놓치면 AP 몇 점' (대략 100 / 대상 수)
- 어려운 비율 (Moderate 이지만 Easy 는 아닌 것), 높이 구간, 가림 1, 잘림
- 붙어 있는 비율 (같은 클래스 다른 물체와 IoU 0.1 이상)
- 헷갈리는 이웃 (Van, Person_sitting), DontCare 수

사용법:
    python scripts/compare_splits.py --labels /content/kitti/training/label_2 \
        --split train=/content/kitti_yolo/train.txt --split dev=/content/kitti_yolo/dev.txt --split eval=eval_val.txt
"""

import argparse
import json
from pathlib import Path

from kitti_eval import CLASSES, MAX_OCCLUSION, MAX_TRUNCATION, MIN_HEIGHT, box_overlap, read_kitti_file

EASY, MOD = 0, 1
HEIGHT_BINS = [(25, 40), (40, 60), (60, 100), (100, 10000)]


def h(o):
    return o["bbox"][3] - o["bbox"][1]


def ok(o, d):
    return o["occluded"] <= MAX_OCCLUSION[d] and o["truncated"] <= MAX_TRUNCATION[d] and h(o) > MIN_HEIGHT[d]


def stats(all_gts):
    n_img = len(all_gts)
    res = {}
    for cls in CLASSES:
        mod, easy, occ1, trunc, crowd, imgs = 0, 0, 0, 0, 0, 0
        bins = [0] * len(HEIGHT_BINS)
        for gts in all_gts:
            same = [o for o in gts if o["name"] == cls]
            targets = [o for o in same if ok(o, MOD)]
            imgs += bool(targets)
            for o in targets:
                mod += 1
                easy += ok(o, EASY)
                occ1 += o["occluded"] == 1
                trunc += o["truncated"] > 0.15
                crowd += any(p is not o and box_overlap(o["bbox"], p["bbox"], -1) >= 0.1 for p in same)
                for k, (lo, hi) in enumerate(HEIGHT_BINS):
                    bins[k] += lo < h(o) <= hi
        pct = lambda v: round(100 * v / mod, 1) if mod else None
        res[cls] = {
            "채점 대상 수": mod,
            "대상이 있는 사진 수": imgs,
            "사진 1장당 대상": round(mod / n_img, 2),
            "1개 놓치면 AP 약 (점)": round(100 / mod, 2) if mod else None,
            "어려운 비율 % (Moderate만)": pct(mod - easy),
            **{f"높이 {lo}~{hi}px %" if hi < 10000 else f"높이 {lo}px 이상 %": pct(b) for (lo, hi), b in zip(HEIGHT_BINS, bins)},
            "가림 1 %": pct(occ1),
            "잘림 0.15 초과 %": pct(trunc),
            "붙어 있는 비율 %": pct(crowd),
        }
    names = [o["name"] for gts in all_gts for o in gts]
    res["기타"] = {"사진 수": n_img,
                 "Van (차 이웃, 채점 제외)": names.count("Van"),
                 "Person_sitting (사람 이웃, 채점 제외)": names.count("Person_sitting"),
                 "DontCare / 사진": round(names.count("DontCare") / n_img, 2)}
    return res


def table(title, rows, splits):
    keys = list(next(iter(rows.values())).keys())
    w = max(len(k) for k in keys) + 2
    out = [f"\n[{title}]", " " * w + "".join(f"{s:>10}" for s in splits)]
    for k in keys:
        out.append(f"{k:<{w}}" + "".join(f"{str(rows[s][k]):>10}" for s in splits))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", action="append", required=True, help="이름=목록파일 (사진 경로나 번호가 한 줄에 하나)")
    parser.add_argument("--out", default=None, help="결과 json")
    args = parser.parse_args()

    labels = Path(args.labels)
    data = {}
    for s in args.split:
        name, path = s.split("=", 1)
        ids = [Path(l.strip()).stem for l in Path(path).read_text().splitlines() if l.strip()]
        data[name] = stats([read_kitti_file(labels / f"{i}.txt", False) for i in ids])
    splits = list(data)

    text = "\n".join(table(c, {s: data[s][c] for s in splits}, splits) for c in CLASSES + ["기타"])
    text += "\n\n읽는 법:\n"
    text += "- '1개 놓치면 AP 약' 이 클수록 그 클래스 점수가 몇 개 차이로 크게 흔들려요 (dev 결과를 덜 믿어야 해요).\n"
    text += "- dev 와 eval 의 어려운 비율·작은 물체 비율·붙어 있는 비율이 많이 다르면, dev 에서 고른 설정이 eval 에선 다를 수 있어요.\n"
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
