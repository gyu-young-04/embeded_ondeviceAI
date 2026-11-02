"""층(블록)별 양자화 민감도 분석.

블록(/model.N/)을 하나씩 FP32 로 남기고 나머지는 INT8 로 양자화해서 dev 점수를 잰다.
점수가 많이 오르는 블록 = 양자화에 약한(민감한) 블록.

- 눈금은 처음 한 번만 재고 파일(calib cache)로 저장해서 재사용한다 → 블록마다 몇십 초.
- 빠르게 하려고 dev 일부(--n-dev 장)로만 순위를 매긴다. 최종 선택은 dev 전체로 다시 확인.

사용법:
    python scripts/sensitivity.py --model fp32.onnx --calib-list train.txt --dev-list dev.txt \
        --images /content/kitti/training/image_2 --labels /content/kitti/training/label_2 --out-dir sens/
"""

import argparse
import shutil
from pathlib import Path

import onnx
import pandas as pd

import kitti_eval
import onnx_infer
from quantize_onnx import layer_ids, prepare, quantize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib-list", required=True)
    parser.add_argument("--dev-list", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", default="Percentile")
    parser.add_argument("--n-dev", type=int, default=300, help="순위 매길 때 쓸 dev 사진 수")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev_ids = [Path(l.strip()).stem for l in open(args.dev_list) if l.strip()][:args.n_dev]
    cache = out / f"calib_{args.method.lower()}.json"
    prep = prepare(args.model, out / "model.onnx")

    def score(model_path, tag):
        pred = out / "preds" / tag
        onnx_infer.predict_folder(model_path, args.images, dev_ids, str(pred))
        result, mod_map = kitti_eval.evaluate(str(pred), args.labels, dev_ids)
        shutil.rmtree(pred, ignore_errors=True)
        return result, mod_map

    def quant_score(blocks, tag):
        q = out / f"{tag}.onnx"
        quantize(prep, q, args.calib_list, method=args.method, keep_head_fp32=True, fp32_blocks=blocks,
                 calib_cache=str(cache))
        r = score(q, tag)
        q.unlink(missing_ok=True)
        return r

    fp32 = score(args.model, "fp32")[1]
    base = quant_score([], "int8_base")[1]
    print(f"dev {len(dev_ids)}장 | FP32 {fp32:.2f} | INT8 {base:.2f} | 손실 {fp32 - base:.2f}")

    rows = []
    blocks = layer_ids(onnx.load(str(prep)))
    for b in blocks:
        result, m = quant_score([b], f"block{b}")
        rows.append({"블록": b, "dev Moderate mAP": round(m, 2), "회복 점수": round(m - base, 2),
                     "Car": round(result["Car"][1], 2), "Pedestrian": round(result["Pedestrian"][1], 2),
                     "Cyclist": round(result["Cyclist"][1], 2)})
        print(f"블록 {b:>2} FP32 → {m:.2f} ({m - base:+.2f})")

    prep.unlink(missing_ok=True)
    table = pd.DataFrame(rows).sort_values("회복 점수", ascending=False)
    table.attrs.update(fp32=fp32, int8=base)
    table.to_csv(out / "sensitivity.csv", index=False)
    (out / "sensitivity_base.txt").write_text(f"fp32 {fp32:.4f}\nint8 {base:.4f}\nn_dev {len(dev_ids)}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
