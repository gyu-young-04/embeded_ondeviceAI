"""ONNX 모델을 INT8로 정적 양자화(PTQ)한다.

- Calibration: 학습(train) 사진 일부로 숫자 범위를 잰다. 시험지(eval)는 쓰지 않는다.
- --keep-head-fp32: 탐지 헤드의 박스 계산(후처리) 부분은 FP32로 남긴다.
  박스 좌표(0~960)와 점수(0~1)가 한 텐서에 합쳐지는 곳을 INT8 하나로 표현하면
  점수가 뭉개지기 쉬워서, 이 부분만 빼는 혼합 정밀도(mixed precision) 실험용이다.
- --fp32-blocks: 민감도 분석에서 찾은 '양자화에 약한 블록'(/model.N/)을 FP32 로 남긴다.
- --calib-cache: 잰 눈금을 파일로 저장해 두고 다음에 다시 쓴다 (민감도 분석에서 매번 다시 재지 않으려고).

사용법:
    python scripts/quantize_onnx.py --model fp32.onnx --out int8.onnx --calib-list /content/kitti_yolo/train.txt
"""

import argparse
import random
from pathlib import Path

import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.calibrate import HistogramCalibrater
from onnxruntime.quantization.shape_inference import quant_pre_process

from onnx_infer import preprocess

_ORIGINAL_COLLECT = HistogramCalibrater.collect_data  # 여러 번 패치해도 원본을 한 번만 감싸도록


class ImageReader(CalibrationDataReader):
    def __init__(self, paths, input_name, height, width):
        self.paths = iter(paths)
        self.input_name = input_name
        self.height, self.width = height, width

    def get_next(self):
        path = next(self.paths, None)
        if path is None:
            return None
        return {self.input_name: preprocess(path, self.height, self.width)[0]}


class ListReader(CalibrationDataReader):
    def __init__(self, items):
        self.items = iter(items)

    def get_next(self):
        return next(self.items, None)


def chunk_histogram_calibration(chunk):
    """Percentile / Entropy 는 사진 전체의 중간 계산값을 한꺼번에 메모리에 쌓아서 Colab RAM 이 터진다.
    사진을 chunk 장씩 나눠 넘기면 히스토그램이 누적되므로 결과는 같고 메모리만 줄어든다."""
    original = _ORIGINAL_COLLECT

    def collect_data(self, data_reader):
        nxt = data_reader.get_next()
        while nxt is not None:
            batch = [nxt]
            nxt = data_reader.get_next()
            while nxt is not None and len(batch) < chunk:
                batch.append(nxt)
                nxt = data_reader.get_next()
            original(self, ListReader(batch))

    HistogramCalibrater.collect_data = collect_data


def head_nodes_to_exclude(model):
    """Detect 레이어(마지막 레이어)에서 Conv 가지(cv2, cv3 …)를 뺀 나머지 = 박스 계산(후처리) 노드"""
    names = [n.name for n in model.graph.node]
    layer_ids = [int(n.split("/")[1].split(".")[1]) for n in names if n.startswith("/model.")]
    head = f"/model.{max(layer_ids)}/"
    return [n for n in names if n.startswith(head) and not n[len(head):].startswith("cv")]


def block_nodes(model, blocks):
    """/model.N/ 블록 안의 노드 이름들"""
    prefixes = tuple(f"/model.{b}/" for b in blocks)
    return [n.name for n in model.graph.node if n.name.startswith(prefixes)] if prefixes else []


def layer_ids(model):
    return sorted({int(n.name.split("/")[1].split(".")[1]) for n in model.graph.node if n.name.startswith("/model.")})


def prepare(model_path, out_path):
    prep = Path(out_path).with_suffix(".prep.onnx")
    quant_pre_process(str(model_path), str(prep))
    return prep


def quantize(prep, out, calib_list, calib_count=300, method="MinMax", keep_head_fp32=False, fp32_blocks=(),
             seed=0, calib_chunk=10, calib_cache=None):
    """prep: quant_pre_process 를 거친 모델 경로. FP32 로 남긴 노드 수를 돌려준다."""
    model = onnx.load(str(prep))
    inp = model.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    height, width = dims[2], dims[3]

    reader = None
    if not (calib_cache and Path(calib_cache).is_file()):
        paths = [l.strip() for l in Path(calib_list).read_text().splitlines() if l.strip()]
        random.Random(seed).shuffle(paths)
        reader = ImageReader(paths[:calib_count], inp.name, height, width)

    chunk_histogram_calibration(calib_chunk)
    exclude = head_nodes_to_exclude(model) if keep_head_fp32 else []
    exclude = sorted(set(exclude) | set(block_nodes(model, fp32_blocks)))
    quantize_static(
        str(prep), str(out), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=getattr(CalibrationMethod, method),
        nodes_to_exclude=exclude,
        calibration_cache_path=calib_cache,
    )
    return len(exclude)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib-list", required=True, help="calibration 에 쓸 이미지 경로 목록 (train.txt)")
    parser.add_argument("--calib-count", type=int, default=300)
    parser.add_argument("--method", default="MinMax", choices=["MinMax", "Entropy", "Percentile"])
    parser.add_argument("--keep-head-fp32", action="store_true")
    parser.add_argument("--calib-chunk", type=int, default=10,
                        help="Percentile / Entropy 에서 몇 장씩 나눠 계산할지 (메모리 절약)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32-blocks", type=int, nargs="*", default=[], help="FP32 로 남길 블록 번호 (예: 10 22)")
    parser.add_argument("--calib-cache", default=None, help="눈금 저장 파일 (.json). 있으면 다시 재지 않음")
    args = parser.parse_args()

    prep = prepare(args.model, args.out)
    n = quantize(prep, args.out, args.calib_list, args.calib_count, args.method, args.keep_head_fp32,
                 args.fp32_blocks, args.seed, args.calib_chunk, args.calib_cache)
    prep.unlink(missing_ok=True)
    print(f"INT8 저장: {args.out} (calibration {args.calib_count}장, {args.method}, "
          f"FP32 블록 {args.fp32_blocks or '없음'}, FP32로 남긴 노드 {n}개)")


if __name__ == "__main__":
    main()
