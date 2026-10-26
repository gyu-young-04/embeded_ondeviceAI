"""ONNX 모델의 CPU 추론 속도(latency)와 최대 메모리를 잰다.

- batch 1, 같은 크기의 입력으로 warm-up 후 여러 번 반복해서 잰다.
- 모델 실행 시간만 잰다 (전처리·NMS 제외).
- 모델마다 새 프로세스로 실행해야 메모리 값이 섞이지 않는다.

사용법:
    python scripts/benchmark_onnx.py --model model.onnx --threads 2
"""

import argparse
import json
import resource
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.random.default_rng(0).random((1, 3, inp.shape[2], inp.shape[3]), dtype=np.float32)

    for _ in range(args.warmup):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t) * 1000)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
    times = np.array(times)
    print(json.dumps({
        "latency_ms_mean": round(float(times.mean()), 2),
        "latency_ms_std": round(float(times.std()), 2),
        "latency_ms_p50": round(float(np.median(times)), 2),
        "fps": round(1000 / float(times.mean()), 1),
        "peak_mem_mb": round(peak_rss / 1024, 1),
        "model_mem_mb": round((peak_rss - base_rss) / 1024, 1),
    }))


if __name__ == "__main__":
    main()
