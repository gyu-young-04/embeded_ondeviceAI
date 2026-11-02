"""QAT 로 export 한 ONNX 의 가중치를 실제 INT8 로 저장해서 파일 크기를 줄인다.

Ultralytics(ModelOpt) QAT export 는 가중치를 FP32 그대로 두고
'FP32 가중치 → QuantizeLinear → DequantizeLinear' 로 저장한다. 그래서 파일 크기가 FP32 와 같다.
여기서는 가중치 쪽 QuantizeLinear 를 미리 계산해서 INT8 가중치 → DequantizeLinear 로 바꾼다.
계산 결과는 똑같고(같은 반올림), 파일만 약 1/4 로 줄어든다.

사용법:
    python scripts/fold_qat_weights.py --model qat.onnx --out qat_int8.onnx
"""

import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = onnx.load(args.model)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    consumers = {}
    for n in graph.node:
        for name in n.input:
            consumers.setdefault(name, []).append(n)

    folded, remove_nodes, remove_inits = 0, [], set()
    for node in graph.node:
        if node.op_type != "QuantizeLinear" or node.input[0] not in inits:
            continue
        x = numpy_helper.to_array(inits[node.input[0]]).astype(np.float32)
        scale = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32)
        zp = numpy_helper.to_array(inits[node.input[2]]) if len(node.input) > 2 and node.input[2] else np.array(0, np.uint8)
        axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
        if scale.ndim == 1 and scale.size > 1:  # 채널별(per-channel)
            shape = [1] * x.ndim
            shape[axis] = -1
            scale, zp_b = scale.reshape(shape), zp.reshape(shape)
        else:
            zp_b = zp
        info = np.iinfo(zp.dtype)
        q = np.clip(np.rint(x / scale) + zp_b.astype(np.int32), info.min, info.max).astype(zp.dtype)
        q_name = node.output[0]  # DequantizeLinear 가 읽던 이름을 그대로 INT8 가중치 이름으로 쓴다
        graph.initializer.append(numpy_helper.from_array(q, q_name))
        remove_nodes.append(node)
        if len(consumers.get(node.input[0], [])) == 1:
            remove_inits.add(node.input[0])
        folded += 1

    for n in remove_nodes:
        graph.node.remove(n)
    for i in [i for i in graph.initializer if i.name in remove_inits]:
        graph.initializer.remove(i)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"가중치 {folded}개를 INT8 로 저장: {args.out}")


if __name__ == "__main__":
    main()
