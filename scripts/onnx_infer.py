"""ONNX 모델로 KITTI 이미지를 추론해서 KITTI 형식 예측 파일을 만든다.

FP32 / INT8 모델을 똑같은 전처리·후처리로 돌려야 공정하게 비교할 수 있어서
Ultralytics 내부 추론 대신 이 코드를 공통으로 쓴다.

- 전처리: 비율을 유지하며 줄이고(letterbox) 남는 곳은 회색(114)으로 채운다.
- 후처리: 점수 필터 → 클래스별 NMS → 원본 이미지 좌표로 되돌리기

사용법:
    python scripts/onnx_infer.py --model model.onnx --images /content/kitti/training/image_2 \
        --ids eval_val.txt --out preds/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["Car", "Pedestrian", "Cyclist"]


def letterbox(img, height, width):
    h, w = img.shape[:2]
    r = min(height / h, width / w)
    nh, nw = round(h * r), round(w * r)
    top = (height - nh) // 2
    left = (width - nw) // 2
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def preprocess(path, height, width):
    img = cv2.imread(str(path))
    boxed, r, left, top = letterbox(img, height, width)
    x = boxed[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), img.shape[:2], (r, left, top)


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def postprocess(output, orig_shape, letterbox_info, conf_thr=0.001, iou_thr=0.7, max_det=100):
    """output: (1, 4 + 클래스 수, N), 박스는 입력 이미지 기준 (cx, cy, w, h)"""
    pred = output[0].T
    boxes_cxcywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
    mask = scores > conf_thr
    boxes_cxcywh, scores, cls_ids = boxes_cxcywh[mask], scores[mask], cls_ids[mask]

    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    keep_all = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep_all.append(idx[nms(boxes[idx], scores[idx], iou_thr)])
    if not keep_all:
        return []
    keep = np.concatenate(keep_all)
    keep = keep[scores[keep].argsort()[::-1][:max_det]]

    r, left, top = letterbox_info
    h, w = orig_shape
    out = []
    for i in keep:
        x1 = np.clip((boxes[i, 0] - left) / r, 0, w)
        y1 = np.clip((boxes[i, 1] - top) / r, 0, h)
        x2 = np.clip((boxes[i, 2] - left) / r, 0, w)
        y2 = np.clip((boxes[i, 3] - top) / r, 0, h)
        if x2 > x1 and y2 > y1:
            out.append((CLASSES[cls_ids[i]], x1, y1, x2, y2, float(scores[i])))
    return out


def to_kitti_lines(dets):
    return "".join(
        f"{name} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {score:.6f}\n"
        for name, x1, y1, x2, y2, score in dets
    )


def make_session(model_path, threads=None):
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(str(model_path), opts, providers=providers)


def predict_folder(model_path, image_dir, ids, out_dir):
    sess = make_session(model_path)
    inp = sess.get_inputs()[0]
    height, width = inp.shape[2], inp.shape[3]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        x, orig_shape, info = preprocess(Path(image_dir) / f"{image_id}.png", height, width)
        output = sess.run(None, {inp.name: x})[0]
        (out_dir / f"{image_id}.txt").write_text(to_kitti_lines(postprocess(output, orig_shape, info)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--ids", default=str(Path(__file__).resolve().parent.parent / "eval_val.txt"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ids = [l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()]
    predict_folder(args.model, args.images, ids, args.out)


if __name__ == "__main__":
    main()
