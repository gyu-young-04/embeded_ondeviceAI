"""QAT 모델의 양자화 손실이 어디서 생기는지 나눠서 잰다. dev 사진만 쓴다.

QAT export ONNX 는 계산마다 'QuantizeLinear → DequantizeLinear'(Q/DQ) 쌍으로 INT8 을 흉내 낸다.
이 쌍을 빼면 그 부분만 FP32 로 계산된다. 빼는 범위를 바꿔 가며 dev 점수를 잰다.

1. 손실 나누기
   - FP32 (출발 모델)             : QAT 전 원래 모델
   - QAT 가중치, 양자화 전부 끔   : QAT 로 바뀐 가중치를 FP32 로 계산 → 출발 모델과의 차이 = '학습이 바뀐 손실'
   - QAT INT8 그대로              : 위와의 차이 = '양자화 잡음 손실'
   - 가중치만 FP32 / 활성값만 FP32: 둘 중 어느 쪽 양자화가 더 아픈지
2. 블록별: 블록(/model.N/) 하나만 FP32 로 계산했을 때 얼마나 회복되는지. 마지막 탐지 헤드는
   cv2(박스) / cv3(분류 점수) 등으로 더 잘게 나눈다.
3. 누적: 회복이 큰 블록부터 1·2·3·5 개를 FP32 로 두면 얼마나 회복되는지 (본선 Hailo 에서 16bit 로 둘 후보).

끊겨도 다시 실행하면 이미 잰 것은 results.json 에서 읽고 건너뛴다.

사용법:
    python scripts/qat_sensitivity.py --qat-onnx qat_qdq.onnx --fp32 fp32.onnx --dev-list dev.txt \
        --images /content/kitti/training/image_2 --labels /content/kitti/training/label_2 \
        --work /content/qat_sens --out-dir <결과 폴더>
"""

import argparse
import json
import re
import shutil
from pathlib import Path

HEAD_SPLIT = True
CUMULATIVE = [1, 2, 3, 5]


def node_group(name, head):
    """노드 이름 → 블록 이름. '/model.7/...' → '7', 헤드는 '23.cv3' 처럼 가지까지."""
    m = re.match(r"/model\.(\d+)/([^/]+)?", name or "")
    if not m:
        return "기타"
    n = int(m.group(1))
    if HEAD_SPLIT and n == head and m.group(2):
        sub = re.match(r"[a-zA-Z]+\d*", m.group(2))
        return f"{n}.{sub.group() if sub else m.group(2)}"
    return str(n)


def group_key(g):
    """'2' < '10' < '23.cv2' < '23.cv3' < '기타' 순서"""
    m = re.match(r"(\d+)(?:\.(.*))?$", g)
    return (0, int(m.group(1)), m.group(2) or "") if m else (1, 0, g)


def find_pairs(nodes, init_names):
    """Q/DQ 쌍 목록. 각 쌍: {q, dq, x(원래 입력), weight(가중치 여부), group}"""
    producer = {o: n for n in nodes for o in n.output}
    consumers = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    ids = [int(m.group(1)) for n in nodes if (m := re.match(r"/model\.(\d+)/", n.name or ""))]
    head = max(ids) if ids else -1
    pairs = []
    for d in nodes:
        if d.op_type != "DequantizeLinear":
            continue
        q = producer.get(d.input[0])
        if q is None or q.op_type != "QuantizeLinear":
            continue   # 가중치가 이미 INT8 로 접힌(fold) 파일이면 여기로 와요 → 접기 전 파일을 써야 해요
        x = q.input[0]
        src = producer.get(x)
        weight = x in init_names or (src is not None and src.op_type == "Constant")
        groups = {node_group(c.name, head) for c in consumers.get(d.output[0], [])}
        pairs.append({"q": q, "dq": d, "x": x, "weight": weight,
                      "group": groups.pop() if len(groups) == 1 else "여러 블록"})
    return pairs, head


def bypass(nodes, graph_outputs, pairs):
    """pairs 의 Q/DQ 를 빼고 원래 입력을 바로 넘긴다. 남길 노드 목록을 돌려준다(입력 이름은 제자리에서 바꿈)."""
    rename, drop = {}, set()
    for p in pairs:
        if p["dq"].output[0] in graph_outputs:
            continue
        rename[p["dq"].output[0]] = p["x"]
        drop.add(id(p["dq"]))
    for n in nodes:
        if id(n) in drop:
            continue
        for k, i in enumerate(n.input):
            if i in rename:
                n.input[k] = rename[i]
    used = {i for n in nodes if id(n) not in drop for i in n.input}
    for p in pairs:   # Q 의 출력을 아무도 안 쓰면 Q 도 지워요
        if id(p["dq"]) in drop and p["q"].output[0] not in used and p["q"].output[0] not in graph_outputs:
            drop.add(id(p["q"]))
    return [n for n in nodes if id(n) not in drop]


def main():
    import onnx

    import kitti_eval
    import onnx_infer

    parser = argparse.ArgumentParser()
    parser.add_argument("--qat-onnx", required=True, help="가중치를 접기(fold) 전 QAT export ONNX")
    parser.add_argument("--fp32", required=True, help="출발 모델의 fp32.onnx")
    parser.add_argument("--dev-list", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-dev", type=int, default=0, help="0 = dev 전체")
    args = parser.parse_args()

    ids = [Path(l.strip()).stem for l in Path(args.dev_list).read_text().splitlines() if l.strip()]
    ids = ids[:args.n_dev] if args.n_dev else ids
    out, work = Path(args.out_dir), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cache_f = out / "results.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    if cache.get("n_dev") != len(ids):
        cache = {"n_dev": len(ids), "runs": {}}

    def score(tag, model_path):
        if tag in cache["runs"]:
            return cache["runs"][tag]
        pred = work / "preds" / re.sub(r"\W", "_", tag)
        onnx_infer.predict_folder(model_path, args.images, ids, str(pred))
        result, m = kitti_eval.evaluate(str(pred), args.labels, ids)
        shutil.rmtree(pred, ignore_errors=True)
        r = {"mAP": round(m, 2), **{c: round(result[c][1], 2) for c in kitti_eval.CLASSES}}
        cache["runs"][tag] = r
        cache_f.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {tag:<28} dev {r['mAP']:.2f} (Car {r['Car']:.2f} / Ped {r['Pedestrian']:.2f} / Cyc {r['Cyclist']:.2f})",
              flush=True)
        return r

    base = onnx.load(args.qat_onnx)
    init_names = {i.name for i in base.graph.initializer}
    pairs, head = find_pairs(list(base.graph.node), init_names)
    assert pairs, "Q/DQ 쌍이 없어요. QAT export 파일(접기 전)을 넣었는지 확인하세요."
    order = {}
    for p in pairs:
        order.setdefault(p["group"], []).append(p)
    print(f"Q/DQ {len(pairs)}쌍 (가중치 {sum(p['weight'] for p in pairs)} / 활성값 {sum(not p['weight'] for p in pairs)})"
          f" | 블록 {len(order)}개 | 헤드 = model.{head} | 여러 블록에 걸친 쌍 {len(order.get('여러 블록', []))}개")

    def score_without(tag, pick):
        if tag in cache["runs"]:
            return cache["runs"][tag]
        m = onnx.load(args.qat_onnx)   # 매번 새로 읽어서 고쳐요
        nodes = list(m.graph.node)
        ps, _ = find_pairs(nodes, {i.name for i in m.graph.initializer})
        keep = bypass(nodes, {o.name for o in m.graph.output}, [p for p in ps if pick(p)])
        del m.graph.node[:]
        m.graph.node.extend(keep)
        f = work / "tmp.onnx"
        onnx.save(m, str(f))
        r = score(tag, f)
        f.unlink(missing_ok=True)
        return r

    print(f"\n[1. 손실 나누기] dev {len(ids)}장")
    fp32 = score("FP32 (출발 모델)", args.fp32)
    off = score_without("QAT 가중치, 양자화 전부 끔", lambda p: True)
    qat = score("QAT INT8 그대로", args.qat_onnx)
    w_fp = score_without("가중치만 FP32", lambda p: p["weight"])
    a_fp = score_without("활성값만 FP32", lambda p: not p["weight"])

    print(f"\n[2. 블록 하나씩 FP32] 기준 = QAT INT8 {qat['mAP']:.2f}")
    rows = []
    for g in sorted(order, key=group_key):
        r = score_without(f"블록 {g} FP32", lambda p, g=g: p["group"] == g)
        rows.append({"블록": g, "Q/DQ 수": len(order[g]), "dev mAP": r["mAP"], "회복": round(r["mAP"] - qat["mAP"], 2),
                     "Car": round(r["Car"] - qat["Car"], 2), "Ped": round(r["Pedestrian"] - qat["Pedestrian"], 2),
                     "Cyc": round(r["Cyclist"] - qat["Cyclist"], 2)})
    rows.sort(key=lambda r: -r["회복"])

    print(f"\n[3. 회복 큰 블록부터 여러 개 FP32]")
    cum = []
    for k in CUMULATIVE:
        top = [r["블록"] for r in rows[:k] if r["블록"] != "여러 블록"]
        r = score_without(f"상위 {k}개 FP32 {top}", lambda p, top=top: p["group"] in top)
        cum.append({"FP32 블록 수": k, "블록": top, "dev mAP": r["mAP"], "회복": round(r["mAP"] - qat["mAP"], 2),
                    "Ped": r["Pedestrian"], "Cyc": r["Cyclist"]})

    total = fp32["mAP"] - qat["mAP"]
    lines = [
        "\n================ 결과 ================",
        f"dev {len(ids)}장 | FP32 {fp32['mAP']:.2f} → QAT INT8 {qat['mAP']:.2f} (손실 {total:.2f})",
        f"  ① 학습이 바뀐 손실  (FP32 → QAT 가중치 FP32 계산): {fp32['mAP'] - off['mAP']:+.2f} 점 줄어듦"
        f"  [Ped {off['Pedestrian'] - fp32['Pedestrian']:+.2f}, Cyc {off['Cyclist'] - fp32['Cyclist']:+.2f}]",
        f"  ② 양자화 잡음 손실 (QAT 가중치 FP32 → INT8)     : {off['mAP'] - qat['mAP']:+.2f} 점 줄어듦"
        f"  [Ped {qat['Pedestrian'] - off['Pedestrian']:+.2f}, Cyc {qat['Cyclist'] - off['Cyclist']:+.2f}]",
        f"  가중치만 FP32 로 두면 {w_fp['mAP'] - qat['mAP']:+.2f}, 활성값만 FP32 로 두면 {a_fp['mAP'] - qat['mAP']:+.2f} 회복",
        "",
        "읽는 법:",
        "  ① 이 크면 QAT 학습이 모델을 망가뜨린 것 → QAT_KD(증류), 학습률 낮추기, 바퀴 수 조정이 해결책",
        "  ② 가 크면 INT8 표현 자체가 부족한 것 → 아래 블록을 16bit/FP32 로 두기(본선 Hailo), 눈금 방식 조정",
        "  사람 344명뿐이라 0.1~0.2 차이는 우연일 수 있어요.",
        "",
        "[블록 하나씩 FP32 — 회복 큰 순서, Car/Ped/Cyc 는 클래스별 회복]",
    ]
    cols = ["블록", "Q/DQ 수", "dev mAP", "회복", "Car", "Ped", "Cyc"]
    lines.append("  ".join(f"{c:>8}" for c in cols))
    lines += ["  ".join(f"{str(r[c]):>8}" for c in cols) for r in rows]
    lines.append("\n[회복 큰 블록부터 여러 개 FP32]")
    lines += [f"  {c['FP32 블록 수']}개 {c['블록']}: dev {c['dev mAP']:.2f} ({c['회복']:+.2f}) Ped {c['Ped']:.2f} Cyc {c['Cyc']:.2f}"
              for c in cum]
    text = "\n".join(lines)
    print(text)
    (out / "qat_sensitivity.txt").write_text(text, encoding="utf-8")
    (out / "qat_sensitivity.json").write_text(json.dumps(
        {"fp32": fp32, "qat_fp32_math": off, "qat_int8": qat, "weights_fp32": w_fp, "acts_fp32": a_fp,
         "blocks": rows, "cumulative": cum}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
