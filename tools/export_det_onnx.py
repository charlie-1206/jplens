"""偵測模型（RapidOCR 附的 PP-OCRv4 mobile det）改成直接吃 uint8 的 BGR 截圖：前處理在模型裡做（顯示卡上）。

    python tools/export_det_onnx.py

產生 data/models/ppocr_det/det_u8_fp32.onnx、det_u8_fp16.onnx（不進版控；需要 onnx 套件，執行時不需要）。
- 原本：CPU 上 cv2.dnn.blobFromImage（2560×1440 約 40～50 ms）轉成 float32 NCHW 再上傳（44 MB）
- 現在：上傳 uint8 NHWC（11 MB），模型開頭 Cast → Transpose → ×(1/127.5) → −1，和 blobFromImage(x, 1/127.5, mean 127.5) 相同
- fp16：權重與中間運算用 fp16（onnxruntime.transformers.float16，輸入輸出維持 float32），快約 20～30%，
  但框和文字有些不同，預設不用（detect.py）。head 的 BatchNorm 變異數約 9790 萬超出 fp16 範圍：先讓 ONNX Runtime 把它併進卷積
recognize.Recognizer 預設用 det_u8_fp32.onnx（環境變數 JPLENS_DET=fp16 用 fp16、=orig 用原本的，對照測試用）。
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "models" / "ppocr_det"


def source_model() -> Path:
    import rapidocr_onnxruntime
    return Path(rapidocr_onnxruntime.__file__).parent / "models" / "ch_PP-OCRv4_det_infer.onnx"


def prepend_uint8(model: onnx.ModelProto) -> onnx.ModelProto:
    """把 float32 NCHW 的輸入 x 換成 uint8 NHWC 的 img，前面接上轉型、轉置、正規化。"""
    g = model.graph
    old = g.input[0]
    name = old.name
    new_in = helper.make_tensor_value_info("img", TensorProto.UINT8, [1, "h", "w", 3])
    nodes = [
        helper.make_node("Cast", ["img"], ["img_f"], to=TensorProto.FLOAT, name="pre_cast"),
        helper.make_node("Transpose", ["img_f"], ["img_chw"], perm=[0, 3, 1, 2], name="pre_transpose"),
        helper.make_node("Mul", ["img_chw", "pre_scale"], ["img_scaled"], name="pre_mul"),
        helper.make_node("Sub", ["img_scaled", "pre_one"], [name], name="pre_sub"),
    ]
    g.initializer.extend([numpy_helper.from_array(np.array(1 / 127.5, np.float32), "pre_scale"),
                          numpy_helper.from_array(np.array(1.0, np.float32), "pre_one")])
    del g.input[:]
    g.input.extend([new_in])
    for n in reversed(nodes):
        g.node.insert(0, n)
    return model


def toposort(model: onnx.ModelProto) -> onnx.ModelProto:
    """fp16 轉換把輸入的 Cast 節點加在最後面：依相依關係重新排序（onnx.checker 要求）。"""
    g = model.graph
    ready = {i.name for i in g.input} | {i.name for i in g.initializer} | {""}
    todo, out = list(g.node), []
    while todo:
        rest = []
        for n in todo:
            if all(x in ready for x in n.input):
                out.append(n)
                ready.update(n.output)
            else:
                rest.append(n)
        if len(rest) == len(todo):
            raise ValueError("圖裡有循環或缺少的輸入")
        todo = rest
    del g.node[:]
    g.node.extend(out)
    return model


def fold_batchnorm(src: Path) -> Path:
    """先用 ONNX Runtime 的基本最佳化（常數摺疊、BatchNorm 併進前面的卷積）存一份 fp32 模型：
    head 的 BatchNorm 變異數約 9790 萬，超出 fp16 範圍（65504），轉換工具會把它截斷，機率圖整個壞掉；
    併進卷積之後就沒有這麼大的數字。"""
    import onnxruntime as ort
    out = OUT / "det_folded_fp32.onnx"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(out)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    return out


def fp16_blocklist(model: onnx.ModelProto) -> list[str]:
    """超出 fp16 範圍（65504）的常數，以及用到它們的節點，留在 fp32：
    PP-OCRv4 det 的 head 有一個 BatchNorm 的變異數約 9790 萬，轉 fp16 會被截斷，機率圖整個壞掉（二值化後 6～18% 不同）。"""
    for i, n in enumerate(model.graph.node):
        if not n.name:
            n.name = f"node_{i}"
    big = {t.name for t in model.graph.initializer if np.abs(numpy_helper.to_array(t)).max(initial=0) > 60000}
    block = set()
    for n in model.graph.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].t.data_type == TensorProto.FLOAT:
            if np.abs(numpy_helper.to_array(n.attribute[0].t)).max(initial=0) > 60000:
                big.update(n.output)
                block.add(n.name)
    block |= {n.name for n in model.graph.node if big & set(n.input)}
    return sorted(block)


def main():
    from onnxruntime.transformers.float16 import convert_float_to_float16
    OUT.mkdir(parents=True, exist_ok=True)
    src = source_model()
    fp32 = prepend_uint8(onnx.load(str(src)))
    onnx.checker.check_model(fp32)
    onnx.save(fp32, str(OUT / "det_u8_fp32.onnx"))
    half = onnx.load(str(fold_batchnorm(src)))
    block = fp16_blocklist(half)
    half = convert_float_to_float16(half, keep_io_types=True, node_block_list=block)
    half = toposort(prepend_uint8(half))
    onnx.checker.check_model(half)
    onnx.save(half, str(OUT / "det_u8_fp16.onnx"))
    print(f"{src.name} → {OUT}：det_u8_fp32.onnx、det_u8_fp16.onnx（留在 fp32 的節點：{block}）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
