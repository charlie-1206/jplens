"""整個螢幕的文字偵測（按下 Alt 時的版面），比 RapidOCR 原本的流程快：
- 模型：tools/export_det_onnx.py 改過的 PP-OCRv4 det，直接吃 uint8 的 BGR 截圖，轉型、轉置、正規化在顯示卡上做
  （原本 CPU 上 cv2.dnn.blobFromImage 2560×1440 約 40～50 ms，上傳的資料也是 4 倍大）。
  預設 fp32：select_eval 讀出的文字和原本完全相同、box_eval 相同（框有 1.5% 差一個像素）。
  fp16 再快約 20～30%，但 21% 的框不同、select_eval 有幾處文字變了（「…おわ。」→「。。おお。」），預設不用
- 後處理（機率圖 → 框）照舊用 RapidOCR 的 DBPostProcess。試過用連通區域＋積分圖改寫：機率圖裡的字塊多半是圓角、
  不規則的（一半填不滿外框的八成），還是得逐個找輪廓，反而比較慢（120 張平均 39 → 49 ms），沒有採用
沒有改過的模型、沒有顯示卡（Recognizer.det_device 不是 GPU）或環境變數 JPLENS_DET=orig 時不用這個模組。
"""

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models" / "ppocr_det"


def model_path() -> Path | None:
    """要用的模型：預設 fp32，環境變數 JPLENS_DET=fp16 用 fp16、=orig 不用（照舊 RapidOCR）。"""
    want = os.environ.get("JPLENS_DET", "fp32")
    if want == "orig":
        return None
    path = MODEL_DIR / f"det_u8_{want}.onnx"
    return path if path.exists() else None


class FastDet:
    def __init__(self, path: Path, providers):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        from . import gpu
        gpu.session_options(so)
        self.path = path
        self.sess = ort.InferenceSession(str(path), so, providers=providers)
        self.out_name = self.sess.get_outputs()[0].name

    def prob(self, img: np.ndarray) -> np.ndarray:
        """機率圖（長寬調成 32 的倍數後的大小，和 RapidOCR 的 DetPreProcess 相同：四捨五入到 32 的倍數再縮放）。"""
        h, w = img.shape[:2]
        rh, rw = int(round(h / 32) * 32), int(round(w / 32) * 32)
        x = img if (rh, rw) == (h, w) else cv2.resize(img, (rw, rh))
        io = self.sess.io_binding()
        io.bind_cpu_input("img", np.ascontiguousarray(x)[None])
        io.bind_output(self.out_name)
        self.sess.run_with_iobinding(io)
        return io.copy_outputs_to_cpu()[0][0, 0]


# ---------------------------------------------------------------- 後處理（和 RapidOCR 的 DBPostProcess 完全相同的數學）
# 原本每個輪廓算分數、外擴時有不少 numpy 小陣列與 shapely 的開銷（字多的畫面 300 多個輪廓、約 50 ms）。
# 這裡照抄同樣的步驟，只把 4 個點的最小值、面積、周長改用 Python 算；結果逐框相同（test_core 有核對）。

def _score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """同 DBPostProcess.box_score_fast。"""
    h, w = bitmap.shape[:2]
    xs, ys = box[:, 0].tolist(), box[:, 1].tolist()
    xmin = min(max(int(np.floor(min(xs))), 0), w - 1)
    xmax = min(max(int(np.ceil(max(xs))), 0), w - 1)
    ymin = min(max(int(np.floor(min(ys))), 0), h - 1)
    ymax = min(max(int(np.ceil(max(ys))), 0), h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    b = box.copy()
    b[:, 0] -= xmin
    b[:, 1] -= ymin
    cv2.fillPoly(mask, b.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray:
    """同 DBPostProcess.unclip（shapely 的面積、周長改用鞋帶公式與邊長和）。"""
    import pyclipper
    pts = [(float(x), float(y)) for x, y in box.tolist()]
    n = len(pts)
    area = abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))) / 2
    length = sum(((pts[(i + 1) % n][0] - pts[i][0]) ** 2 + (pts[(i + 1) % n][1] - pts[i][1]) ** 2) ** 0.5 for i in range(n))
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return np.array(offset.Execute(area * ratio / length)).reshape((-1, 1, 2))


def db_boxes(pp, preds: np.ndarray, ori_shape: tuple[int, int], box_thresh: float):
    """同 DBPostProcess.__call__（score_mode="fast"），門檻用 box_thresh：回傳 (四角 [N, 4, 2] int32, 分數)。"""
    src_h, src_w = ori_shape
    pred = preds[0, 0]
    seg = pred > pp.thresh
    mask = seg if pp.dilation_kernel is None else cv2.dilate(seg.astype(np.uint8), pp.dilation_kernel)
    height, width = mask.shape
    contours = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    boxes, scores = [], []
    for contour in contours[:pp.max_candidates]:
        points, sside = pp.get_mini_boxes(contour)
        if sside < pp.min_size:
            continue
        score = _score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        box, sside = pp.get_mini_boxes(_unclip(points, pp.unclip_ratio))
        if sside < pp.min_size + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] / width * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * src_h), 0, src_h)
        boxes.append(box.astype(np.int32))
        scores.append(score)
    return np.array(boxes, dtype=np.int32), scores
