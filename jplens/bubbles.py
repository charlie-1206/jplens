"""漫畫模式：找出畫面上漫畫的文字區塊（對話框裡的字、對話框外的旁白與效果字）。

模型：ogkalu/comic-text-and-bubble-detector（RT-DETR-v2，Apache-2.0，Hugging Face）的 ONNX 版，放在
`data/models/ogkalu/`（不進版控）。有 NVIDIA 顯示卡用 `detector.onnx`（fp32，168 MB：GPU 上整個螢幕 0.08 秒、
記憶體 +164 MB）；沒有就用 `detector_int8.onnx`（44 MB，CPU 0.37 秒；int8 在 GPU 上反而慢，0.32 秒、+302 MB）。
兩個版本在測試裡準確度相同。輸入整個畫面縮成 640×640，輸出原圖座標的框，
類別 0＝對話框外框、1＝對話框裡的字、2＝對話框外的字；只用 1、2。

為什麼不用通用的文字偵測（PP-OCR）：它不懂漫畫，字小、欄間夾振假名時會漏欄、把兩欄黏成一欄，依行讀就漏字。
這個模型直接給出一整塊字，整塊交給 manga-ocr（本來就是為一整個對話框設計的）。
比較過 comic-text-detector（GPL-3.0）：漫畫題目準確度打平，但它會把相鄰的對話框黏在一起、效果字抓得少，
授權也比較麻煩（GPL-3.0）。只適合漫畫：動畫字幕、小說內文、遊戲對話框抓不到或切錯，所以只在漫畫模式用。
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import APP_DIR

MODEL_DIR = APP_DIR / "data" / "models" / "ogkalu"
MODEL_GPU, MODEL_CPU = MODEL_DIR / "detector.onnx", MODEL_DIR / "detector_int8.onnx"
MODEL_URL = "https://huggingface.co/ogkalu/comic-text-and-bubble-detector/tree/main"
SIZE = 640
SCORE = 0.3     # 分數門檻
NMS_IOU = 0.5   # 模型偶爾對同一塊字給兩個框
INSIDE = 0.6    # 文字區塊至少這麼多面積在使用者的框裡，才算框到它
PAD = 0.05      # 讀之前四邊各外擴短邊的 5%＋4 像素（框太貼時邊緣的字會被切到）
TEXT_CLASSES = {1, 2}
log = logging.getLogger("jplens")


@dataclass(frozen=True)
class Bubble:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float

    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def overlap(self, box) -> int:
        x0, y0, x1, y1 = box
        return max(0, min(x1, self.x1) - max(x0, self.x0)) * max(0, min(y1, self.y1) - max(y0, self.y0))

    def iou(self, o: "Bubble") -> float:
        inter = self.overlap((o.x0, o.y0, o.x1, o.y1))
        union = self.area + o.area - inter
        return inter / union if union else 0.0

    def crop(self, w: int, h: int) -> tuple[int, int, int, int]:
        """外擴 PAD 後的讀取範圍（不超出圖）。"""
        m = int(PAD * min(self.x1 - self.x0, self.y1 - self.y0)) + 4
        return max(0, self.x0 - m), max(0, self.y0 - m), min(w, self.x1 + m), min(h, self.y1 + m)


class BubbleDetector:
    def __init__(self, path=None):
        import onnxruntime as ort

        from . import gpu as _gpu
        gpu = _gpu.cuda_ready()  # 有 NVIDIA 顯示卡、而且找得到 CUDA 程式庫（不然 ONNX Runtime 會悄悄退回 CPU）
        if path is None:
            prefer = [MODEL_GPU, MODEL_CPU] if gpu else [MODEL_CPU, MODEL_GPU]
            path = next((p for p in prefer if p.exists()), prefer[0])
        if not path.exists():
            raise FileNotFoundError(f"找不到漫畫文字偵測模型 {path.name}，請從 {MODEL_URL} 下載到 {MODEL_DIR}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # 不和其他 session 搶 CPU（見 ocr._make_rapidocr）
        providers = _gpu.providers()
        self.path = path
        self.sess = ort.InferenceSession(str(path), so, providers=providers)
        self.device = "GPU" if self.sess.get_providers()[0] == "CUDAExecutionProvider" else "CPU"

    def detect(self, img: np.ndarray) -> list[Bubble]:
        h, w = img.shape[:2]
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = x.transpose(2, 0, 1)[None].astype(np.float32) / 255
        labels, boxes, scores = self.sess.run(None, {"images": x, "orig_target_sizes": np.array([[w, h]], np.int64)})
        cands = []
        for lb, (x0, y0, x1, y1), s in zip(labels[0], boxes[0], scores[0]):
            if s >= SCORE and int(lb) in TEXT_CLASSES:
                b = Bubble(max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)), float(s))
                if b.x1 - b.x0 >= 4 and b.y1 - b.y0 >= 4:
                    cands.append(b)
        out = []
        for b in sorted(cands, key=lambda b: -b.score):
            if all(b.iou(o) <= NMS_IOU for o in out):
                out.append(b)
        return out


def reading_order(blocks: list[Bubble]) -> list[Bubble]:
    """漫畫的閱讀順序：由右而左，同一欄由上而下。"""
    return sorted(blocks, key=lambda b: (-b.x1, b.y0))


def pick(blocks: list[Bubble], box) -> list[Bubble]:
    """框到的文字區塊（至少 INSIDE 的面積在框裡），依閱讀順序。框只碰到區塊的一小部分（只想查幾個字）時不算。"""
    return reading_order([b for b in blocks if b.overlap(box) >= INSIDE * b.area])


def at(blocks: list[Bubble], x, y, near: int = 12) -> list[Bubble]:
    """游標所在的文字區塊（點在對話框的空白處也算：允許離字 near 像素）；有好幾個重疊時取最小的。"""
    hits = [b for b in blocks if b.x0 - near <= x < b.x1 + near and b.y0 - near <= y < b.y1 + near]
    return [min(hits, key=lambda b: b.area)] if hits else []
