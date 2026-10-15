"""OCR 引擎：RapidOCR (DBNet) 偵測文字框 + manga-ocr 辨識日文。

manga-ocr 會把輸入縮放成 224x224，過長的單行會被壓扁而掉字，
所以長行會先在字距空隙處切段再分別辨識。
manga-ocr 以漫畫（白底黑字）訓練，深色底的字幕先反白再辨識，
實測可大幅減少「っ→つ」等小字誤認。
"""

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .textutil import clean, fix_common_ocr, join_lines

MANGA_OCR_REPO = "kha-white/manga-ocr-base"
# 官方 repo 的 main 只有 pytorch_model.bin（新版 transformers 要求 torch>=2.6 才能載入），
# refs/pr/4 是同一份權重的 safetensors 版本。
MANGA_OCR_REVISION = "refs/pr/4"
CHUNK_RATIO = 6.0   # 單段最長邊 / 短邊 的上限（實測 3~7 中以 6 最準：切太短會失去上下文）
DET_MAX_SIDE = 1280  # 偵測前縮圖的最長邊（加速，對字幕大小的文字足夠）


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def vertical(self):
        return self.h > self.w * 1.5

    @property
    def thickness(self):
        """文字行的「字高」：橫排取高度、直排取寬度。"""
        return self.w if self.vertical else self.h

    def iou(self, o: "Box") -> float:
        ix = max(0, min(self.x1, o.x1) - max(self.x0, o.x0))
        iy = max(0, min(self.y1, o.y1) - max(self.y0, o.y0))
        inter = ix * iy
        union = self.w * self.h + o.w * o.h - inter
        return inter / union if union > 0 else 0.0

    def union(self, o: "Box") -> "Box":
        return Box(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1))

    def shift(self, dx, dy) -> "Box":
        return Box(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def to_list(self):
        return [self.x0, self.y0, self.x1, self.y1]


def _load_manga_ocr():
    from huggingface_hub import snapshot_download
    from manga_ocr import MangaOcr

    patterns = ["*.json", "*.txt", "*.safetensors"]
    try:
        path = snapshot_download(MANGA_OCR_REPO, revision=MANGA_OCR_REVISION,
                                 allow_patterns=patterns, local_files_only=True)
    except Exception:
        path = snapshot_download(MANGA_OCR_REPO, revision=MANGA_OCR_REVISION, allow_patterns=patterns)
    return MangaOcr(path)


def _make_rapidocr():
    """建立 RapidOCR，並關閉 ONNX Runtime 的執行緒空轉（spinning）。

    ONNX Runtime 預設每次推論完會讓執行緒空轉等待下一次呼叫，和 PyTorch（manga-ocr）
    以及另一個 ONNX session 互搶 CPU；實測關掉後整體掃描快約 4 倍。
    """
    from rapidocr_onnxruntime import RapidOCR
    try:
        from rapidocr_onnxruntime.utils import infer_engine
        orig = infer_engine.OrtInferSession._init_sess_opts
    except (ImportError, AttributeError):
        return RapidOCR()

    def no_spin(config):
        opts = orig(config)
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return opts

    infer_engine.OrtInferSession._init_sess_opts = staticmethod(no_spin)
    try:
        return RapidOCR()
    finally:
        infer_engine.OrtInferSession._init_sess_opts = staticmethod(orig)


class OcrEngine:
    def __init__(self):
        logging.getLogger("transformers").setLevel(logging.ERROR)
        try:
            from loguru import logger
            logger.remove()  # manga-ocr 的 INFO 訊息太吵
        except ImportError:
            pass
        rapid = _make_rapidocr()
        self._det = rapid.text_det
        self._det.limit_type = "max"
        self._latin = rapid.text_rec  # 中英辨識模型，只用來判斷「這是不是英文」
        self._rec = _load_manga_ocr()

    def detect(self, img: np.ndarray) -> list[Box]:
        h, w = img.shape[:2]
        scale = min(1.0, DET_MAX_SIDE / max(h, w))
        small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
        quads, _ = self._det(small)
        boxes = []
        for q in (quads if quads is not None else []):
            q = np.asarray(q) / scale
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                boxes.append(Box(int(x0), int(y0), int(x1), int(y1)))
        return boxes

    def read(self, img: np.ndarray, box: Box) -> str:
        pad = max(2, box.thickness // 6)
        H, W = img.shape[:2]
        crop = img[max(0, box.y0 - pad):min(H, box.y1 + pad), max(0, box.x0 - pad):min(W, box.x1 + pad)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if np.median(gray) < 128:  # 深色底淺色字 → 反白成白底黑字
            gray = 255 - gray
        parts = [self._rec(Image.fromarray(chunk).convert("RGB")) for chunk in _split_long(gray, box.vertical)]
        text = clean(join_lines(parts))
        return "" if self._hallucinated_latin(crop, text) else text

    def _hallucinated_latin(self, crop: np.ndarray, text: str) -> bool:
        """manga-ocr 只認得日文，遇到英文招牌會硬讀成假名或數字（Random Play → ツドの）。
        中英模型很有把握是英文、而 manga-ocr 讀出的英文字母遠少於它 → 判定為幻覺。
        日英混排（撚り合う (Promise you)）時 manga-ocr 也會讀出英文，所以不受影響。"""
        (latin, score), = self._latin([crop])[0]
        letters = re.findall(r"[A-Za-z0-9]", latin)
        body = re.sub(r"[\W_]", "", latin)
        if score < 0.7 or len(letters) < 4 or len(letters) < 0.8 * len(body):
            return False
        return len(re.findall(r"[A-Za-z]", text)) < 0.5 * len(re.findall(r"[A-Za-z]", latin))


def _split_long(gray: np.ndarray, vertical: bool) -> list[np.ndarray]:
    """沿長軸把過長的文字行（灰階）切成數段，切點選在附近墨水最少（字與字之間）的位置。"""
    if vertical:
        return [c.T for c in _split_long(gray.T, False)]
    h, w = gray.shape
    if w <= h * CHUNK_RATIO:
        return [gray]
    g = gray.astype(np.float32)
    ink = np.abs(g - np.median(g)).sum(0)
    n = int(np.ceil(w / (h * CHUNK_RATIO)))
    step = w / n
    cuts = [0]
    for k in range(1, n):
        c = int(k * step)
        lo = max(cuts[-1] + h, c - int(h * 0.8))
        hi = min(w - h, c + int(h * 0.8))
        if hi > lo:
            cuts.append(lo + int(np.argmin(ink[lo:hi])))
    cuts.append(w)
    return [np.ascontiguousarray(gray[:, a:b]) for a, b in zip(cuts, cuts[1:]) if b - a > 2]


@dataclass
class Line:
    box: Box
    text: str


def _continues(a: Box, b: Box) -> bool:
    """b 是否緊接在 a 後面、屬於同一行（偵測器常在標點或空白處把一行切成兩段）。"""
    def squarish(x):
        return not x.vertical and x.w <= x.h * 1.5

    if a.vertical or b.vertical:
        if not ((a.vertical or squarish(a)) and (b.vertical or squarish(b))):
            return False
        th = min(a.w, b.w)
        overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
        gap = b.y0 - a.y1
    else:
        th = min(a.h, b.h)
        overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
        gap = b.x0 - a.x1
    return overlap >= 0.6 * th and -0.3 * th <= gap <= 1.5 * th


def _merge_fragments(lines: list[Line]) -> list[Line]:
    lines = list(lines)
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(lines):
            j = next((j for j, b in enumerate(lines) if j != i and _continues(a.box, b.box)), None)
            if j is not None:
                b = lines[j]
                lines[i] = Line(a.box.union(b.box), join_lines([a.text, b.text]))
                del lines[j]
                merged = True
                break
    return lines


def group_blocks(lines: list[Line]) -> list[tuple[Box, str]]:
    """把相鄰的文字行合併成區塊（例如同一個對話框裡換行的句子），並依閱讀順序排列。

    先把被切斷的同一行接回來，再合併多行：
    橫排：上下間距 < 0.6 個字高、且水平範圍重疊 → 同一區塊，由上而下。
    直排：左右間距 < 0.6 個字寬、且垂直範圍重疊 → 同一區塊，由右而左。
    """
    lines = _merge_fragments(lines)
    blocks: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (-l.box.x1, l.box.y0) if l.box.vertical else (l.box.y0, l.box.x0)):
        for blk in blocks:
            last = blk[-1]
            if last.box.vertical != ln.box.vertical:
                continue
            th = min(last.box.thickness, ln.box.thickness)
            if ln.box.vertical:
                gap = last.box.x0 - ln.box.x1
                overlap = min(last.box.y1, ln.box.y1) - max(last.box.y0, ln.box.y0)
            else:
                gap = ln.box.y0 - last.box.y1
                overlap = min(last.box.x1, ln.box.x1) - max(last.box.x0, ln.box.x0)
            if -0.3 * th <= gap <= 0.6 * th and overlap > 0:
                blk.append(ln)
                break
        else:
            blocks.append([ln])

    out = []
    for blk in blocks:
        box = blk[0].box
        for ln in blk[1:]:
            box = box.union(ln.box)
        # 括號可能跨行（「在第一行、」在第二行），所以規則修正在整個區塊合併後才做
        out.append((box, fix_common_ocr(join_lines([ln.text for ln in blk]))))
    out.sort(key=lambda b: (b[0].y0, b[0].x0))
    return out
