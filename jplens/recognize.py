"""框選區域 → 日文文字區塊。

沿用 video_translate 的 OcrEngine（RapidOCR 偵測文字框＋manga-ocr 辨識），
但辨識改成**批次**：一次框選裡所有的文字段落一起送進 GPU（逐段呼叫每段約 70 ms，
一頁直排小說有 17 段就要 1 秒多）。

辨識策略（M0 用 tests/eval/m0_ocr_eval.py 比較）：
- lines ：逐行辨識後合併成區塊（video_translate 的做法）
- whole ：整個框選區域直接丟給 manga-ocr（只適合單一個漫畫對話框，多行時準確率很差）
- blocks：先把文字框分組成區塊；直排的小區塊（2～4 欄）整塊丟給 manga-ocr，其餘逐行（預設）
"""

import logging
import os
import re
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from . import gpu, mangaocr_onnx
from .jlearn import ocr, textutil

Box = ocr.Box
BATCH = 16
log = logging.getLogger("jplens")


@dataclass
class Block:
    box: Box
    text: str
    vertical: bool


def _gray_for_rec(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return 255 - gray if np.median(gray) < 128 else gray  # 深色底反白（manga-ocr 以白底黑字訓練）


def _clear_margins(crop: np.ndarray, a: int, b: int, vertical: bool) -> np.ndarray:
    """讀一行時兩側多切的邊（[0, a) 與 [b, 結尾)）：和這一行之間隔著一條空白的墨水是別行的
    （隔壁欄的振假名切到一半，會被讀成多出來的字：正文後面多了幾個假名），塗成背景色。"""
    if crop.size == 0:
        return crop
    g = crop.mean(axis=2)
    bg = np.median(g)
    ink = (np.abs(g - bg) > 40).sum(axis=0 if vertical else 1)
    out = None
    inset = int(0.2 * (b - a))  # 分界的空白也可能在框內側一點（偵測框比字寬）
    left = [i for i in range(min(a + inset, len(ink))) if ink[i] == 0]
    right = [i for i in range(max(0, b - inset), len(ink)) if ink[i] == 0]
    if left and left[-1] > 0:
        out = crop.copy()
        if vertical:
            out[:, :left[-1]] = np.median(crop.reshape(-1, crop.shape[2]), axis=0)
        else:
            out[:left[-1], :] = np.median(crop.reshape(-1, crop.shape[2]), axis=0)
    if right and right[0] < len(ink) - 1:
        out = crop.copy() if out is None else out
        if vertical:
            out[:, right[0] + 1:] = np.median(crop.reshape(-1, crop.shape[2]), axis=0)
        else:
            out[right[0] + 1:, :] = np.median(crop.reshape(-1, crop.shape[2]), axis=0)
    return crop if out is None else out


def _crop(img: np.ndarray, box: Box, pad: int) -> np.ndarray:
    H, W = img.shape[:2]
    return img[max(0, box.y0 - pad):min(H, box.y1 + pad), max(0, box.x0 - pad):min(W, box.x1 + pad)]


def _merge_boxes(boxes: list[Box]) -> list[Box]:
    """把偵測器在標點或空白處切斷的同一行接回來。
    接好一段就繼續看同一個框還能不能再接，接不下去才換下一個框；整輪都沒有再接就結束
    （以前每接一次就從頭重新找：漫畫整頁 119 個框要比 10 萬次、0.8 秒，現在 0.05 秒以內）。"""
    boxes = list(boxes)
    # 座標另外存成 numpy 陣列，距離篩選一次算整排（結果和逐一比相同；字多的畫面 300 個框約 30 → 幾 ms）
    c = np.array([(b.x0, b.y0, b.x1, b.y1) for b in boxes], np.float64).reshape(-1, 4)
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(boxes):
            a = boxes[i]
            # 先用距離篩掉不可能接上的框（_continues 要求兩框相距不到 1.5 個字寬），省掉大部分比對
            r = 1.6 * (max(a.w, a.h) if a.w <= 1.5 * a.h and a.h <= 1.5 * a.w else min(a.w, a.h))
            near = np.flatnonzero((c[:, 0] < a.x1 + r) & (c[:, 2] > a.x0 - r) & (c[:, 1] < a.y1 + r) & (c[:, 3] > a.y0 - r))
            j = next((int(j) for j in near if j != i and ocr._continues(a, boxes[j])), None)
            if j is None:
                i += 1
                continue
            boxes[i] = u = a.union(boxes[j])
            c[i] = (u.x0, u.y0, u.x1, u.y1)
            del boxes[j]
            c = np.delete(c, j, axis=0)
            if j < i:
                i -= 1
            merged = True
    return boxes


def _union(boxes):
    u = boxes[0]
    for b in boxes[1:]:
        u = u.union(b)
    return u


def group_boxes(boxes: list[Box]) -> list[list[Box]]:
    """把相鄰的文字行分成區塊（同 ocr.group_blocks，但在辨識前就分組）。
    區塊內：橫排由上而下、直排由右而左；區塊之間由上而下（同高度時直排右邊先、橫排左邊先）。"""
    lines = _merge_boxes(boxes)
    groups: list[list[Box]] = []
    for b in sorted(lines, key=lambda b: (-b.x1, b.y0) if b.vertical else (b.y0, b.x0)):
        for g in groups:
            last = g[-1]
            if last.vertical != b.vertical:
                continue
            th = min(last.thickness, b.thickness)
            if b.vertical:
                gap = last.x0 - b.x1
                overlap = min(last.y1, b.y1) - max(last.y0, b.y0)
            else:
                gap = b.y0 - last.y1
                overlap = min(last.x1, b.x1) - max(last.x0, b.x0)
            if -0.3 * th <= gap <= 0.8 * th and overlap > 0:
                g.append(b)
                break
        else:
            groups.append([b])

    def order(g):
        u = _union(g)
        return (u.y0, -u.x1) if g[0].vertical else (u.y0, u.x0)
    return sorted(groups, key=order)


OVERLAP = 2.2  # 長行切段時，相鄰兩段重疊幾個字寬（字高）


def split_overlap(gray: np.ndarray, vertical: bool) -> list[np.ndarray]:
    """把長行切成長寬比不超過 ocr.CHUNK_RATIO 的幾段，相鄰兩段重疊 OVERLAP 個字寬。

    原本的 ocr._split_long 把切點選在墨水最少處，但字幕疊在畫面上時背景很雜，
    切點常落在字的中間（部、出 被切成兩半），兩段各讀到半個字就會重複（出出す）或讀錯。
    重疊 2.2 個字寬保證重疊區裡至少有一個字在兩段都完整，merge_chunks 用它對齊接起來。"""
    if vertical:
        return [c.T for c in split_overlap(gray.T, False)]
    return [np.ascontiguousarray(gray[:, a:b]) for a, b in overlap_spans(*gray.shape)]


def overlap_spans(h: int, w: int) -> list[tuple[int, int]]:
    """split_overlap 的切法：橫排（高 h、長 w）的行切成的每一段 [a, b)。"""
    max_len = ocr.CHUNK_RATIO * h
    if w <= max_len:
        return [(0, w)]
    ov = OVERLAP * h
    n = int(np.ceil((w - ov) / (max_len - ov)))
    step = (w - ov) / n
    return [(int(i * step), min(w, int(i * step + step + ov))) for i in range(n)]


BRIDGE = 4.5  # 接縫重讀的長度（字寬的幾倍），以兩段的重疊區為中心
BRIDGE_GAP = 4  # 和橋對齊時兩邊各容許幾個不吻合的字


def merge_pair(out: str, t: str, gap: int = 2) -> str | None:
    """找 out 結尾和 t 開頭相同的部分只留一份；找不到回傳 None。
    段落邊緣只拍到半個字時會讀成別的字（今 的右半 → 〈ま），所以對齊時容許兩邊各有最多 gap 個不吻合的字。"""
    best = None  # (重疊字數, -丟掉的字數, 接好的結果)
    for k in range(min(6, len(out), len(t)), 0, -1):
        for ga in range(gap + 1):
            for gb in range(gap + 1):
                shared = t[gb:gb + k]
                if len(shared) == k and len(out) - ga >= k and out[:len(out) - ga].endswith(shared):
                    cand = (k, -(ga + gb), out[:len(out) - ga] + t[gb + k:])
                    if best is None or cand[:2] > best[:2]:
                        best = cand
        if best is not None:
            break  # 重疊字數多的優先
    return best[2] if best else None


def merge_chunks(texts: list[str], bridges: dict[int, str] | None = None) -> str:
    """把重疊切段的辨識結果接起來。bridges：第 i 段和第 i+1 段的接縫重讀的結果（接不起來的接縫才有）。
    兩段把重疊區的字讀成不同的字（遺憾／遠慮）就找不到相同的部分，原本只能直接相接（遺憾をに遠慮），
    這時改用以接縫為中心重讀的那張當橋：前一段 → 橋 → 後一段。"""
    out = ""
    for i, t in enumerate(texts):
        if not out:
            out = t
            continue
        m = merge_pair(out, t)
        if m is None and bridges and bridges.get(i - 1):
            # 橋和兩段不同的字（遺憾／遠慮）正好在重疊區，可能比一般接縫多，容許到 4 個字
            mb = merge_pair(out, bridges[i - 1], BRIDGE_GAP)
            m = merge_pair(mb, t, BRIDGE_GAP) if mb is not None else None
        out = m if m is not None else textutil.join_lines([out, t])
    return out


def _hallucinated_latin(latin: tuple[str, float], text: str) -> bool:
    """同 OcrEngine._hallucinated_latin，但中英模型的結果是批次算好傳進來的。"""
    latin, score = latin
    letters = re.findall(r"[A-Za-z0-9]", latin)
    body = re.sub(r"[\W_]", "", latin)
    if score < 0.7 or len(letters) < 4 or len(letters) < 0.8 * len(body):
        return False
    return len(re.findall(r"[A-Za-z]", text)) < 0.5 * len(re.findall(r"[A-Za-z]", latin))


_UNUSED_PACKAGES = ("sklearn", "scipy", "pandas", "av")


def _load_manga_ocr_onnx():
    """manga-ocr 的 ONNX 版（jplens/mangaocr_onnx.py）：不載入 PyTorch，常駐記憶體約少一半。"""
    return mangaocr_onnx.MangaOcrOnnx(gpu.providers())


def use_onnx() -> bool:
    """預設用 ONNX 版；模型還沒匯出（tools/export_manga_ocr_onnx.py）或設了環境變數 JPLENS_OCR=torch 時用 PyTorch 版。"""
    return os.environ.get("JPLENS_OCR", "onnx") != "torch" and mangaocr_onnx.available()


def _load_manga_ocr():
    """PyTorch 版（舊做法；ONNX 模型沒匯出時才用）。同 video_translate 的 ocr._load_manga_ocr
    （同一個 repo 與 revision：safetensors 版），但有顯示卡時權重直接以 fp16 載到 GPU。

    MangaOcr() 會先以 fp32 把整個模型載進 RAM 再搬到 GPU；反正之後也是用 fp16 在 GPU 上推論
    （Recognizer 原本會再 .half()），結果完全相同。也略過 MangaOcr() 結尾辨識範例圖那一次（Recognizer.warmup 會暖機）。

    transformers 偵測到有安裝就會順便 import 的套件（sklearn → pandas、scipy、torchvision → av），
    manga-ocr 都用不到：標成「沒安裝」，認可記憶體少約 500 MB、載入快約 1 秒。
    torchvision 保留：transformers 5 有它時影像前處理改用 torchvision 縮放，換回 PIL 版輸入會差一個灰階。"""
    for name in _UNUSED_PACKAGES:
        sys.modules.setdefault(name, None)  # None＝import 時直接 ImportError，find_spec 也當成沒安裝
    import torch
    from huggingface_hub import snapshot_download
    from manga_ocr.ocr import MangaOcr, MangaOcrModel
    from transformers import AutoTokenizer, ViTImageProcessor

    patterns = ["*.json", "*.txt", "*.safetensors"]
    try:
        path = snapshot_download(ocr.MANGA_OCR_REPO, revision=ocr.MANGA_OCR_REVISION,
                                 allow_patterns=patterns, local_files_only=True)
    except Exception:
        path = snapshot_download(ocr.MANGA_OCR_REPO, revision=ocr.MANGA_OCR_REVISION, allow_patterns=patterns)
    mo = MangaOcr.__new__(MangaOcr)
    mo.processor = ViTImageProcessor.from_pretrained(path)
    mo.tokenizer = AutoTokenizer.from_pretrained(path, tokenizer_type="bert-japanese")
    if torch.cuda.is_available():
        mo.model = MangaOcrModel.from_pretrained(path, dtype=torch.float16, device_map="cuda")
    else:
        mo.model = MangaOcrModel.from_pretrained(path)
    mo.model.eval()
    return mo


def _make_engine() -> ocr.OcrEngine:
    """建立 OcrEngine，並開啟 RapidOCR（ONNX Runtime）的 CPU 記憶體池；manga-ocr 改用本模組的 _load_manga_ocr。

    RapidOCR 預設關閉記憶體池，每遇到新的輸入尺寸就要重新配置記憶體：偵測從 35 ms 變 90 ms、
    中英辨識從 40 ms 變 230 ms。影片每幀尺寸相同所以影響不大，但每次框選的大小都不一樣。"""
    from rapidocr_onnxruntime.utils import infer_engine
    orig = infer_engine.OrtInferSession._init_sess_opts
    orig_load = ocr._load_manga_ocr

    def with_arena(config):
        opts = orig(config)
        opts.enable_cpu_mem_arena = True
        return opts

    infer_engine.OrtInferSession._init_sess_opts = staticmethod(with_arena)
    ocr._load_manga_ocr = _load_manga_ocr_onnx if use_onnx() else _load_manga_ocr
    try:
        return ocr.OcrEngine()  # 裡面會再包一層「關閉執行緒空轉」
    finally:
        infer_engine.OrtInferSession._init_sess_opts = staticmethod(orig)
        ocr._load_manga_ocr = orig_load


class Recognizer:
    def __init__(self):
        self.engine = _make_engine()
        self._mo = self.engine._rec  # MangaOcrOnnx（預設）或 MangaOcr（PyTorch 版）
        self.onnx = isinstance(self._mo, mangaocr_onnx.MangaOcrOnnx)
        # PyTorch 版有顯示卡時 _load_manga_ocr 已直接載成半精度：video_translate 真實影片 40 幀、267 個區塊的結果
        # 和 fp32 完全相同，快約 30%
        self.det_device = self._det_to_gpu()
        self.fastdet = self._load_fastdet() if self.det_device == "GPU" else None
        if self.fastdet is not None:
            self.det_device += "（" + ("fp16" if "fp16" in self.fastdet.path.name else "fp32") + "）"
        self.rec_device = ("ONNX " + ("GPU" if self._mo.device != "cpu" else "CPU")) if self.onnx else \
            ("PyTorch " + ("GPU" if str(self._mo.model.device).startswith("cuda") else "CPU"))

    def _load_fastdet(self):
        """整個螢幕的偵測改用吃 uint8 的模型（detect.py；tools/export_det_onnx.py 產生）：前處理在顯示卡上做、fp16。
        和原本的模型核對一次（二值化後不同的像素要很少）才用；沒有模型或核對不過就照舊。"""
        from . import detect
        path = detect.model_path()
        if path is None:
            return None
        try:
            fd = detect.FastDet(path, [("CUDAExecutionProvider", gpu.CUDA_OPTS), "CPUExecutionProvider"])
            img = np.full((320, 640, 3), 245, np.uint8)
            cv2.putText(img, "check 0123", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 4)
            det = self.engine._det
            a = det.infer(cv2.dnn.blobFromImage(img, 1 / 127.5, None, (127.5, 127.5, 127.5)))[0][0, 0]
            b = fd.prob(img)
            if ((a > 0.3) != (b > 0.3)).mean() > 0.002:
                log.warning("快速文字偵測（%s）和原本的結果差太多，照舊", path.name)
                return None
            return fd
        except Exception:
            log.exception("快速文字偵測（%s）無法使用，照舊", path.name)
            return None

    def _det_to_gpu(self) -> str:
        """文字偵測（RapidOCR）預設跑 CPU；有 NVIDIA 顯示卡、裝了 onnxruntime-gpu 就改用 GPU：
        2560×1440 整個螢幕 0.65 → 0.11 秒，輸出幾乎相同（二值化後只有 0.001% 的像素不同）。
        沒有顯示卡、沒裝 GPU 版或 GPU 出錯，就維持 CPU（和以前完全一樣）。回傳實際使用的裝置。"""
        try:
            import onnxruntime as ort
            if not gpu.cuda_ready():
                return "CPU"
            infer = self.engine._det.infer
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # HEURISTIC：預設的 EXHAUSTIVE 每遇到新的輸入尺寸要先試遍演算法（框選備案的裁切圖每次尺寸都不同）
            cuda = ("CUDAExecutionProvider", gpu.CUDA_OPTS)
            sess = ort.InferenceSession(infer.session._model_path, so, providers=[cuda, "CPUExecutionProvider"])
            if "CUDAExecutionProvider" not in sess.get_providers():
                return "CPU"
            # 和 CPU 的結果核對一次再換：顯示卡驅動、CUDA 版本不合時寧可用 CPU
            x = np.random.default_rng(0).random((1, 3, 320, 320), dtype=np.float32)
            name = sess.get_inputs()[0].name
            a = infer.session.run(None, {name: x})[0]
            b = sess.run(None, {name: x})[0]
            if np.abs(a - b).max() > 0.05:
                log.warning("GPU 文字偵測和 CPU 結果差太多（%.3f），改用 CPU", np.abs(a - b).max())
                return "CPU"
            infer.session = sess
            return "GPU"
        except Exception:
            log.exception("文字偵測無法使用 GPU，改用 CPU")
            return "CPU"

    def warmup(self):
        img = np.full((64, 256, 3), 255, np.uint8)
        cv2.putText(img, "warm", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self.recognize(img)

    def read_whole(self, img: np.ndarray, crop=None, blank=()) -> str:
        """整塊交給 manga-ocr（一個漫畫對話框）。crop：(x0, y0, x1, y1)；blank：要先塗成背景色的範圍
        （框邊切到一半的字，整張圖的座標）。"""
        if crop is not None:
            x0, y0, x1, y1 = crop
            img = img[y0:y1, x0:x1].copy()
            bg = np.median(img.reshape(-1, img.shape[2]), axis=0)
            for bx0, by0, bx1, by1 in blank:
                img[max(0, by0 - y0):max(0, by1 - y0), max(0, bx0 - x0):max(0, bx1 - x0)] = bg
        return self.rec_batch([_gray_for_rec(img)])[0]

    def read_regions(self, img: np.ndarray, crops) -> list[str]:
        """漫畫模式：好幾個文字區塊各自整塊交給 manga-ocr（一起批次辨識）。crops：[(x0, y0, x1, y1)]。"""
        return self.rec_batch([_gray_for_rec(img[y0:y1, x0:x1]) for x0, y0, x1, y1 in crops])

    def rec_batch(self, grays: list[np.ndarray]) -> list[str]:
        """manga-ocr 批次辨識（灰階圖）。結果和逐張呼叫相同（輸入都是 224×224，每張圖各自 beam search）。
        解碼設定來自模型的 config.json：beam search 4 束、不重複 3-gram、length_penalty 2.0（不是 greedy）。"""
        if self.onnx:
            return [t for i in range(0, len(grays), BATCH) for t in self._mo(grays[i:i + BATCH])]
        import torch
        from manga_ocr.ocr import post_process
        mo = self._mo
        out = []
        with torch.inference_mode():
            for i in range(0, len(grays), BATCH):
                x = torch.stack([mo._preprocess(Image.fromarray(g).convert("RGB")) for g in grays[i:i + BATCH]])
                ids = mo.model.generate(x.to(mo.model.device, mo.model.dtype), max_length=300).cpu()
                out += [post_process(mo.tokenizer.decode(t, skip_special_tokens=True)) for t in ids]
        return out

    def read_lines(self, img: np.ndarray, boxes: list[Box]) -> list[str]:
        """同 OcrEngine.read（長行切段、深色底反白、英文幻覺過濾），但所有行一起批次辨識。"""
        crops = [_crop(img, b, max(2, b.thickness // 6)) for b in boxes]
        texts = [textutil.clean(t) for t in self.read_long([_gray_for_rec(c) for c in crops],
                                                           [b.vertical for b in boxes])]
        latin = self.engine._latin(crops)[0] if crops else []  # 回傳 ([(文字, 分數), ...], 耗時)
        return ["" if _hallucinated_latin(la, t) else t for la, t in zip(latin, texts)]

    def read_long(self, grays: list[np.ndarray], verticals: list[bool]) -> list[str]:
        """每一行（灰階圖）重疊切段後一起批次辨識、接起來。接不起來的接縫（兩段把重疊區讀成不同的字）
        再批次讀一次以接縫為中心的圖，當成接兩段的橋（merge_chunks）。"""
        chunks, owner, spans = [], [], []
        for k, (g, v) in enumerate(zip(grays, verticals)):
            g = g.T if v else g  # 直排轉成橫排來切，送進模型前再轉回來
            for a, b in overlap_spans(*g.shape):
                c = np.ascontiguousarray(g[:, a:b])
                chunks.append(c.T if v else c)
                owner.append(k)
                spans.append((a, b))
        texts = self.rec_batch(chunks) if chunks else []
        parts = [[] for _ in grays]
        idx = [[] for _ in grays]  # 每一行的各段在 chunks 裡的位置
        for i, (k, t) in enumerate(zip(owner, texts)):
            parts[k].append(t)
            idx[k].append(i)
        bridges, where = [], []
        for k, (g, v) in enumerate(zip(grays, verticals)):
            g = g.T if v else g
            h, w = g.shape
            for j in range(len(idx[k]) - 1):
                p, q = idx[k][j], idx[k][j + 1]
                if not texts[p] or not texts[q] or merge_pair(texts[p], texts[q]) is not None:
                    continue
                mid = (spans[q][0] + spans[p][1]) / 2  # 重疊區的中間
                a, b = max(0, int(mid - BRIDGE * h / 2)), min(w, int(mid + BRIDGE * h / 2))
                c = np.ascontiguousarray(g[:, a:b])
                bridges.append(c.T if v else c)
                where.append((k, j))
        found = [{} for _ in grays]
        if bridges:
            for (k, j), t in zip(where, self.rec_batch(bridges)):
                found[k][j] = t
            log.info("接縫重讀 %d 處", len(bridges))
        return [merge_chunks(p, f) if p else "" for p, f in zip(parts, found)]

    def detect_screen(self, img: np.ndarray) -> list[Box]:
        """反白模式：偵測整個螢幕的文字行（不先縮圖；RapidOCR 內部會把長邊限制在 2000）。

        同一次推論用兩個門檻（box_thresh）各取一次框：
        - 0.5（預設）：框準，但小說整頁會漏掉幾欄、漫畫對話框也會漏
        - 0.3：漏掉的欄抓得回來，但偶爾把相鄰幾欄、甚至好幾個對話框框成一個大框
        以 0.5 為主，0.3 只補「和 0.5 的框幾乎不重疊」的框：重疊不到一半的也要
        （偵測框會往外擴，小字的小說漏掉的那一欄常和隔壁欄的框重疊一點；重疊處之後由 layout 在中間分開）。"""
        det = self.engine._det
        h, w = img.shape[:2]
        if min(h, w) < 16:
            return []
        if self.fastdet is not None:  # uint8 直接上傳、前處理在顯示卡上做（detect.py）
            preds = self.fastdet.prob(np.ascontiguousarray(img))[None, None]
        else:
            # 不縮圖：RapidOCR 預設把長邊縮到 2000，2560 寬的螢幕小字（漫畫、小字小說）會縮成 0.78 倍而漏抓
            # （漫畫頁多找到不少行，整個螢幕偵測 0.57 → 0.84 秒）
            from rapidocr_onnxruntime.ch_ppocr_det.utils import DetPreProcess
            resized = DetPreProcess(max(h, w), "max", det.mean, det.std).resize(img)  # 只把長寬調成 32 的倍數
            # 正規化 (x/255 - 0.5) / 0.5 = (x - 127.5) / 127.5。RapidOCR 的寫法會先轉成 float64，整個螢幕要 0.15 秒
            if resized is None:
                return []
            pre = cv2.dnn.blobFromImage(resized, 1 / 127.5, None, (127.5, 127.5, 127.5), swapRB=False, crop=False)
            preds = det.infer(pre)[0]
        from .detect import db_boxes
        # 兩個門檻只差在「框內平均分數」的篩選，候選框完全相同：用 0.3 跑一次，再依分數分成兩組（省一半後處理）
        # db_boxes 和 RapidOCR 的 DBPostProcess 結果相同，只是少了一些小陣列的開銷
        quads, scores = db_boxes(det.postprocess_op, preds, (h, w), 0.3)
        main, extra = [], []
        for q, sc in zip(quads, scores):
            # 同 filter_tag_det_res（一次一個，才知道是哪個分數）
            q = det.clip_det_res(det.order_points_clockwise(q), h, w)
            if int(np.linalg.norm(q[0] - q[1])) <= 3 or int(np.linalg.norm(q[0] - q[3])) <= 3:
                continue
            (main if sc >= 0.5 else extra).extend(self._quads_to_boxes([q], w, h))
        if not main or not extra:
            return main + extra
        # 0.3 的框和 0.5 的框重疊（交集 ÷ 較小的面積）都不到四成才補
        m = np.array([(b.x0, b.y0, b.x1, b.y1) for b in main], np.int64)
        m_area = (m[:, 2] - m[:, 0]) * (m[:, 3] - m[:, 1])
        keep = []
        for b in extra:
            ix = np.maximum(0, np.minimum(b.x1, m[:, 2]) - np.maximum(b.x0, m[:, 0]))
            iy = np.maximum(0, np.minimum(b.y1, m[:, 3]) - np.maximum(b.y0, m[:, 1]))
            if (ix * iy / np.maximum(1, np.minimum(b.w * b.h, m_area)) < 0.4).all():
                keep.append(b)
        return main + keep

    @staticmethod
    def _quads_to_boxes(quads, w, h) -> list[Box]:
        boxes = []
        for q in quads:
            q = np.asarray(q)
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                boxes.append(Box(int(x0), int(y0), int(x1), int(y1)))
        return boxes

    def read_segments(self, img: np.ndarray, lines, segs) -> str:
        """反白模式：辨識選到的片段（layout.Segment），依序接成一段文字。"""
        return self.join_segments(self.read_segment_parts(img, lines, segs))

    @staticmethod
    def join_segments(parts: list[str]) -> str:
        text = textutil.join_lines([p for p in parts if p])
        return textutil.fix_common_ocr(textutil.clean(text))

    def read_segment_parts(self, img: np.ndarray, lines, segs) -> list[str]:
        """每個片段各自的文字（還沒接起來；讀振假名時要用每欄的文字對字格，見 ruby.py）。"""
        from .layout import cross_pad  # layout 也 import 這個模組，放在這裡避免循環 import
        grays, verts = [], []
        for k, s in enumerate(segs):
            ln = lines[s.line]
            x0, y0, x1, y1 = ln.rect(s.a, s.b)
            lo, hi = cross_pad(lines, s.line, max(2, ln.thickness // 6))
            H, W = img.shape[:2]
            # 沿著行的方向：行首行尾多留 2 像素；切在行中間的切點已經在字距空隙裡，不要再外擴
            # （會帶到隔壁字的一點邊，被讀成別的字：ちょく → あらよく、んが、→ あ、、できて｜ます → できて、）
            pa, pb = (2 if s.a <= 0 else 0), (2 if s.b >= ln.length else 0)
            if ln.vertical:
                cx0 = max(0, x0 - lo)
                crop = img[max(0, y0 - pa):min(H, y1 + pb), cx0:min(W, x1 + hi)]
                crop = _clear_margins(crop, x0 - cx0, x1 - cx0, True)
            else:
                cy0 = max(0, y0 - lo)
                crop = img[cy0:min(H, y1 + hi), max(0, x0 - pa):min(W, x1 + pb)]
                crop = _clear_margins(crop, y0 - cy0, y1 - cy0, False)
            if crop.size == 0:
                grays.append(None)
                verts.append(False)
                continue
            grays.append(_gray_for_rec(crop))
            verts.append(ln.vertical)
        keep = [i for i, g in enumerate(grays) if g is not None]
        texts = self.read_long([grays[i] for i in keep], [verts[i] for i in keep])
        out = [""] * len(segs)
        for i, t in zip(keep, texts):
            out[i] = t
        return out

    def recognize(self, img: np.ndarray, mode: str = "blocks") -> list[Block]:
        if mode == "whole":
            text = textutil.fix_common_ocr(textutil.clean(self.rec_batch([_gray_for_rec(img)])[0]))
            h, w = img.shape[:2]
            return [Block(Box(0, 0, w, h), text, h > w * 1.5)] if text else []

        boxes = self.engine.detect(img)
        if mode == "lines":
            lines = [ocr.Line(b, t) for b, t in zip(boxes, self.read_lines(img, boxes)) if t]
            return [Block(box, text, box.vertical) for box, text in ocr.group_blocks(lines)]

        groups = group_boxes(boxes)
        # 直排小區塊整塊辨識（manga-ocr 本來就是讀整個漫畫對話框），其餘逐行
        whole = [g[0].vertical and 1 < len(g) <= 4 and _union(g).h <= _union(g).w * 4 for g in groups]
        line_boxes = [b for g, w in zip(groups, whole) if not w for b in g]
        line_text = dict(zip(map(id, line_boxes), self.read_lines(img, line_boxes)))
        whole_grays = [_gray_for_rec(_crop(img, _union(g), max(2, min(b.thickness for b in g) // 4)))
                       for g, w in zip(groups, whole) if w]
        whole_text = iter(self.rec_batch(whole_grays) if whole_grays else [])

        out = []
        for g, w in zip(groups, whole):
            if w:
                text = textutil.clean(next(whole_text))
            else:
                text = textutil.join_lines([t for b in g if (t := line_text[id(b)])])
            text = textutil.fix_common_ocr(text)
            if text:
                out.append(Block(_union(g), text, g[0].vertical))
        return out
