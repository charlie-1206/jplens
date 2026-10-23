"""ONNX Runtime 用顯示卡的準備：CUDA（NVIDIA，onnxruntime-gpu）或 DirectML（其他顯示卡，DirectML 版的打包用 onnxruntime-directml）。
兩個套件不能裝在同一個環境，所以實際上只會有其中一種；都不能用時用 CPU。用哪個由 backend() 決定。

onnxruntime-gpu 需要 CUDA 12＋cuDNN 9 的 DLL。以前程式有 import torch，這些 DLL 是 torch 順便載入的；
manga-ocr 改用 ONNX、不再 import torch 之後，要自己先載入，否則 ONNX Runtime 會「悄悄」退回 CPU。
依序找（找到就停）：
1. 已經找得到（在 PATH 上，或這個程序已經載入過）
2. data/cuda/（打包給別人用時放這裡）
3. pip 的 nvidia-* 套件（onnxruntime.preload_dlls() 預設的位置）
4. 開發環境：torch 的 lib 資料夾（只借 DLL，不 import torch）
"""

import ctypes
import importlib.util
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path

from .i18n import t

log = logging.getLogger("jplens")
LOCAL_CUDA = Path(__file__).resolve().parents[1] / "data" / "cuda"
_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
CUDA_OPTS = {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}  # 每遇到新尺寸不用先試遍演算法
# 筆電有兩張顯示卡時預設的 0 號常是內顯：選效能高的那張（10-22 實測 RTX 4060 筆電：0 號＝NVIDIA，1 號 Intel UHD 慢 30 倍）
DML_OPTS = {"performance_preference": "high_performance"}
CPU = "CPUExecutionProvider"
_lock = threading.Lock()
_backend: str | None = None


def _loadable() -> bool:
    try:
        for name in _DLLS:
            ctypes.WinDLL(name)
        return True
    except OSError:
        return False


def _candidates():
    if LOCAL_CUDA.is_dir():
        yield str(LOCAL_CUDA)
    yield None  # pip 的 nvidia-* 套件
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        lib = Path(spec.origin).parent / "lib"
        if lib.is_dir():
            yield str(lib)


@lru_cache(maxsize=1)
def cuda_ready() -> bool:
    """有 NVIDIA 顯示卡、裝了 onnxruntime-gpu、找得到 CUDA／cuDNN 的 DLL → True。"""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    if _loadable():
        return True
    for d in _candidates():
        try:
            ort.preload_dlls(directory=d) if d is not None else ort.preload_dlls()
        except Exception:  # noqa: BLE001（找不到時各版本的行為不同：警告或例外）
            continue
        if _loadable():
            log.info("CUDA 程式庫：%s", d or "pip 的 nvidia 套件")
            return True
    log.info("找不到 CUDA 12／cuDNN 9 的程式庫，用 CPU")
    return False


def _dml_faster() -> bool:
    """DirectML 實測比 CPU 快才用：弱的內顯反而慢很多（10-22 實測 Intel UHD：整個螢幕偵測 4.1 秒，CPU 0.66 秒）。
    用整個螢幕偵測的模型跑一張 640×640：各先跑一次暖機，再跑 3 次取最快的（10-23：只量一次時，
    打包後的自我測試偶爾量成 DirectML 比 CPU 慢、改用 CPU；顯示卡剛從省電狀態喚醒之類的雜訊）。
    約 1～2 秒，只在沒有 CUDA、有 DirectML 時跑一次。"""
    import numpy as np
    import onnxruntime as ort

    from .detect import model_path
    path = model_path()
    if path is None:
        return True
    x = np.full((1, 640, 640, 3), 245, np.uint8)
    times = {}
    for name, prov in (("cpu", [CPU]), ("dml", [("DmlExecutionProvider", DML_OPTS), CPU])):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if name == "dml":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(str(path), so, providers=prov)
        feed = {sess.get_inputs()[0].name: x}
        sess.run(None, feed)
        runs = []
        for _ in range(3):
            t = time.perf_counter()
            sess.run(None, feed)
            runs.append(time.perf_counter() - t)
        times[name] = min(runs)
    log.info("DirectML %.0f ms、CPU %.0f ms（640×640 文字偵測）", times["dml"] * 1000, times["cpu"] * 1000)
    return times["dml"] < times["cpu"] * 0.7


def backend() -> str:
    """"cuda"、"dml" 或 "cpu"。第一次呼叫時決定（DirectML 要實測），之後都一樣。"""
    global _backend
    with _lock:
        if _backend is None:
            _backend = "cpu"
            if cuda_ready():
                _backend = "cuda"
            else:
                try:
                    import onnxruntime as ort
                    if "DmlExecutionProvider" in ort.get_available_providers() and _dml_faster():
                        _backend = "dml"
                except Exception:  # noqa: BLE001（驅動太舊、DirectML 初始化失敗）
                    log.exception("DirectML 無法使用，用 CPU")
            log.info("推論裝置：%s", name())
        return _backend


def name() -> str:
    return {"cuda": t("CUDA（NVIDIA 顯示卡）"), "dml": t("DirectML（顯示卡）"), "cpu": "CPU"}[_backend or "cpu"]


def providers() -> list:
    b = backend()
    if b == "cuda":
        return [("CUDAExecutionProvider", CUDA_OPTS), CPU]
    if b == "dml":
        return [("DmlExecutionProvider", DML_OPTS), CPU]
    return [CPU]


def session_options(so):
    """DirectML 的 session 要關掉 memory pattern、照順序執行（ONNX Runtime 的要求）。回傳 so。"""
    if backend() == "dml":
        so.enable_mem_pattern = False
        import onnxruntime as ort
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def on_gpu(sess) -> bool:
    """這個 session 實際有用到顯示卡（ONNX Runtime 找不到時會悄悄退回 CPU）。"""
    return sess.get_providers()[0] != CPU
