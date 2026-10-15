"""ONNX Runtime 用 NVIDIA 顯示卡（CUDA）的準備。

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
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("jplens")
LOCAL_CUDA = Path(__file__).resolve().parents[1] / "data" / "cuda"
_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
CUDA_OPTS = {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}  # 每遇到新尺寸不用先試遍演算法


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


def providers() -> list:
    return [("CUDAExecutionProvider", CUDA_OPTS), "CPUExecutionProvider"] if cuda_ready() else ["CPUExecutionProvider"]
