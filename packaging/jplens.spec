# PyInstaller 設定：python packaging/build.py 會呼叫（不要直接用，build.py 還會放 CUDA、說明檔）。
# 只帶執行需要的東西：模型、字典、日檢單字表、文法資料；不帶 testdata/、tests/、vocab/、logs/、config.json、data/raw/ 等。
# 輸出：dist/jplens/（jplens.exe＋_internal/）。
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "data" / "dict.sqlite"), "data"),
    (str(ROOT / "data" / "jlpt"), "data/jlpt"),
    (str(ROOT / "data" / "models" / "manga-ocr-onnx"), "data/models/manga-ocr-onnx"),
    (str(ROOT / "data" / "models" / "ogkalu"), "data/models/ogkalu"),
    (str(ROOT / "data" / "models" / "ppocr_det" / "det_u8_fp32.onnx"), "data/models/ppocr_det"),
    (str(ROOT / "dict_translate" / "zh.jsonl"), "dict_translate"),
    (str(ROOT / "dict_translate" / "zh_api.jsonl"), "dict_translate"),
    (str(ROOT / "jplens" / "jlearn" / "grammar_notes.json"), "jplens/jlearn"),
    (str(ROOT / "jplens" / "jlearn" / "grammar_rules_extra.jsonl"), "jplens/jlearn"),
    (str(ROOT / "NOTICE.md"), "."),
    (str(ROOT / "LICENSE"), "."),
]
datas += collect_data_files("rapidocr_onnxruntime")
datas += collect_data_files("sudachipy")
datas += collect_data_files("sudachidict_core")

# 只有舊的 PyTorch 版 manga-ocr、建字典、匯出模型的工具用得到，打包不帶（torch 一個就好幾 GB）
excludes = ["torch", "torchvision", "torchaudio", "transformers", "manga_ocr", "huggingface_hub", "fugashi", "unidic_lite",
            "sklearn", "scipy", "pandas", "matplotlib", "tensorflow", "IPython", "jupyter", "notebook", "tkinter",
            "opencc", "google", "onnx", "pytest", "sympy", "numba", "av"]

a = Analysis(
    [str(ROOT / "packaging" / "jplens_main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=collect_submodules("jplens") + collect_submodules("dxcam") + ["comtypes.stream"],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="jplens",
    icon=str(ROOT / "build" / "jplens.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="jplens", upx=False)
