# PyInstaller 設定：python packaging/build.py 會呼叫（不要直接用，build.py 還會放 CUDA、說明檔）。
# 只帶執行需要的東西：模型、字典、日檢單字表、文法資料；不帶 testdata/、tests/、vocab/、logs/、config.json、data/raw/ 等。
# 輸出：dist/jplens/（jplens.exe＋_internal/）。
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
REAZON = ROOT / "data" / "models" / "asr" / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"

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
    # 無字幕模式（listen.py）：語音偵測、ReazonSpeech 只帶 int8 版（約 160 MB；完整版 600 MB 不帶）
    (str(ROOT / "data" / "models" / "asr" / "silero_vad.onnx"), "data/models/asr"),
    (str(ROOT / "data" / "models" / "asr" / "silero_vad_v6.onnx"), "data/models/asr"),
    (str(REAZON / "tokens.txt"), "data/models/asr/" + REAZON.name),
    (str(REAZON / "README.md"), "data/models/asr/" + REAZON.name),
    *[(str(f), "data/models/asr/" + REAZON.name) for f in REAZON.glob("*.int8.onnx")],
    # 單字測驗（quiz.py）：判斷選項有沒有同義的語意模型，只帶 int8 版（24 MB）
    (str(ROOT / "data" / "models" / "bge-small-zh" / "model_quantized.onnx"), "data/models/bge-small-zh"),
    (str(ROOT / "data" / "models" / "bge-small-zh" / "tokenizer.json"), "data/models/bge-small-zh"),
    (str(ROOT / "NOTICE.md"), "."),
    (str(ROOT / "LICENSE"), "."),
]
GRAMMAR_EN = ROOT / "jplens" / "jlearn" / "grammar_en.json"  # 文法說明的英文版（tools/grammar_en_batches.py；還沒翻就沒有）
if GRAMMAR_EN.exists():
    datas.append((str(GRAMMAR_EN), "jplens/jlearn"))
UI_EN = ROOT / "jplens" / "i18n_en.json"  # 英文介面的對照表（tools/i18n_check.py）
if UI_EN.exists():
    datas.append((str(UI_EN), "jplens"))
datas += collect_data_files("rapidocr_onnxruntime")
datas += collect_data_files("sudachipy")
datas += collect_data_files("sudachidict_core")

# 只有舊的 PyTorch 版 manga-ocr、建字典、匯出模型的工具用得到，打包不帶（torch 一個就好幾 GB）
excludes = ["torch", "torchvision", "torchaudio", "transformers", "manga_ocr", "huggingface_hub", "fugashi", "unidic_lite",
            "sklearn", "scipy", "pandas", "matplotlib", "tensorflow", "IPython", "jupyter", "notebook", "tkinter",
            "opencc", "onnx", "pytest", "sympy", "numba", "av",
            "faster_whisper", "ctranslate2"]  # 只有 tools/asr_eval.py 用到

a = Analysis(
    [str(ROOT / "packaging" / "jplens_main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    # google.genai、keyring 在 ai.py 裡用到時才 import，PyInstaller 看不到，要明列
    hiddenimports=collect_submodules("jplens") + collect_submodules("dxcam") + ["comtypes.stream"]
    + collect_submodules("google.genai") + collect_submodules("keyring.backends")
    + ["sherpa_onnx", "pyaudiowpatch"]  # 無字幕模式（listen.py 開啟時才 import）
    + ["tokenizers"],  # 單字測驗的語意模型（quiz.py 用到時才 import）
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
