"""介面語言（Config.ui_lang）：選單、按鈕、提示、訊息的文字。

程式裡給使用者看的中文都包成 t("中文")：中文原文就是查表的鍵，英文介面時查 i18n_en.json（{中文: 英文}），查不到就顯示中文。
有變數的寫成 t("已複製「{word}」", word=w)：先查表再填變數（英文的語序可能不同）。
日誌、例外的內部訊息不翻。

語言在 import 時從 config.json 讀一次（模組層級的常數，例如選取模式的名稱，在 import 時就呼叫 t()），
所以改了介面語言要重開程式才生效。對照表的維護：tools/i18n_check.py。
"""

import json
from pathlib import Path

EN_PATH = Path(__file__).with_name("i18n_en.json")
UI_LANGS = {"zh-TW": "繁體中文", "en": "English"}


def _load_lang() -> str:
    """config.json 的 ui_lang；環境變數 JPLENS_UI_LANG 優先（測試、截圖用）。"""
    import os
    from .config import Config
    if os.environ.get("JPLENS_UI_LANG") in UI_LANGS:
        return os.environ["JPLENS_UI_LANG"]
    try:
        lang = Config.load().ui_lang
    except Exception:  # noqa: BLE001（設定檔壞掉：Config.load 本身會用預設值，這裡只是保險）
        lang = "zh-TW"
    return lang if lang in UI_LANGS else "zh-TW"


_lang = _load_lang()
_table: dict[str, str] = {}
if _lang == "en":
    try:
        _table = json.loads(EN_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _table = {}


def ui_lang() -> str:
    return _lang


def english() -> bool:
    return _lang == "en"


def t(text: str, /, **kw) -> str:
    """介面文字：英文介面時換成英文（沒翻到就用原文）；kw 是要填進 {名稱} 的變數（可以叫 text，所以 text 只能用位置傳）。"""
    s = _table.get(text) or text
    return s.format(**kw) if kw else s


def mark(*texts: str) -> tuple[str, ...]:
    """列出從資料來、顯示時才 t(變數) 的值（詞性、級數名稱），讓 tools/i18n_check.py collect 收進要翻的字串。原樣回傳。"""
    return texts
