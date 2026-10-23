"""解釋的語言（Config.explain_lang）：單字意思、整句翻譯、講解用繁體中文還是英文。

結果視窗的單字列、單字本的卡片這些地方拿不到 Config，所以存成模組變數：app.py 啟動時、設定改了時呼叫 set_explain()。
介面語言（Config.ui_lang：選單、按鈕的文字）是另一個設定，見 i18n.py。
"""

EXPLAIN_LANGS = {"zh-TW": "繁體中文", "en": "English"}  # i18n: 不翻
_explain = "zh-TW"


def set_explain(lang: str):
    global _explain
    _explain = lang if lang in EXPLAIN_LANGS else "zh-TW"


def explain() -> str:
    return _explain


def english() -> bool:
    """單字意思、翻譯用英文：字典不取中文（只有 JMdict 的英文，維基詞典的條目不用），翻譯、講解的提示詞改成英文。"""
    return _explain == "en"


def rec_meanings(rec: dict) -> tuple[list[str], bool]:
    """單字本記錄要顯示的意思：（意思, 是不是「備用」的英文）。
    中文模式：有中文用中文，沒有才用英文（標〔英〕）；英文模式：有英文用英文，沒有才用中文（中文模式時記的字）。"""
    zh, en = list(rec.get("meaning_zh") or []), list(rec.get("meaning_en") or [])
    if english():
        return (en or zh), False
    return (zh, False) if zh else (en, bool(en))
