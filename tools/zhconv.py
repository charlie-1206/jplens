"""簡轉繁：只轉真的含有簡體字的文字。

OpenCC 對已經是繁體的文字也會改：s2twp 會換詞（掛斷 → 結束通話、打開 → 開啟），
連只轉字的 s2tw 也會把繁體裡本來就正確的字改掉（傢伙 → 傢夥、酸痛 → 痠痛）。
所以先檢查有沒有「簡體專用字」，有才轉；沒有就原樣保留。
"""

from functools import lru_cache

# OpenCC 會轉換、但在繁體中文裡本來就是正確用字的字（不能拿來判斷是不是簡體）
_SHARED = set("伙干后里台系只余准制松才冲卷斗朴范谷丑征志吁布向尽了借酸面几占据采游划污喫")


@lru_cache(maxsize=1)
def _cc():
    import opencc
    return opencc.OpenCC("s2tw"), opencc.OpenCC("s2twp")  # s2tw：轉成台灣標準字形（不會把「吃、群、床」換成「喫、羣、牀」）


def simplified_chars(text: str) -> list[str]:
    """text 裡的簡體專用字（轉成繁體後字形會變、而且不是繁體也通用的字）。"""
    s2t, _ = _cc()
    return [ch for ch in dict.fromkeys(text) if ch not in _SHARED and s2t.convert(ch) != ch]


def to_traditional(text: str) -> str:
    """含簡體專用字才用 s2twp（連詞彙一起轉成台灣用法：软件 → 軟體）；已經是繁體就不動。"""
    return _cc()[1].convert(text) if simplified_chars(text) else text
