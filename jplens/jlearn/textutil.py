"""文字正規化與日文判斷工具。"""

import re
import unicodedata
from difflib import SequenceMatcher

# 平假名、片假名（含半形）、長音、々
_KANA = r"぀-ゟ゠-ヿㇰ-ㇿｦ-ﾟ"
_KANJI = r"㐀-䶿一-鿿豈-﫿々"
JP_CHAR = re.compile(f"[{_KANA}{_KANJI}]")
KANA_CHAR = re.compile(f"[{_KANA}]")

# \W 在 Python 中不含假名/漢字，所以這會移除標點、空白與符號
_NON_WORD = re.compile(r"[\W_]+")
_FULLWIDTH_ALNUM = re.compile(r"[０-９Ａ-Ｚａ-ｚ]")
_ELLIPSIS = re.compile(r"[．.・･]{2,}|…+")
SENTENCE_PUNCT = re.compile(r"[。、！？!?…「」『』（）()]")


def norm(s: str) -> str:
    """比對用：NFKC 後去除所有標點與空白。"""
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", s))


def jp_count(s: str) -> int:
    return len(JP_CHAR.findall(s))


def jp_ratio(s: str) -> float:
    n = norm(s)
    return jp_count(n) / len(n) if n else 0.0


def is_japanese(s: str, min_ratio: float = 0.3) -> bool:
    return jp_count(s) >= 1 and jp_ratio(s) >= min_ratio


def looks_chinese(s: str) -> bool:
    """夠長卻完全沒有假名 → 幾乎可以確定是中文（例如雙語字幕的中文那一行）。"""
    k = norm(s)
    return len(k) >= 6 and not KANA_CHAR.search(k)


def clean(s: str) -> str:
    """顯示用：統一省略號、全形英數轉半形、收斂空白。"""
    s = s.replace("＂", '"').replace("＇", "'")
    s = _FULLWIDTH_ALNUM.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    s = _ELLIPSIS.sub(lambda m: "……" if len(m.group()) >= 4 else "…", s)
    return " ".join(s.split())


# 片假名後面的「一／－」幾乎都是長音「ー」被誤認（パエト一ン）；後面接漢字時保留（ケーキ一個）
_LONG_VOWEL = re.compile(r"(?<=[ァ-ヺ])[一－‐](?=[ァ-ヺー」』）)、。…！？!?\s]|$)")


def fix_common_ocr(s: str) -> str:
    """修正 manga-ocr 固定會犯、且可安全判斷的錯誤。"""
    s = _LONG_VOWEL.sub("ー", s)
    # 「 還沒閉合時，緊接在日文字後的 J／j 一定是 」 的誤認（「パエトーンJ様）；
    # 「！」後面直接接日文、而且後面再也沒有 」 時也是（「パエトーン！様）
    out, open_ = list(s), False
    for i, ch in enumerate(s):
        if ch == "「":
            open_ = True
        elif ch == "」":
            open_ = False
        elif open_ and i > 0 and JP_CHAR.match(s[i - 1]):
            nxt = s[i + 1] if i + 1 < len(s) else ""
            rest = s[i + 1:].split("「", 1)[0]
            if ch in "Jj" or (ch in "！!" and JP_CHAR.match(nxt) and "」" not in rest):
                out[i] = "」"
                open_ = False
    # 「く」常被讀成「〈」（ちょくちょ〈）：後面沒有對應的「〉」、前面緊接平假名的「〈」改回「く」
    # （真的括號「画〈字〉」「太郎〈17歳〉」有成對的「〉」，前面也多半不是平假名）
    for i, ch in enumerate(out):
        if ch == "〈" and i > 0 and "ぁ" <= out[i - 1] <= "ゖ" and "〉" not in out[i + 1:]:
            out[i] = "く"
    return "".join(out)


def join_lines(parts) -> str:
    """把同一個文字區塊的多行接起來；日文不加空白，英文單字之間補空白。"""
    out = ""
    for p in parts:
        if out and re.match(r"[A-Za-z0-9]", p[:1]) and re.search(r"[A-Za-z0-9]$", out):
            out += " "
        out += p
    return out


def same_text(a: str, b: str) -> bool:
    """兩段（已 norm 的）字串是否應視為同一句台詞。

    - 完全相同
    - 一方是另一方的前綴（打字機逐字出現；容許少量 OCR 誤差）
    - 夠長且高度相似（OCR 個別字誤認）
    """
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 2 and (len(short) >= 6 or len(short) >= 0.3 * len(long_)):
        head = long_[:len(short)]
        if head == short or (len(short) >= 6 and SequenceMatcher(None, short, head, autojunk=False).ratio() >= 0.85):
            return True
    if min(len(a), len(b)) >= 6:
        return SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.75
    return False


def fmt_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"
