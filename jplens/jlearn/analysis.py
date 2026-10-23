"""本機分析：Sudachi 斷詞 → 單字（辭書形、讀音、日檢級數）與文法（規則比對）。

單字級數一律查 data/jlpt/*.csv，不交給 LLM 判斷（實測 LLM 標的級數只有 67% 和單字表一致）。
LLM 只負責翻譯、中文解釋、例句與情境解說；API 不能用時，這裡的結果就是離線版筆記的全部內容。
"""

import csv
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

JLPT_DIR = Path(__file__).resolve().parents[2] / "data" / "jlpt"

_KATA2HIRA = str.maketrans({chr(c): chr(c - 0x60) for c in range(ord("ァ"), ord("ヶ") + 1)})
_KANA_ONLY = re.compile(r"^[ぁ-ゟァ-ヿー]+$")
_KATAKANA_ONLY = re.compile(r"^[ァ-ヿー]+$")
JP_CHAR = re.compile(r"[ぁ-ゟァ-ヿ㐀-䶿一-鿿々]")


def to_hira(s: str) -> str:
    return s.translate(_KATA2HIRA)


@dataclass
class Token:
    surface: str
    lemma: str        # 辭書形（保留原文的寫法：わかっ → わかる）
    norm: str         # 標準寫法（わかる → 分かる）
    reading: str      # 表層形的讀音（平假名）
    pos: tuple        # Sudachi 的 6 欄詞性
    oov: bool = False

    @property
    def pos1(self) -> str:
        return self.pos[0]

    @property
    def cform(self) -> str:
        return self.pos[5]

    def is_punct(self) -> bool:
        return self.pos1 in ("補助記号", "記号", "空白")


@lru_cache(maxsize=1)
def _dictionary():
    from sudachipy import dictionary
    return dictionary.Dictionary(dict="core")


_local = threading.local()


def _tokenizer():
    """每個執行緒一個斷詞器（共用同一份字典，多一個只多一點記憶體；另外載一份字典要 74 MB）。
    同一個斷詞器被兩個執行緒同時用會丟 SudachiError（查字的背景執行緒和無字幕模式的錄音執行緒都會斷詞）。"""
    tok = getattr(_local, "tok", None)
    if tok is None:
        tok = _local.tok = _dictionary().create()
    return tok


def tokenize(text: str) -> list[Token]:
    from sudachipy import SplitMode
    return [Token(m.surface(), m.dictionary_form(), m.normalized_form(), to_hira(m.reading_form()),
                  tuple(m.part_of_speech()), m.is_oov())
            for m in _tokenizer().tokenize(text, SplitMode.C)]


@lru_cache(maxsize=4096)
def lemma_reading(lemma: str) -> str:
    """辭書形的讀音（Sudachi 只給表層形讀音，所以把辭書形再斷一次詞）。"""
    return "".join(t.reading for t in tokenize(lemma))


# ---------------------------------------------------------------- 日檢單字表

@dataclass(frozen=True)
class JlptEntry:
    word: str
    reading: str
    meaning: str   # 英文
    level: int     # 5 = N5 … 1 = N1


def _forms(s: str, affix: bool = False) -> list[str]:
    """「(花を〜) 生ける, 活ける」→ ["生ける", "活ける"]。
    affix=False 時不收以「～」開頭或結尾的詞綴；True 時只收詞綴並去掉「～」（～頃 → 頃）。"""
    s = re.sub(r"\s*[(（][^)）]*[)）]\s*", "", s)
    out = []
    for f in re.split(r"[;；,、]", s):
        f = f.strip()
        is_affix = bool(re.search(r"[〜～~]", f))
        if f and is_affix == affix:
            f = re.sub(r"[〜～~]", "", f).strip()
            if f:
                out.append(f)
    return out


@lru_cache(maxsize=1)
def jlpt_table() -> tuple[dict, dict]:
    """回傳 (寫法 → entry, 詞綴寫法 → entry)。同一個詞出現在多個級別時取較簡單的級別
    （也就是最早要求會的級別）。

    不做「讀音 → 寫法」的反查：實測會把感嘆詞「おや」配到「親」、「ふむ」配到「踏む」。
    原文寫假名、表上寫漢字的情況（わかる／分かる）靠 Sudachi 的標準寫法就能對上。"""
    by_word, by_affix = {}, {}
    for level in (5, 4, 3, 2, 1):  # 由簡單到難：先登記的優先
        path = JLPT_DIR / f"n{level}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for table, affix in ((by_word, False), (by_affix, True)):
                    words, readings = _forms(row["expression"], affix), _forms(row["reading"], affix)
                    for i, w in enumerate(words):
                        rd = to_hira(readings[min(i, len(readings) - 1)]) if readings else w
                        table.setdefault(w, JlptEntry(w, rd, row["meaning"].strip(), level))
    return by_word, by_affix


def lookup_jlpt(lemma: str, norm: str = "") -> JlptEntry | None:
    """依序用 原文寫法 → 標準寫法 查表，查不到再查詞綴（表上只有「～頃」時，「頃」算同一級）。"""
    for table in jlpt_table():
        for w in (lemma, norm):
            if w and w in table:
                return table[w]
    return None


# ---------------------------------------------------------------- 單字

# 太基本、列了也沒有學習價值的字（相當於提示詞裡說的「する、ある、いる、これ、私」）
STOP_LEMMAS = {"する", "為る", "ある", "有る", "在る", "いる", "居る", "です", "だ", "ます",
               "こと", "事", "もの", "物", "の", "よう", "そう", "ん", "ない", "無い"}
# 接在て形或動詞連用形後面時是補助動詞，屬於文法（〜てしまう、〜すぎる…），不列為單字
_AUX_VERBS = {"いる", "ある", "おく", "しまう", "くる", "いく", "みる", "みせる", "くれる", "もらう",
              "あげる", "やる", "くださる", "いただく", "すぎる", "える", "うる", "なさる", "いらっしゃる",
              "おる", "始める", "出す", "続ける", "終わる", "合う", "直す", "切る", "込む"}


@dataclass
class Word:
    word: str          # 顯示用的辭書形（查到單字表時用表上的寫法）
    reading: str       # 平假名
    pos: str           # 中文化的詞性
    level: int | None  # 5～1；None = 級外
    proper: bool       # 專有名詞
    line_id: int
    meaning_en: str = ""
    key: str = ""      # 去重與對應 LLM 結果用（標準寫法）

    @property
    def jlpt(self) -> str:
        if self.proper:
            return "專有名詞"
        return f"N{self.level}" if self.level else "級外"


def pos_label(t: Token) -> str:
    p = t.pos
    if p[0] == "動詞":
        ctype = p[4]
        if ctype.startswith("五段"):
            return "五段動詞"
        if "一段" in ctype:
            return "一段動詞"
        if ctype.startswith("サ行"):
            return "サ變動詞"
        if ctype.startswith("カ行"):
            return "カ變動詞"
        return "動詞"
    if p[0] == "形容詞":
        return "い形容詞"
    if p[0] == "形状詞":
        return "な形容詞"
    if p[0] == "名詞":
        if p[1] == "固有名詞":
            return "專有名詞"
        if p[2] == "形状詞可能":
            return "名詞／な形容詞"
        if p[2] == "サ変可能":
            return "名詞（する動詞）"
        return "名詞"
    return {"副詞": "副詞", "連体詞": "連體詞", "接続詞": "接續詞", "感動詞": "感嘆詞"}.get(p[0], p[0])


def _is_aux_use(tokens: list[Token], i: int) -> bool:
    t = tokens[i]
    if t.pos1 != "動詞" or i == 0:
        return False
    prev = tokens[i - 1]
    if t.lemma == "する" or t.norm == "為る":
        return True
    if t.lemma in _AUX_VERBS or t.norm in _AUX_VERBS:
        if prev.pos1 == "助詞" and prev.lemma in ("て", "で", "ちゃ", "じゃ"):
            return True
        if prev.pos1 in ("動詞", "形容詞") and prev.cform.startswith(("連用形", "語幹")):
            return True
    return False


def _candidate(tokens: list[Token], i: int, speakers: set[str], pronouns: bool = False) -> bool:
    """pronouns：代名詞也算（呼叫端自己再排除基本的，jplens/words.py）。"""
    t = tokens[i]
    p1 = t.pos1
    if not JP_CHAR.search(t.surface):
        return False  # 英文、數字、符號
    # 感嘆詞不列：えっ、うーん 沒有學習價值，而且查表會配錯（感嘆詞「ほう」配到「方」）
    if p1 in ("助詞", "助動詞", "補助記号", "記号", "空白", "接頭辞", "接尾辞", "感動詞") or (p1 == "代名詞" and not pronouns):
        return False
    if t.pos[1] == "数詞" or t.lemma in STOP_LEMMAS or t.norm in STOP_LEMMAS:
        return False
    if t.surface in speakers:
        return False
    if p1 == "形状詞" and t.pos[1] == "助動詞語幹":
        return False
    if _is_aux_use(tokens, i):
        return False
    if t.oov and not (_KATAKANA_ONLY.match(t.surface) and len(t.surface) >= 3):
        return False  # 字典裡沒有的假名碎片多半是 OCR 雜訊
    if len(t.surface) == 1 and _KANA_ONLY.match(t.surface):
        return False
    return True


def _words_in_line(tokens: list[Token], line_id: int, speakers: set[str], skip: set[int],
                   seen: set[str]) -> list[Word]:
    """skip：已經屬於某個文法句型的詞（にとって 的「取る」、つもり、わけ…），不重複列為單字。"""
    words = []
    for i, t in enumerate(tokens):
        if i in skip or not _candidate(tokens, i, speakers):
            continue
        entry = lookup_jlpt(t.lemma, t.norm)
        proper = t.pos[1] == "固有名詞"
        key = entry.word if entry else (t.norm or t.lemma)
        if key in seen:
            continue
        seen.add(key)
        word = entry.word if entry else t.lemma
        if _KATAKANA_ONLY.match(word):
            reading = word  # 外來語：讀音就是本身（單字檔的 comment 會留空）
        else:
            reading = entry.reading if entry else lemma_reading(t.lemma)
        words.append(Word(
            word=word,
            reading=reading,
            pos=pos_label(t),
            level=None if proper or not entry else entry.level,
            proper=proper,
            line_id=line_id,
            meaning_en=entry.meaning if entry else "",
            key=key,
        ))
    return words


# ---------------------------------------------------------------- 整體分析

@dataclass
class Analysis:
    words: list[Word] = field(default_factory=list)
    grammar: list = field(default_factory=list)  # list[grammar.GrammarHit]


def analyze(entries: list[dict]) -> Analysis:
    """單字：依出現順序、同一個字只列一次；文法：依第一次出現的順序，附上所有出現的句子。"""
    from .grammar import GrammarHit, match_line_spans
    speakers = {e["speaker"] for e in entries if e.get("speaker")}
    words, seen, hits = [], set(), {}
    for line_id, e in enumerate(entries, 1):
        tokens = tokenize(e["text"])
        in_grammar = set()
        for rule, start, end in match_line_spans(tokens):
            h = hits.setdefault(rule.id, GrammarHit(rule, []))
            if line_id not in h.line_ids:
                h.line_ids.append(line_id)
            if end - start > 1 or tokens[start].pos1 in ("名詞", "動詞", "形容詞"):
                in_grammar.update(range(start, end))
        words += _words_in_line(tokens, line_id, speakers, in_grammar, seen)
    return Analysis(words=words, grammar=list(hits.values()))
