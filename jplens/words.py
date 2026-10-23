"""OCR 文字 → 單字清單（辭書形、讀音、詞性、日檢級數、意思、出現的句子）。

斷詞與「哪些詞值得列」沿用 video_translate 的 analysis（Sudachi；略過助詞、補助動詞、
已被文法句型涵蓋的詞等）。查日檢單字表這一步自己做，因為要比對讀音：
- 原文「そのとき」的標準寫法是「時」，但表上的「時」是「～時（じ）」→ 讀音對不上就不採用
- 「信じてる」的辭書形 Sudachi 給「信ずる」（N2）、標準寫法是「信じる」（N3）→ 取讀音和原文相符的「信じる」
"""

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import lang
from .dictionary import Dictionary, token_cats
from .i18n import mark
from .jlearn import analysis, grammar

# 句尾：句號、問號、驚嘆號、右引號之後；後面緊接著右引號或其他句尾符號時不切（「…。」的」留在同一句）
_SENT_END = re.compile(r"(?<=[。！？!?」』])(?![。！？!?」』）)])")
GRAMMAR_POS = "文法"
# 顯示時才翻的值（WordInfo.pos、jlpt 存的是中文；結果視窗、單字本顯示時 t(值)）：analysis.pos_label 的詞性名稱、
# Sudachi 原本的詞性（pos_label 沒對照的）、慣用語、複合詞、文法、級數
mark("五段動詞", "一段動詞", "サ變動詞", "カ變動詞", "動詞", "い形容詞", "な形容詞", "專有名詞", "名詞／な形容詞", "名詞（する動詞）",
     "名詞", "副詞", "連體詞", "接續詞", "感嘆詞", "代名詞", "接頭辞", "接尾辞", "助詞", "助動詞", "形状詞", "連体詞",
     "慣用語", "複合詞", "文法", "名詞（專有名詞？）", "級外")


@dataclass
class WordInfo:
    word: str
    reading: str
    pos: str
    level: int | None      # 5～1；None = 級外
    proper: bool
    sentence: str
    zh: list[str] = field(default_factory=list)
    en: list[str] = field(default_factory=list)
    jlpt_en: str = ""      # 日檢單字表的英文意思
    formation: str = ""    # 文法句型的接續方式（只有文法有）
    also: list[tuple[str, str]] = field(default_factory=list)  # 同寫法的其他常用讀音：(讀音, 意思)
    rare_zh: int = 0       # zh／en 最後幾個是冷僻的詞義（清單不顯示，滑鼠提示才列）
    rare_en: int = 0
    notes: list[dict] = field(default_factory=list)  # 文法：JLPT_grammer 的說明（滑鼠提示用，grammar_notes.json）

    @property
    def is_grammar(self) -> bool:
        return self.pos == GRAMMAR_POS

    @property
    def jlpt(self) -> str:
        if self.proper:
            return "專有名詞"
        return f"N{self.level}" if self.level else "級外"

    @property
    def meaning(self) -> str:
        """顯示與存檔用的意思：有中文用中文，否則用英文。"""
        if self.zh:
            return "；".join(self.zh)
        return "; ".join(self.en) if self.en else self.jlpt_en


# ---------------------------------------------------------------- 日檢單字表（保留同一寫法的所有讀音）

@lru_cache(maxsize=1)
def _jlpt_lists() -> tuple[dict, dict]:
    """(寫法 → [entry...], 詞綴寫法 → [entry...])，同一寫法依級別由簡單到難排列。
    和 analysis.jlpt_table 相同的解析方式，但不會因為寫法相同就只留第一個讀音。"""
    by_word, by_affix = {}, {}
    for level in (5, 4, 3, 2, 1):
        path = analysis.JLPT_DIR / f"n{level}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for table, affix in ((by_word, False), (by_affix, True)):
                    words, readings = analysis._forms(row["expression"], affix), analysis._forms(row["reading"], affix)
                    for i, w in enumerate(words):
                        rd = analysis.to_hira(readings[min(i, len(readings) - 1)]) if readings else w
                        if rd.endswith("する") and not w.endswith("る"):
                            rd = rd[:-2]  # N4 的サ變名詞讀音欄寫成「にゅうがくする」，寫法只有「入学」（擦る／為る 的 する 是真的讀音）
                        entry = analysis.JlptEntry(w, rd, row["meaning"].strip(), level)
                        if all(e.reading != rd for e in table.get(w, [])):
                            table.setdefault(w, []).append(entry)
    return by_word, by_affix


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _base_form(t: analysis.Token) -> str:
    """辭書形。Sudachi 會把「信じ」「感じ」的辭書形給成「信ずる」「感ずる」，這時改用標準寫法「信じる」。"""
    if t.lemma.endswith("ずる") and t.norm.endswith("じる"):
        return t.norm
    return t.lemma or t.surface


_E_ROW = dict(zip("うくぐすつぬぶむる", "えけげせてねべめれ"))


def _norm(t: analysis.Token, dictionary: Dictionary) -> str:
    """Sudachi 的標準寫法，只在它是同一個詞時才用：JMdict 同一個條目的寫法（ひっぱる → 引っ張る、だいぶ → 大分），
    辭書形是標準寫法的可能動詞（言える → 言う、書ける → 書く）、文語或「〜す」的舊形（若し → 若い、愛す → 愛する）。
    標準寫法也會把口語變體換成另一個詞（乗っかる → 乗る、ぶっ壊す → 打ち壊す，字典裡是不同條目）：這時改用辭書形。
    不用 Sudachi 的讀音判斷：單獨念漢字常念錯（言う → ゆう、大分 → おおいた）。"""
    n, lemma = t.norm, t.lemma
    if not n or n == lemma or not lemma or t.pos[4].startswith("文語") or lemma + "る" == n:
        return n
    if n[-1:] in _E_ROW and lemma == n[:-1] + _E_ROW[n[-1]] + "る":
        return n
    return n if dictionary.same_word(lemma, n) else lemma


def _ref_reading(t: analysis.Token) -> str:
    """用來比對的讀音：沒有活用的詞用原文讀音（風＝ふう／かぜ 要看原文），
    有活用的詞用辭書形的讀音（「来た」的 き 會誤配到「来る（きたる）」）。"""
    if t.surface == t.lemma:
        return t.reading
    return analysis.lemma_reading(_base_form(t))


def _entry_reading(e) -> str:
    """單字表少數列的寫法與讀音欄位對調（いただく／頂く），取有假名的那一欄。"""
    return e.reading if analysis._KANA_ONLY.match(e.reading) else analysis.to_hira(e.word)


def lookup_jlpt(t: analysis.Token, norm: str, dictionary: Dictionary | None = None):
    """依 辭書形 → 標準寫法 找候選，取讀音開頭和原文最相符的；完全對不上（第一個假名就不同）視為別的詞。
    有 dictionary 時，對不上的讀音如果和表上的是 JMdict 同一條目的兩個念法，也算同一個詞
    （Sudachi 單獨念「言う」給 ゆう，表上是 いう：取表上的讀音和級數）。"""
    for table in _jlpt_lists():
        cands = [e for w in dict.fromkeys((_base_form(t), t.lemma, norm)) for e in table.get(w, [])]
        if cands:
            break
    else:
        # 原文寫漢字、表上只有假名寫法（出来る → できる）：只配「表上本來就寫假名」的條目，
        # 不做反方向（假名 → 漢字）的配對，以免「おや」配到「親」
        rd = analysis.lemma_reading(_base_form(t))
        cands = [e for e in _jlpt_lists()[0].get(rd, []) if analysis._KANA_ONLY.match(e.word)]
        if not cands:
            return None
    # 辭書形讀音和原文讀音都比：Sudachi 單獨念「言う」會給 ゆう，這時要靠原文的「言っ（いっ）」
    refs = [r for r in dict.fromkeys((_ref_reading(t), t.reading)) if r]
    if not refs:
        return cands[0]

    def score(e):
        return max(_common_prefix(_entry_reading(e), r) for r in refs)
    best = max(cands, key=score)  # 同分取前面的（較簡單的級別）
    if score(best) > 0:
        return best
    if dictionary is not None:
        for e in cands:
            if any(dictionary.same_entry(e.word, _entry_reading(e), r) for r in refs):
                return e
    return None


# ---------------------------------------------------------------- 慣用說法（多個詞合起來才有意思）

EXPR_POS = "慣用語"
COMPOUND_POS = "複合詞"  # 全部由名詞（與接頭、接尾詞）組成的：ボードゲーム、お言葉、賛否両論
_COMPOUND_PARTS = {"名詞", "接頭辞", "接尾辞"}
MAX_EXPR = 6  # 最多幾個詞組成一個慣用說法
_EXPR_START = {"名詞", "動詞", "形容詞", "副詞", "形状詞", "連体詞", "代名詞", "接頭辞"}
_EXPR_END = {"名詞", "動詞", "形容詞", "形状詞", "副詞"}
_NEG = {"ない", "ぬ", "ず", "ん"}
_INFLECTING = {"動詞", "形容詞", "助動詞"}
# 動詞未然形＋文語助動詞合成一個詞（知ら＋しめる → 知らしめる、思わ＋ず → 思わず）：Sudachi 會拆開，助動詞結尾也要試
_EXPR_END_AUX = {"しめる", "ず", "ぬ", "まい", "べし"}
# 慣用語中間的が／を被換成提示、強調的助詞（気もつかない、腹は立つ）：換回が／を再查
_TOPIC = {"は", "も", "こそ", "さえ", "でも", "なんか", "なんて", "すら"}
# 以這些形式名詞開頭的是文法句型（ことにする、はずがない、わけじゃない），字典常配到同音的別的詞（異にする），不列
_FORMAL_NOUNS = {"こと", "事", "もの", "物", "はず", "筈", "わけ", "訳", "ところ", "所", "よう", "様", "つもり", "積もり"}
# 「そう言われると」「ああ言っている」的 言う 是真的在說話，不是「そういう／ああいう（那樣的）」
_DEMONSTRATIVE = {"こう", "そう", "ああ", "どう"}
# 五段動詞的 u 段 → a 段（否定形）
_A_ROW = dict(zip("うくぐすつぬぶむる", "わかがさたなばまら"))


def _negative(t: analysis.Token) -> str:
    """動詞辭書形 → ない形（なる → ならない、つく → つかない、改める → 改めない）。只處理常見活用。"""
    lemma, ctype = t.lemma, t.pos[4]
    if not lemma:
        return ""
    if ctype.startswith("五段") and lemma[-1] in _A_ROW:
        return lemma[:-1] + _A_ROW[lemma[-1]] + "ない"
    if "一段" in ctype and lemma.endswith("る"):
        return lemma[:-1] + "ない"
    if ctype.startswith("サ行") and lemma.endswith("する"):
        return lemma[:-2] + "しない"
    if ctype.startswith("カ行"):
        return lemma[:-2] + "こない"
    return ""


def _expr_forms(tokens: list[analysis.Token], i: int, j: int) -> list[str]:
    """tokens[i:j] 可能的字典寫法：前面照原文，最後一個詞換回辭書形；
    後面接否定（なりません）時也試否定形（聞き捨てならない）；名詞直接接動詞時也試補上を／が（胸張る → 胸を張る）。
    不用標準寫法拼（たった一つ 的標準寫法 唯一つ 會配到「唯一」）；有活用的詞不用活用後的樣子（説明し、してい）。"""
    span, last = tokens[i:j], tokens[j - 1]
    if span[0].lemma in _FORMAL_NOUNS or span[0].surface in _FORMAL_NOUNS:
        return []
    if last.lemma == "言う" and span[0].surface in _DEMONSTRATIVE and last.surface != last.lemma:
        return []
    negated = any(t.lemma in _NEG or t.surface in _NEG for t in tokens[j:j + 3] if t.pos1 == "助動詞")
    head = "".join(t.surface for t in span[:-1])
    heads = [head]
    if len(span) >= 3 and span[0].pos1 == "名詞" and span[1].pos1 == "助詞":
        rest = "".join(t.surface for t in span[2:-1])
        if span[1].surface in _TOPIC:  # 気もつかない → 気がつく、腹は立つ → 腹が立つ
            heads += [span[0].surface + p + rest for p in ("が", "を")]  # 換回に會配錯（糸は紡がれ → 糸に紡ぐ）
        elif span[1].surface == "に" and len(span) >= 4 and span[2].surface in ("は", "も"):  # 手には負えない → 手に負えない
            heads.append(span[0].surface + "に" + "".join(t.surface for t in span[3:-1]))
    tails = [last.lemma]
    if last.pos1 not in _INFLECTING:
        tails.append(last.surface)
    if negated and last.pos1 == "動詞":
        tails.insert(0, _negative(last))
    forms = [h + tail for h in heads for tail in tails if tail]
    if len(span) == 2 and span[0].pos1 == "名詞" and last.pos1 == "動詞":
        forms += [head + p + last.lemma for p in ("を", "が")]  # 口語常省略を、が；に很少省略（ともすると ≠ ともにする）
    return list(dict.fromkeys(forms))


def find_expressions(tokens: list[analysis.Token], dictionary: Dictionary) -> list[tuple[int, int, str]]:
    """找出句子裡字典有收錄的慣用說法 (起點, 終點, 寫法)：由左而右，同一起點取最長的，不重疊。
    例：気になって → 気になる、聞き捨てなりません → 聞き捨てならない、ため息をついた → ため息をつく。"""
    cands = []
    for i, t in enumerate(tokens):
        if t.pos1 not in _EXPR_START:
            continue
        for j in range(i + 2, min(len(tokens), i + MAX_EXPR) + 1):
            if tokens[j - 1].is_punct():
                break
            last = tokens[j - 1]
            if last.pos1 in _EXPR_END or (last.pos1 == "助動詞" and last.lemma in _EXPR_END_AUX
                                          and tokens[j - 2].pos1 == "動詞"
                                          and not (j - i == 2 and t.lemma in ("する", "す", "為る"))):  # すべし、せず 是文法，不是詞
                cands.append((i, j, _expr_forms(tokens, i, j)))
    found = dictionary.existing_phrases(f for _, _, forms in cands for f in forms)
    out, covered = [], -1
    for i in range(len(tokens)):
        if i < covered:
            continue
        best = max(((j, f) for ci, j, forms in cands if ci == i for f in forms if f in found),
                   key=lambda x: x[0], default=None)
        if best:
            out.append((i, best[0], best[1]))
            covered = best[0]
    return out


PROVERB_MAX = 20  # 諺語最多幾個詞
_CONTENT = {"名詞", "動詞", "形容詞", "副詞", "形状詞", "代名詞"}


def find_proverbs(tokens: list[analysis.Token], dictionary: Dictionary) -> list[tuple[int, int, str]]:
    """長的諺語、慣用句 (起點, 終點, 字典寫法)：原文照字面和字典的寫法相同（不受 MAX_EXPR 限制），
    或是只選到前半句（敵を知り己を知れば → 敵を知り己を知れば百戦危うからず）。前半句要同時符合：
    在原文結尾或標點前結束（被選取範圍、引號截斷）、至少 5 個字而且是整句的一半以上、至少兩個實詞；
    tools/proverb_eval.py 量誤配。由左而右，同一起點取最長的，不重疊。"""
    import bisect
    keys = dictionary.proverbs()
    if not keys:
        return []
    starts = [0]
    for t in tokens:
        starts.append(starts[-1] + len(t.surface))
    text = "".join(t.surface for t in tokens)
    out, covered = [], 0
    for i, t in enumerate(tokens):
        if i < covered or t.pos1 not in _EXPR_START:
            continue
        best = None
        for j in range(i + 2, min(len(tokens), i + PROVERB_MAX) + 1):
            if tokens[j - 1].is_punct():
                break
            span = text[starts[i]:starts[j]]
            k = bisect.bisect_left(keys, span)
            if k == len(keys) or not keys[k].startswith(span):
                break  # 再長也不會是任何一條的開頭
            if keys[k] == span:
                best = (j, span)
                continue
            cut = j == len(tokens) or tokens[j].is_punct()
            content = sum(tokens[m].pos1 in _CONTENT for m in range(i, j))
            full = min((key for key in keys[k:k + 50] if key.startswith(span)), key=len)
            if cut and content >= 2 and len(span) >= max(5, len(full) / 2):
                best = (j, full)
        if best:
            out.append((i, best[0], best[1]))
            covered = best[0]
    return out


# ---------------------------------------------------------------- 擷取

def _sentence_at(text: str, pos: int) -> str:
    """text 中包含第 pos 個字的那一句。"""
    start = 0
    for s in _SENT_END.split(text):
        if start <= pos < start + len(s):
            return s.strip()
        start += len(s)
    return text.strip()


def _expression(form: str, text: str, pos_in_text: int, dictionary: Dictionary, compound: bool) -> WordInfo:
    lists = _jlpt_lists()[0].get(form, [])
    reading = analysis.to_hira(dictionary.reading_of(form)) or analysis.lemma_reading(form)
    if analysis._KATAKANA_ONLY.match(form):
        reading = form
    # 慣用語優先用「慣用語」詞性的條目（そうする 是「這樣做」，不是同音的「奏する（上奏）」）
    m = dictionary.lookup([form], reading, None if compound else [{"phrase", "any"}, None])
    return WordInfo(word=form, reading=reading, pos=COMPOUND_POS if compound else EXPR_POS,
                    level=lists[0].level if lists else None,
                    proper=False, sentence=_sentence_at(text, pos_in_text), zh=m.zh, en=m.en, rare_zh=m.rare_zh, rare_en=m.rare_en,
                    jlpt_en=lists[0].meaning if lists else "")


def _grammar_points(tokens: list[analysis.Token], hits) -> list:
    """要顯示的文法句型。同一段字比對到兩條同級的規則時（「ので」也比對到「〜のだ／〜んです」），
    只留名稱裡就有這段字的那條。"""
    out = []
    for rule, start, end in hits:
        same = [h for h in hits if (h[1], h[2]) == (start, end)]
        surface = "".join(t.surface for t in tokens[start:end])
        named = [h for h in same if surface in h[0].pattern]
        if len(same) == 1 or not named or (rule, start, end) in named:
            out.append((rule, start, end))
    return out


@lru_cache(maxsize=1)
def _grammar_notes() -> dict:
    """規則編號 → JLPT_grammer 的說明 [{pattern, level, meaning, formation, example}]（tools/grammar_notes.py 產生）。"""
    import json
    from pathlib import Path
    path = Path(grammar.__file__).with_name("grammar_notes.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _grammar_en() -> dict:
    """文法說明的英文版 {"rules": {規則編號: {meaning, formation[, pattern]}}, "jlpt": {key: {meaning, formation, example[, pattern]}}}
    （tools/grammar_en_batches.py 產生；還沒翻好的條目沒有，照舊顯示中文）。"""
    import json
    from pathlib import Path
    path = Path(grammar.__file__).with_name("grammar_en.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _grammar_info(rule, sentence: str) -> WordInfo:
    notes = _grammar_notes().get(rule.id, [])
    en = _grammar_en().get("rules", {}).get(rule.id) if lang.english() else None
    if en is None:
        return WordInfo(word=rule.pattern, reading="", pos=GRAMMAR_POS, level=rule.level, proper=False,
                        sentence=sentence, zh=[rule.meaning], formation=rule.formation, notes=notes)
    jlpt = _grammar_en().get("jlpt", {})
    notes = [n | jlpt[n["key"]] if n.get("key") in jlpt else n for n in notes]
    return WordInfo(word=en.get("pattern", rule.pattern), reading="", pos=GRAMMAR_POS, level=rule.level, proper=False,
                    sentence=sentence, en=[en["meaning"]], formation=en["formation"], notes=notes)


# ---------------------------------------------------------------- 振假名（ruby.py 讀出來的讀音提示）

RUBY_NOTE = "（字典沒有這個詞；讀音是書上標的振假名）"
RUBY_NOTE_EN = "(not in the dictionary; the reading is the furigana printed in the text)"
_KANA_TAIL = re.compile(r"^[ぁ-ゟ]*$")


def _ruby_words(tokens: list[analysis.Token], text: str, ruby: list[tuple[str, str]],
                dictionary: Dictionary) -> dict[int, tuple[int, WordInfo]]:
    """書上標了振假名的詞：{第一個詞的位置: (結束位置, WordInfo)}。長的提示先用，用過的詞不再用。
    - 振假名標的漢字剛好是幾個完整的詞（小鳥|遊 → 小鳥遊）：整個當成一個詞，讀音用振假名
    - 標在一個詞的開頭（喋っ 的 喋、美味しい 的 美味）：讀音＝振假名＋辭書形後面的假名（しゃべる、おいしい）
    字典有這個寫法＋讀音（或日檢單字表有）才當一般的詞；字典沒有的，只有 Sudachi 也不認識（拆成好幾個詞、
    專有名詞、未知詞）時才當成專有名詞（人名、地名），否則不用（振假名可能讀錯：慮 おもんばか → おもんぱか）。
    讀音和 Sudachi 一樣的不用處理。"""
    starts = [0]
    for t in tokens:
        starts.append(starts[-1] + len(t.surface))
    at = {s: i for i, s in enumerate(starts)}
    out, used = {}, set()
    for base, kana in sorted(ruby, key=lambda r: -len(r[0])):
        p = text.find(base)
        while p >= 0:
            i, end = at.get(p), p + len(base)
            p = text.find(base, p + 1)
            if i is None:
                continue
            j = next((k for k in range(i + 1, len(starts)) if starts[k] >= end), None)
            if j is None or used & set(range(i, j)):
                continue
            t = tokens[i]
            if starts[j] == end:  # 剛好是幾個完整的詞
                word, reading = base, kana
                if j - i == 1 and analysis.to_hira(t.reading) == kana:
                    continue
            elif j - i == 1 and t.lemma.startswith(base) and _KANA_TAIL.match(t.lemma[len(base):]):
                word, reading = t.lemma, kana + t.lemma[len(base):]
                if analysis.to_hira(_ref_reading(t)) == reading:
                    continue
            else:
                continue
            lists = [e for e in _jlpt_lists()[0].get(word, []) if _entry_reading(e) == reading]
            known = bool(lists) or dictionary.has_reading(word, reading)
            unknown_to_sudachi = j - i > 1 or t.pos[1] == "固有名詞" or t.oov
            if not known and not unknown_to_sudachi:
                continue
            sentence = _sentence_at(text, starts[i])
            if known:
                cats = token_cats(t.pos) if j - i == 1 else None
                m = dictionary.lookup([word], reading, cats)
                proper = j - i == 1 and t.pos[1] == "固有名詞" and not lists
                info = WordInfo(word=word, reading=reading, pos=analysis.pos_label(t) if j - i == 1 else "名詞",
                                level=lists[0].level if lists else None, proper=proper, sentence=sentence,
                                zh=m.zh, en=m.en, rare_zh=m.rare_zh, rare_en=m.rare_en,
                                jlpt_en=lists[0].meaning if lists else "",
                                also=dictionary.other_readings(word, reading))
            else:
                info = WordInfo(word=word, reading=reading, pos="名詞（專有名詞？）", level=None, proper=True,
                                sentence=sentence,
                                **({"en": [RUBY_NOTE_EN]} if lang.english() else {"zh": [RUBY_NOTE]}))
            out[i] = (j, info)
            used.update(range(i, j))
    return out


# 清單是空的時候照樣列出的詞（平常因為太基本、是代名詞而略過）：「何をする」不會什麼都沒有
_FALLBACK_POS = {"名詞", "代名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞"}


def _word_info(t: analysis.Token, text: str, pos_in_text: int, dictionary: Dictionary):
    """一個詞的單字資訊；回傳 (去重用的鍵, WordInfo)。"""
    norm = _norm(t, dictionary)
    entry = lookup_jlpt(t, norm, dictionary)
    proper = t.pos[1] == "固有名詞"
    word = entry.word if entry else _base_form(t)
    if entry and entry.word not in (t.lemma, norm, _base_form(t)):
        word = _base_form(t)  # 透過假名配到表上的「できる」：照原文寫「出来る」（查字典也用漢字，免得配到別的詞）
    key = entry.word if entry else (norm or word)
    if analysis._KATAKANA_ONLY.match(word):
        reading = word  # 外來語：讀音就是本身
    elif entry:
        reading = _entry_reading(entry)
    else:
        reading = _ref_reading(t)
    m = dictionary.lookup([word, key, t.lemma, norm], reading, token_cats(t.pos))
    return (key, reading), WordInfo(
        word=word, reading=reading, pos=analysis.pos_label(t),
        level=None if proper or not entry else entry.level, proper=proper,
        sentence=_sentence_at(text, pos_in_text), zh=m.zh, en=m.en, rare_zh=m.rare_zh, rare_en=m.rare_en,
        jlpt_en=entry.meaning if entry else "",
        also=[] if proper else dictionary.other_readings(word, reading))


def _head_words(head: str, surface: str, text: str, pos_in_text: int, dictionary: Dictionary) -> list:
    """複合詞的前半（練り直す 的「練り」）是一個動詞時，列出那個動詞（練る）。
    後面接「ます」再斷詞：單獨的「塗り」「読み」Sudachi 會斷成名詞。"""
    if not surface.startswith(head) and surface[len(head) - 1:len(head)] == head[-1]:
        head = surface[:len(head)]  # 前半取自標準寫法（遣り直す）時改用原文的寫法（やり）
    toks = analysis.tokenize(head + "ます")
    if len(toks) != 2 or toks[0].surface != head or toks[0].pos1 != "動詞" or not analysis._candidate(toks, 0, set()):
        return []
    return [_word_info(toks[0], text, pos_in_text, dictionary)]


def extract(texts: list[str], dictionary: Dictionary, ruby: list[tuple[str, str]] | None = None) -> list[WordInfo]:
    """依出現順序列出單字，同一個字只列一次；後面接著句子裡的文法句型（pos＝GRAMMAR_POS，同一條只列一次）。
    一個單字都沒有時（「何をする」：何 是代名詞、する 太基本），改列這些平常略過的詞。
    ruby：書上標的振假名 [(漢字, 讀音)]（ruby.py），對得上的詞用它當讀音（_ruby_words）。"""
    out, seen = [], set()
    points, seen_rules = [], set()
    skipped = []  # (詞, 句子, 位置)：清單是空的時候才用
    for text in texts:
        tokens = analysis.tokenize(text)
        starts = [0]
        for t in tokens:
            starts.append(starts[-1] + len(t.surface))
        hits = grammar.match_line_spans(tokens)
        for rule, start, _ in _grammar_points(tokens, hits):
            if rule.id not in seen_rules:
                seen_rules.add(rule.id)
                points.append(_grammar_info(rule, _sentence_at(text, starts[start])))
        in_grammar, heads = set(), {}
        for rule, start, end in hits:
            head = grammar.rule_compound_head(rule, tokens[start]) if end - start == 1 else None
            if head:
                heads[start] = head  # 複合詞（練り直す、読み方）本身照樣列，另外列前半的動詞（練る）
            elif end - start > 1 or (tokens[start].pos1 in ("名詞", "動詞", "形容詞")
                                     and not grammar.rule_form_only(rule, tokens[start])):  # 命令形 的 突っ走れ 照樣列
                in_grammar.update(range(start, end))  # 已屬於文法句型的詞（にとって 的「取る」）不列
        exprs = {i: (j, form) for i, j, form in find_expressions(tokens, dictionary)}
        # 諺語另外列一條，組成的詞照常列（敵、己 也是要學的字）
        provs = {i: form for i, _, form in find_proverbs(tokens, dictionary)}
        # 組成慣用說法的詞單獨列出時意思常會誤導（ため息をつく 的 つく ≠ 到達、気になる 的 気），不另外列；
        # 只保留慣用語裡兩個字以上的名詞（ため息、仕方）。複合詞（ボードゲーム）的組成部分都不列。
        in_expr = set()
        for i, (j, _) in exprs.items():
            compound = all(tokens[k].pos1 in _COMPOUND_PARTS for k in range(i, j))
            in_expr.update(k for k in range(i, j)
                           if compound or tokens[k].pos1 != "名詞" or len(tokens[k].surface) < 2)
        marked = _ruby_words(tokens, text, ruby, dictionary) if ruby else {}
        in_ruby = {k for i, (j, _) in marked.items() for k in range(i, j)}
        offset = 0
        for i, t in enumerate(tokens):
            pos_in_text, offset = offset, offset + len(t.surface)
            if i in marked:
                info = marked[i][1]
                if (info.word, info.reading) not in seen:
                    seen.add((info.word, info.reading))
                    out.append(info)
            if i in provs and provs[i] not in seen:
                seen.add(provs[i])
                out.append(_expression(provs[i], text, pos_in_text, dictionary, False))
            if i in in_ruby:
                continue
            if i in exprs:
                j, form = exprs[i]
                if form not in seen:
                    seen.add(form)
                    compound = all(tokens[k].pos1 in _COMPOUND_PARTS for k in range(i, j))
                    out.append(_expression(form, text, pos_in_text, dictionary, compound))
            if i in in_grammar or i in in_expr:
                continue
            info = None
            if not analysis._candidate(tokens, i, set()):
                # 代名詞平常略過是因為太基本（これ、私、彼）：日檢 N5、N4 以外的（己、汝、貴様、俺、お前）照樣列
                if t.pos1 == "代名詞" and analysis._candidate(tokens, i, set(), pronouns=True):
                    entry = lookup_jlpt(t, _norm(t, dictionary), dictionary)
                    if entry is None or entry.level <= 3:
                        key, info = _word_info(t, text, pos_in_text, dictionary)
                if info is None:
                    if t.pos1 in _FALLBACK_POS and not t.is_punct():
                        skipped.append((t, text, pos_in_text))
                    continue
            if info is None:
                key, info = _word_info(t, text, pos_in_text, dictionary)
            if i in heads and not (info.zh or info.en or info.jlpt_en):
                pass  # 字典沒收的複合詞（塗り直す）不列，只列前半的動詞
            elif key not in seen:  # 同寫法不同讀音（辛い：からい／つらい）是不同的字，都列
                seen.add(key)
                out.append(info)
            if i in heads:
                for key, info in _head_words(heads[i], t.surface, text, pos_in_text, dictionary):
                    if key not in seen:
                        seen.add(key)
                        out.append(info)
    if not out:
        for t, text, pos_in_text in skipped:
            key, info = _word_info(t, text, pos_in_text, dictionary)
            if key not in seen:
                seen.add(key)
                out.append(info)
    return out + points


def visible(words: list[WordInfo], min_level: int, show_proper: bool) -> list[WordInfo]:
    """級數過濾：min_level=3 → 只顯示 N3、N2、N1 和級外；min_level=5 → 全部。"""
    return [w for w in words
            if (show_proper if w.proper else (w.level is None or w.level <= min_level))]
