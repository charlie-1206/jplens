"""查字典：data/dict.sqlite（由 tools/build_dict.py 建立）＋ dict_translate/zh.jsonl（Gemini 翻譯）。

中文意思的優先順序：
1. Gemini 翻譯：逐個 JMdict 意思翻成中文。日檢單字表的詞手動分批翻（tools/zh_batches.py → zh.jsonl），
   其他 JMdict 條目由背景程式呼叫 API 翻（tools/zh_api.py → zh_api.jsonl），同一條目以手動翻的為準。
   和 JMdict 的意思一一對應，所以依詞性挑出來的意思也有對應的中文。查字時才讀檔，匯入新的一批不用重建字典。
2. Tomoshi 開放資料：JMdict 幾乎所有條目逐個意思的繁體中文（建字典時存進 entries.zh，tools/build_dict.py）
3. 維基詞典（約 4 萬詞）
4. 都沒有就顯示 JMdict 的英文（約 22 萬詞）
同一個寫法常有多個條目（「いい」：形容詞／「言う」的連用形／「飯」），依序挑：
1. 寫法：先用顯示的寫法（漢字）查，查不到才用其他寫法；避免假名查到同音的別的詞（とき → 朱鷺）
2. 讀音和 Sudachi 讀音相符（「辛い」からい／つらい）
3. 詞性：主要詞性優先（な形容詞「結構」先用形容詞「不錯」，沒有才用名詞「構造」）
"""

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .jlearn import analysis

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "dict.sqlite"
ZH_PATH = Path(__file__).resolve().parents[1] / "dict_translate" / "zh.jsonl"
ZH_API_PATH = Path(__file__).resolve().parents[1] / "dict_translate" / "zh_api.jsonl"
MAX_SENSES = 4
MAX_ALSO = 2  # 「也讀」最多列幾個讀音
# 只當詞綴、量詞用的條目（辛い づらい、日 か、中 じゅう）不列在「也讀」
_AFFIX_POS = ("suffix", "prefix", "counter")
# 冷僻的詞義（JMdict 的 misc 標籤）：排到後面，結果視窗的清單只在沒有其他意思時才顯示（滑鼠提示照樣全列）。
# 泥 →「安卓系統」（網路用語）、古語、專有名詞…；口語、敬語、俗語、慣用語這類「語體」標籤不算冷僻，
# 歷史用語也不算（侍、殿様 在遊戲、動畫很常見）
RARE_MISC = {"archaic", "obsolete term", "rare term", "dated term", "Internet slang",
             "manga slang", "organization name", "company name", "product name", "service",
             "work of art, literature, music, etc. name", "character", "family or surname", "place name",
             "given name or forename, gender not specified", "full name of a particular person", "ship name",
             "event", "deity", "fiction", "legend", "mythology", "creature", "object", "document", "group",
             "religion", "unclassified name"}
# 第二個以後的意思附學名的（莫大 →「胖大海（Scaphium affine）」）：動植物名，一般文章幾乎用不到；
# 第一個意思就是動植物的（麻）照常
_BINOMIAL = re.compile(r"\([A-Z][a-z]+ [a-z]+\)")

# 維基詞典的詞性標題（簡繁混雜）→ 類別
_WIKT_CAT = {
    "名詞": "noun", "名词": "noun", "專有名詞": "proper", "专有名词": "proper",
    "動詞": "verb", "动词": "verb", "形容詞": "adj", "形容词": "adj", "形容動詞": "adjna", "形容动词": "adjna",
    "副詞": "adv", "副词": "adv", "連體詞": "rentai", "连体词": "rentai", "連体詞": "rentai",
    "感嘆詞": "intj", "感叹词": "intj", "代詞": "pron", "代词": "pron", "接續詞": "conj", "连词": "conj",
    "連詞": "conj", "數詞": "num", "量詞": "counter",
    "短語": "phrase", "短语": "phrase", "慣用語": "phrase", "惯用语": "phrase", "成語": "phrase", "成语": "phrase",
    "漢字": "kanji",   # 漢字條目：只用在單一漢字的詞（窓、扉），而且找不到相容條目時才用
    "unknown": "any",  # 沒標詞性的條目：找不到相容條目時才用
}


def jm_cat(pos: str) -> set[str]:
    """JMdict 詞性（可能多個，以 | 分隔）→ 類別。"""
    cats = set()
    for p in pos.split("|"):
        p = p.lower()
        if "adverb" in p:
            cats.add("adv")
        elif "verb" in p and "prenominally" not in p:
            cats.add("verb")
        elif "keiyoushi" in p or "yoi/ii" in p:
            cats.add("adj")
        elif "keiyodoshi" in p or "'taru' adjective" in p or "'no'" in p:
            cats.add("adjna" if "'no'" not in p else "noun")
        elif "pre-noun" in p or "prenominally" in p:
            cats.add("rentai")
        elif "pronoun" in p:
            cats.add("pron")
        elif "noun" in p:
            cats.add("noun")
        elif "interjection" in p:
            cats.add("intj")
        elif "conjunction" in p:
            cats.add("conj")
        elif "expression" in p:
            cats.add("any")
    return cats or {"any"}


def token_cats(pos: tuple) -> list[set[str]] | None:
    """Sudachi 詞性 → 可以接受的字典詞性類別，依優先順序分層（前一層找不到才用下一層）；None = 不篩選。"""
    p0 = pos[0] if pos else ""
    p1 = pos[1] if len(pos) > 1 else ""
    p2 = pos[2] if len(pos) > 2 else ""
    if p0 == "動詞":
        return [{"verb"}]
    if p0 == "形容詞":
        return [{"adj"}]
    if p0 == "形状詞":
        return [{"adjna", "adj"}, {"noun", "rentai"}]  # そんな／こんな 常被標成形状詞，字典是連體詞
    if p0 == "名詞":
        if p1 == "固有名詞":
            return [{"proper"}, {"noun"}]
        if p2 == "形状詞可能":
            return [{"noun", "adjna", "adj"}]
        if p2 == "サ変可能":
            return [{"noun", "verb"}]
        if p2 == "副詞可能":
            return [{"noun", "adv"}]
        return [{"noun", "pron"}]
    if p0 == "副詞":
        return [{"adv"}, {"noun"}]  # 「あと少し」的あと：字典常只收名詞「後」
    if p0 == "連体詞":
        return [{"rentai"}, {"adj", "pron"}]
    if p0 == "感動詞":
        return [{"intj"}]
    if p0 == "代名詞":
        return [{"pron"}, {"noun"}]
    if p0 == "接続詞":
        return [{"conj"}, {"adv"}]
    return None


@dataclass
class Meaning:
    zh: list[str] = field(default_factory=list)  # 維基詞典（繁中）
    en: list[str] = field(default_factory=list)  # JMdict
    rare_zh: int = 0  # zh 最後幾個是冷僻的詞義（RARE_MISC）
    rare_en: int = 0


def _rare_senses(senses: list[str], misc) -> set[int]:
    misc = misc or []
    return {i for i, gl in enumerate(senses)
            if (i < len(misc) and RARE_MISC & set(misc[i])) or (i > 0 and _BINOMIAL.search(gl))}


class Dictionary:
    def __init__(self, path: Path = DB_PATH, zh_path: Path = ZH_PATH, zh_api_path: Path = ZH_API_PATH):
        self.ok = path.exists()
        self._path = path
        self._local = threading.local()  # sqlite 連線不能跨執行緒共用
        self._zh_paths = [zh_path, zh_api_path]  # 前面的優先（手動翻的日檢詞 → API 翻的其他詞）
        self._zh_mtime = None
        self._same_cache: dict[tuple[str, str], bool] = {}
        self._entry_cache: dict[tuple[str, str, str], bool] = {}
        self._zh_seq: dict[str, list[str]] = {}   # JMdict 條目編號 → 各意思的中文
        self._zh_word: dict[str, list[str]] = {}  # JMdict 找不到的詞（編號 J…）：寫法 → 中文

    def _gemini(self):
        """讀 Gemini 翻譯；檔案更新過就重讀（匯入新的一批、背景翻好一批後不用重開程式）。"""
        mtime = []
        for p in self._zh_paths:
            try:
                mtime.append(p.stat().st_mtime)
            except OSError:
                mtime.append(None)
        if mtime == self._zh_mtime:
            return
        seq, word = {}, {}
        for p in reversed(self._zh_paths):  # 優先的檔案最後讀，覆蓋同編號的翻譯
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    r = json.loads(line) if line.strip() else None
                except ValueError:  # 背景程式正寫到一半
                    r = None
                if not r:
                    continue
                if r["id"].startswith("J"):
                    word[r["word"]] = r["zh"]
                else:
                    seq[r["id"]] = r["zh"]
        self._zh_seq, self._zh_word, self._zh_mtime = seq, word, mtime

    def _db(self):
        if not hasattr(self._local, "db"):
            self._local.db = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        return self._local.db

    def attribution(self) -> str:
        if not self.ok:
            return ""
        row = self._db().execute("SELECT v FROM meta WHERE k='attribution'").fetchone()
        return row[0] if row else ""

    def _entries(self, keys):
        keys = [k for k in dict.fromkeys(keys) if k]
        if not keys:
            return []
        misc = "e.sense_misc" if self._has_column("sense_misc") else "NULL"
        q = (f"SELECT DISTINCT e.id, e.src, e.pos, e.readings, e.senses, e.sense_pos, e.seq, e.zh, {misc} FROM keys k "
             "JOIN entries e ON e.id = k.entry_id "
             f"WHERE k.key IN ({','.join('?' * len(keys))}) ORDER BY e.id")
        return [(src, pos, json.loads(rd), json.loads(ss), json.loads(sp), seq, json.loads(zh) if zh else [],
                 json.loads(mc) if mc else None)
                for _, src, pos, rd, ss, sp, seq, zh, mc in self._db().execute(q, keys)]

    def same_word(self, a: str, b: str) -> bool:
        """a、b 是不是 JMdict 同一個條目的寫法（いう／言う、だいぶ／大分）。a 字典裡沒有就當成是（沒有反證）。"""
        if not self.ok:
            return True
        key = (a, b)
        if key not in self._same_cache:
            q = ("SELECT k.key, e.seq FROM keys k JOIN entries e ON e.id = k.entry_id "
                 "WHERE k.key IN (?, ?) AND e.src = 'jmdict'")
            seqs = {a: set(), b: set()}
            for k, seq in self._db().execute(q, (a, b)):
                seqs[k].add(seq)
            self._same_cache[key] = not seqs[a] or bool(seqs[a] & seqs[b])
        return self._same_cache[key]

    def existing_phrases(self, keys) -> set[str]:
        """慣用說法用的 existing：純假名的寫法容易配到同音的別的詞（こうする → 攻する、そうする → 奏する），
        只接受「字典的標題就是這樣寫」或詞性是慣用語／副詞等不會被誤配的條目。"""
        keys = list(dict.fromkeys(k for k in keys if k))
        if not self.ok or not keys:
            return set()
        ok_pos = ("expression", "adverb", "conjunction", "interjection", "pre-noun", "短語", "惯用语", "慣用語")
        found = set()
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = ("SELECT k.key, e.word, e.pos FROM keys k JOIN entries e ON e.id = k.entry_id "
                 f"WHERE k.key IN ({','.join('?' * len(part))})")
            for key, word, pos in self._db().execute(q, part):
                if not analysis._KANA_ONLY.match(key) or word == key or any(p in pos for p in ok_pos):
                    found.add(key)
        return found

    def has_reading(self, word: str, reading: str) -> bool:
        """字典裡有沒有這個寫法、這個讀音的條目（振假名的讀音提示要確認是真的詞，見 words._ruby_words）。"""
        if not self.ok or not word or not reading:
            return False
        reading = analysis.to_hira(reading)
        return any(reading in [analysis.to_hira(r) for r in e[2]] for e in self._entries([word]))

    def reading_of(self, key: str) -> str:
        """字典記載的讀音（JMdict 優先）；慣用說法的讀音不能逐字拼（一日の長 是 いちじつのちょう）。"""
        entries = self._entries([key])
        for src in ("jmdict", "wikt"):
            for e in entries:
                if e[0] == src and e[2]:
                    return e[2][0]
        return ""

    def same_entry(self, word: str, a: str, b: str) -> bool:
        """a、b 是不是 JMdict 同一個條目（標題是 word）的兩個讀音，也就是同一個詞的不同念法（言う：いう／ゆう）。
        不同條目的讀音（辛い からい／つらい、汚す よごす／けがす）是不同的詞。"""
        if not self.ok or not word or not a or not b:
            return False
        key = (word, a, b)
        if key not in self._entry_cache:
            a, b = analysis.to_hira(a), analysis.to_hira(b)
            q = ("SELECT e.readings FROM keys k JOIN entries e ON e.id = k.entry_id "
                 "WHERE k.key = ? AND e.word = ? AND e.src = 'jmdict'")
            self._entry_cache[key] = any({a, b} <= {analysis.to_hira(r) for r in json.loads(rd)}
                                         for (rd,) in self._db().execute(q, (word, word)))
        return self._entry_cache[key]

    def other_readings(self, word: str, reading: str) -> list[tuple[str, str]]:
        """同一個寫法的其他常用讀音：[(讀音, 意思)…]，最多 MAX_ALSO 個。
        Sudachi 會選比較常見的讀音（「要」→ よう），但文中可能是另一個（かなめ：樞紐、關鍵）；
        只列 JMdict 標為常用（re_pri）的讀音，冷僻的（上手 うわて、生 ふ）不列。"""
        if not self.ok or not word or analysis._KANA_ONLY.match(word) or not self._has_column("common"):
            return []
        reading = analysis.to_hira(reading)
        misc = "e.sense_misc" if self._has_column("sense_misc") else "NULL"
        q = (f"SELECT e.readings, e.common, e.senses, e.sense_pos, e.seq, e.zh, {misc} FROM keys k "
             "JOIN entries e ON e.id = k.entry_id WHERE k.key = ? AND e.word = ? AND e.src = 'jmdict' "
             "AND e.common IS NOT NULL ORDER BY e.id")
        self._gemini()
        out = []
        for rd, common, senses, sense_pos, seq, zh, mc in self._db().execute(q, (word, word)):
            rd, common = json.loads(rd), json.loads(common)
            if reading in rd or reading in common or any(r in common for r, _ in out):
                continue
            if len(word) > 1 and seq not in self._zh_seq:
                continue  # 兩個字以上的詞只列日檢單字表也有的讀音（下手 的 したて 是相撲用語）；單字漢字（要 かなめ）不限
            idx = [i for i, p in enumerate(json.loads(sense_pos))
                   if not all(any(a in x.lower() for a in _AFFIX_POS) for x in p.split("|") if x)]
            if not idx:
                continue
            senses, zh = json.loads(senses), json.loads(zh) if zh else []
            rare = _rare_senses(senses, json.loads(mc) if mc else None)
            idx.sort(key=lambda i: i in rare)
            gemini = self._zh_seq.get(seq, [])
            mean = [gemini[i] if i < len(gemini) and gemini[i] else zh[i] if i < len(zh) and zh[i] else senses[i]
                    for i in idx[:2]]
            out.append((common[0], "；".join(mean)))
            if len(out) >= MAX_ALSO:
                break
        return out

    def _has_column(self, name: str) -> bool:
        """舊版字典沒有的欄位（重建字典前照樣能用，只是沒有新功能）。"""
        cols = getattr(self._local, "cols", None)
        if cols is None:
            cols = self._local.cols = {r[1] for r in self._db().execute("PRAGMA table_info(entries)")}
        return name in cols

    def lookup(self, keys: list[str], reading: str = "", tiers: list[set[str]] | None = None) -> Meaning:
        """keys：依優先順序的寫法；reading：平假名讀音；tiers：可接受的詞性類別（token_cats）。"""
        m = Meaning()
        if not self.ok:
            return m
        reading = analysis.to_hira(reading)
        keys = [k for k in dict.fromkeys(keys) if k]
        by_key = [(k, self._entries([k])) for k in keys]
        if not any(es for _, es in by_key) and reading:
            by_key = [(reading, self._entries([reading]))]  # 原文寫假名、字典只有漢字寫法時（ため息 → 溜め息）
        tiers = tiers or [None]

        def fits(rd):
            return not reading or not rd or reading in rd

        self._gemini()
        (m.en, m.rare_en), gemini, tomoshi = self._pick_jmdict(by_key, reading, tiers, self._zh_seq)
        if not gemini[0]:
            gemini = (next((self._zh_word[k] for k in keys if k in self._zh_word), [])[:MAX_SENSES], 0)
        if gemini[0] or tomoshi[0]:
            m.zh, m.rare_zh = gemini if gemini[0] else tomoshi
        else:
            m.zh = self._pick_wikt(by_key, fits, tiers)
        return m

    @staticmethod
    def _pick_wikt(by_key, fits, tiers) -> list[str]:
        """維基詞典：寫法順序 → 詞性分層，第一組找得到的條目；都沒有才用沒標詞性的條目或單字漢字條目。"""
        def senses(entries):
            out = []
            for e in entries:
                for s in e[3]:
                    s = s.rstrip("。")  # 有些解釋句尾帶句號，接起來會變成「明白。；知道」
                    if s and s not in out:
                        out.append(s)
            return out[:MAX_SENSES]

        # 顯示的寫法有漢字時，不用純假名的寫法查（とき → 齋、朱鷺 這類同音詞）
        has_kanji = bool(by_key) and not analysis._KANA_ONLY.match(by_key[0][0])
        cands = [(k, [(_WIKT_CAT.get(e[1], "other"), e) for e in es if e[0] == "wikt" and fits(e[2])])
                 for k, es in by_key if not (has_kanji and analysis._KANA_ONLY.match(k))]
        for cats in tiers:
            for _, ce in cands:
                chosen = [e for c, e in ce if cats is None or c in cats]
                if chosen:
                    return senses(chosen)
        for k, ce in cands:
            chosen = [e for c, e in ce if c == "any" or (c == "kanji" and len(k) == 1)]
            if chosen:
                return senses(chosen)
        return []

    @staticmethod
    def _pick_jmdict(by_key, reading, tiers, zh_seq) -> tuple:
        """JMdict：詞性分層 → 寫法順序 → 讀音相符、有中文翻譯（Gemini、Tomoshi）的條目優先，取詞性相容的詞義；
        都不相容才用第一個條目。冷僻的詞義排到後面。
        回傳 (英文, 對應的 Gemini 中文, 對應的 Tomoshi 中文)，每個都是 (意思, 最後幾個是冷僻的)。"""
        ordered = []
        for _, es in by_key:
            jm = [e for e in es if e[0] == "jmdict" and e not in ordered]
            jm.sort(key=lambda e: (not (reading and reading in e[2]), e[5] not in zh_seq, not e[6]))
            ordered += jm

        def result(e, idx):
            zh = zh_seq.get(e[5], [])
            rare = _rare_senses(e[3], e[7])
            idx = sorted(idx, key=lambda i: i in rare)[:MAX_SENSES]
            en = [i for i in idx]
            gm = [i for i in idx if i < len(zh)]
            tm = [i for i in idx if i < len(e[6]) and e[6][i]]
            return ([e[3][i] for i in en], sum(i in rare for i in en)),                 ([zh[i] for i in gm], sum(i in rare for i in gm)), ([e[6][i] for i in tm], sum(i in rare for i in tm))
        # 沒標詞性（any）的詞義只在最後才用：「どう」不該先配到沒標詞性的「仝（同上符號）」
        for cats in [*tiers, {"any"}]:
            for e in ordered:
                idx = [i for i, p in enumerate(e[4]) if cats is None or jm_cat(p) & cats]
                if idx:
                    return result(e, idx)
        return result(ordered[0], list(range(len(ordered[0][3])))) if ordered else (([], 0), ([], 0), ([], 0))
