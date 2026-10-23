"""單字本：使用者在結果視窗按「記錄」的單字，存在 vocab/saved_words.jsonl（一行一個 JSON）；
按「記錄原句」的句子存在 vocab/saved_sentences.jsonl（查完單字還是看不懂、想之後再查的句子）。

每筆保留查到的所有資訊（中英文意思、級數、出現的句子、時間），單字另有單字本視窗（manager.py）改的
marked（重點單字）、learned（已學會），舊的紀錄沒有這兩個欄位＝False。
需要時可以用 export_app_backup() 轉成使用者單字 App 的備份格式。

單字本視窗改紀錄時整個檔案重寫（先寫暫存檔再取代）；每次啟動後第一次重寫前，先把原檔複製成 .bak。
"""

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import lang
from .config import ROOT
from .i18n import t
from .words import WordInfo

VOCAB_DIR = ROOT / "vocab"
SAVED = VOCAB_DIR / "saved_words.jsonl"
SAVED_NAME = SAVED.name
SENTENCES = "saved_sentences.jsonl"  # 和單字本放在同一個資料夾
APP_LEVELS = ("N5", "N4", "N3", "N2", "N1")  # 匯出時 jlpt 只寫這幾種，其他（級外、專有名詞）寫空字串


def _key(word: str, reading: str) -> str:
    return f"{word}\t{reading}"


def key_of(rec: dict) -> str:
    return _key(rec.get("word", ""), rec.get("reading", ""))


def uid_of(rec: dict) -> str:
    """匯出給單字 App 的固定編號（同一個字永遠相同；App 匯入時用它判斷是不是同一個字）。"""
    return f"jplens:{rec.get('word', '')}|{rec.get('reading', '')}"


def one_line(text: str) -> str:
    """原句有好幾行（一次選到好幾段）時接成一行。"""
    return " ".join(ln.strip() for ln in (text or "").splitlines() if ln.strip())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # 手動編輯壞掉的行略過
    return out


def _short_pos(pos: str) -> str:
    """詞性轉成單字 App 使用的縮寫（n. / v. / adj. / adv.），其他詞性留空（同 video_translate 的 notes._short_pos）。"""
    if pos.startswith("名詞"):  # 名詞（する動詞）→ n.；名詞／な形容詞 → adj.  # i18n: 不翻
        return "adj." if "形容" in pos else "n."  # i18n: 不翻
    if "動詞" in pos and "助動詞" not in pos:  # i18n: 不翻
        return "v."
    if "形容" in pos:  # i18n: 不翻
        return "adj."
    if "副詞" in pos:  # i18n: 不翻
        return "adv."
    if "名詞" in pos:  # i18n: 不翻
        return "n."
    return ""


_POS_FROM_APP = {"n.": "名詞", "v.": "動詞", "adj.": "形容詞", "adv.": "副詞"}  # i18n: 不翻
_ALSO_MEANING = re.compile(r"^讀(.+?)的意思：(.*)$")  # i18n: 不翻
_HAS_CJK = re.compile(r"[㐀-鿿]")  # i18n: 不翻


def from_app_word(w: dict) -> dict | None:
    """單字 App 備份檔的一個字 → 單字本紀錄。本程式匯出時把也讀接在 reading（よう；かなめ）、
    definition（…；讀かなめ的意思：…）後面，這裡拆回 reading、also。沒有單字或解釋的回傳 None。"""
    term = str(w.get("term") or "").strip()
    definition = str(w.get("definition") or "").strip()
    if not term or not definition:
        return None
    readings = [x.strip() for x in re.split(r"[；;]", str(w.get("reading") or "")) if x.strip()]
    reading = readings[0] if readings else term
    also_mean, senses = {}, []
    for part in (x.strip() for x in re.split(r"[；;]", definition)):
        m = _ALSO_MEANING.match(part)
        if m:
            also_mean[m.group(1)] = m.group(2)
        elif part:
            senses.append(part)
    try:
        saved_at = datetime.fromtimestamp(int(w.get("date")) / 1000).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        saved_at = datetime.now().isoformat(timespec="seconds")
    zh = bool(_HAS_CJK.search(definition))
    jlpt = str(w.get("jlpt") or "").upper()
    return {
        "word": term, "reading": reading, "pos": _POS_FROM_APP.get(str(w.get("type") or "").strip(), ""),
        "jlpt": jlpt if jlpt in APP_LEVELS else "",
        "meaning_zh": senses if zh else [], "meaning_en": [] if zh else senses,
        "sentence": str(w.get("example") or ""), "also": [[rd, also_mean.get(rd, "")] for rd in readings[1:]],
        "comment": str(w.get("comment") or ""), "saved_at": saved_at,
        "marked": w.get("isMarked") in (1, True, "1"), "learned": False,
    }


def parse_import(text: str) -> list[dict]:
    """匯入檔的內容 → 紀錄列表。單字 App 備份（JSON，有 words）或完整備份（一行一個紀錄）。"""
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("words"), list):
        version = data.get("version") or 1
        if not isinstance(version, int) or version > 2:
            raise ValueError(t("備份檔的版本（{version}）比這個程式支援的新", version=version))
        return [r for r in (from_app_word(w) for w in data["words"] if isinstance(w, dict)) if r]
    out = []
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("word"):
            out.append(r)
    if not out:
        raise ValueError(t("看不懂這個檔案：要單字 App 的備份檔或單字本的完整備份（.jsonl）"))
    return out


class VocabStore:
    def __init__(self, path: Path = SAVED):
        self.listeners = []  # 紀錄改變時呼叫（單字本視窗用來重新整理）
        self.dictionary = None  # 舊紀錄沒有「也讀」時拿來補查（App 設定；沒有就不補）
        self._also_cache: dict[str, list] = {}
        self._backed_up: set[Path] = set()
        self.set_path(path)

    def set_path(self, path: Path):
        """換單字本資料夾（設定視窗）；path 是 saved_words.jsonl 的路徑。"""
        self.path = Path(path)
        self._keys = {key_of(r) for r in self.records()}
        self._changed()

    def _changed(self):
        for f in list(self.listeners):
            f()

    def records(self) -> list[dict]:
        return _read_jsonl(self.path)

    def has(self, w: WordInfo) -> bool:
        return _key(w.word, w.reading) in self._keys

    def add(self, w: WordInfo) -> bool:
        """記錄一個單字；已經記錄過就不重複寫入，回傳 False。"""
        if self.has(w):
            return False
        rec = {
            "word": w.word,
            "reading": w.reading,
            "pos": w.pos,
            "jlpt": w.jlpt,
            "meaning_zh": w.zh,
            "meaning_en": w.en or ([w.jlpt_en] if w.jlpt_en else []),
            "sentence": w.sentence,
            "also": [[rd, mean] for rd, mean in w.also],  # 同寫法的其他常用讀音（結果視窗的「也讀」）
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "marked": False,
            "learned": False,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._keys.add(_key(w.word, w.reading))
        self._changed()
        return True

    def also_of(self, rec: dict) -> list[tuple[str, str]]:
        """單字的「也讀」[(讀音, 意思)]。2026-10-21 以前記錄的沒有 also 欄位：用字典現查補上（不寫回檔案）。"""
        if "also" in rec:
            return [(rd, mean) for rd, mean in rec.get("also") or []]
        if self.dictionary is None or rec.get("jlpt") in ("專有名詞", "文法") or rec.get("pos") == "文法":  # i18n: 不翻
            return []
        key = key_of(rec)
        if key not in self._also_cache:
            try:
                self._also_cache[key] = self.dictionary.other_readings(rec.get("word", ""), rec.get("reading", ""))
            except Exception:
                self._also_cache[key] = []
        return self._also_cache[key]

    def reading_with_also(self, rec: dict) -> str:
        """讀音後面接也讀：「よう；かなめ」（單字本卡片、複製、匯出共用；紀錄裡的 reading 不變，查重複、uid 才對得上）。"""
        return "；".join([rec.get("reading", "")] + [rd for rd, _ in self.also_of(rec)])

    def also_meanings(self, rec: dict) -> list[str]:
        """也讀的意思接在原本的意思後面：「讀かなめ的意思：樞紐、關鍵」（裡面改用頓號，才不會和意思之間的「；」混在一起）。"""
        return [f"讀{rd}的意思：{mean.replace('；', '、')}" for rd, mean in self.also_of(rec)]  # i18n: 不翻

    def also_display(self, rec: dict) -> list[str]:
        """畫面上顯示、複製用的也讀（介面語言）；匯出和匯入解析用 also_meanings 的固定格式。"""
        return [t("讀{rd}的意思：{mean}", rd=rd, mean=mean.replace("；", "、")) for rd, mean in self.also_of(rec)]

    # ---------------------------------------------------------------- 單字本視窗：修改、刪除
    def _rewrite(self, path: Path, records: list[dict]):
        """整個檔案重寫：先寫暫存檔再取代，寫到一半當掉也不會留下半個檔案。
        這次啟動後第一次重寫前，先把原檔複製成 .bak（改壞了還能救）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path not in self._backed_up:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
            self._backed_up.add(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                       encoding="utf-8", newline="\n")
        os.replace(tmp, path)

    def _save_records(self, records: list[dict]):
        self._rewrite(self.path, records)
        self._keys = {key_of(r) for r in records}
        self._changed()

    def update(self, key: str, **fields) -> dict:
        """改一個單字（marked、learned、讀音、意思、原句…）。改了單字或讀音、又和別的紀錄重複時丟 ValueError。"""
        records = self.records()
        idx = next((i for i, r in enumerate(records) if key_of(r) == key), None)
        if idx is None:
            raise KeyError(key)
        new = {**records[idx], **fields}
        if key_of(new) != key and any(key_of(r) == key_of(new) for r in records):
            raise ValueError(t("單字本裡已經有「{word}【{reading}】」", word=new.get("word"), reading=new.get("reading")))
        records[idx] = new
        self._save_records(records)
        return new

    def delete(self, key: str):
        """刪除一個單字，回傳 (位置, 紀錄) 給 restore() 復原；找不到回傳 None。"""
        records = self.records()
        idx = next((i for i, r in enumerate(records) if key_of(r) == key), None)
        if idx is None:
            return None
        rec = records.pop(idx)
        self._save_records(records)
        return idx, rec

    def restore(self, idx: int, rec: dict):
        records = self.records()
        if any(key_of(r) == key_of(rec) for r in records):
            return
        records.insert(min(idx, len(records)), rec)
        self._save_records(records)

    def add_record(self, rec: dict) -> bool:
        """手動新增一個單字（單字本視窗的「新增單字」）；已經有了回傳 False。"""
        if key_of(rec) in self._keys:
            return False
        rec = {"pos": "", "jlpt": "", "meaning_zh": [], "meaning_en": [], "sentence": "", "also": [],
               "saved_at": datetime.now().isoformat(timespec="seconds"), "marked": False, "learned": False, **rec}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._keys.add(key_of(rec))
        self._changed()
        return True

    def clear(self) -> tuple[list[dict], Path | None]:
        """清空單字本：先把整個檔案另存成 saved_words.cleared-時間.jsonl（關掉程式後還找得回來），
        回傳 (清掉的紀錄, 備份檔) 給 restore_all() 復原。"""
        old = self.records()
        backup = None
        if old:
            backup = self.path.with_name(f"{self.path.stem}.cleared-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
            shutil.copy2(self.path, backup)
        self._save_records([])
        return old, backup

    def restore_all(self, recs: list[dict]):
        """清空後按「復原」：放回清掉的紀錄（清空後又新增的照樣保留，排在後面）。"""
        now = self.records()
        keys = {key_of(r) for r in recs}
        self._save_records(recs + [r for r in now if key_of(r) not in keys])

    # ---------------------------------------------------------------- 匯入
    def import_file(self, path: Path) -> dict:
        """匯入單字：單字 App 的備份檔（version 1、2，本程式匯出的也是這個格式）或本程式的完整備份（.jsonl）。
        單字本已經有的字（同寫法同讀音）不重複加入，只補空的欄位、把重點單字的標記合起來。
        回傳 {"added", "updated", "skipped"} 的數量。格式不對時丟 ValueError。"""
        text = Path(path).read_text(encoding="utf-8-sig")
        incoming = parse_import(text)
        records = self.records()
        index = {key_of(r): i for i, r in enumerate(records)}
        added = updated = skipped = 0
        seen = set()
        for rec in incoming:
            k = key_of(rec)
            if k in seen or not rec.get("word"):
                skipped += 1
                continue
            seen.add(k)
            if k not in index:
                index[k] = len(records)
                records.append(rec)
                added += 1
                continue
            old = records[index[k]]
            new = dict(old)
            for f in ("pos", "jlpt", "meaning_zh", "meaning_en", "sentence", "also", "comment"):
                if not new.get(f) and rec.get(f):
                    new[f] = rec[f]
            new["marked"] = bool(old.get("marked") or rec.get("marked"))
            new["learned"] = bool(old.get("learned") or rec.get("learned"))
            if new != old:
                records[index[k]] = new
                updated += 1
            else:
                skipped += 1
        if added or updated:
            self._save_records(records)
        return {"added": added, "updated": updated, "skipped": skipped}

    def export_jsonl(self, out: Path, records: list[dict] | None = None) -> int:
        """完整備份（所有欄位，含已學會、英文意思、也讀），之後可以用 import_file 匯回來。"""
        recs = self.records() if records is None else records
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8", newline="\n")
        return len(recs)

    # ---------------------------------------------------------------- 原句
    @property
    def sentence_path(self) -> Path:
        return self.path.parent / SENTENCES

    def sentences(self) -> list[dict]:
        return _read_jsonl(self.sentence_path)

    def add_sentence(self, text: str, words: list[str], translation: str = "") -> bool:
        """記錄原句（附上當時列出的單字、Gemini 的譯文）；同一句已經記錄過就回傳 False。"""
        if any(r.get("text") == text for r in self.sentences()):
            return False
        rec = {"text": text, "words": words, "saved_at": datetime.now().isoformat(timespec="seconds")}
        if translation:
            rec["translation"] = translation
        self.sentence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sentence_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._changed()
        return True

    def update_sentence(self, text: str, saved_at: str, **fields) -> bool:
        """改一句的欄位（單字本句子頁的講解：explanation、explain_model）。"""
        recs = self.sentences()
        for r in recs:
            if r.get("text") == text and r.get("saved_at") == saved_at:
                r.update(fields)
                self._rewrite(self.sentence_path, recs)
                self._changed()
                return True
        return False

    def delete_sentence(self, text: str, saved_at: str):
        """刪除一句（同一句只會記錄一次，saved_at 是保險），回傳 (位置, 紀錄) 給 restore_sentence()。"""
        recs = self.sentences()
        idx = next((i for i, r in enumerate(recs) if r.get("text") == text and r.get("saved_at") == saved_at), None)
        if idx is None:
            return None
        rec = recs.pop(idx)
        self._rewrite(self.sentence_path, recs)
        self._changed()
        return idx, rec

    def restore_sentence(self, idx: int, rec: dict):
        recs = self.sentences()
        if any(r.get("text") == rec.get("text") for r in recs):
            return
        recs.insert(min(idx, len(recs)), rec)
        self._rewrite(self.sentence_path, recs)
        self._changed()

    # ---------------------------------------------------------------- 匯出
    def export_sentences(self, out: Path, records: list[dict] | None = None) -> int:
        """記錄的原句匯出成純文字，依記錄順序編號（1. 2. 3.），不附其他查詢資料。
        records：只匯出這些（單字本視窗的篩選結果）；None＝全部。回傳句數。"""
        recs = self.sentences() if records is None else records
        texts = [t for t in (one_line(r.get("text", "")) for r in recs) if t]
        out.write_text("".join(f"{i}. {t}\n" for i, t in enumerate(texts, 1)), encoding="utf-8", newline="\n")
        return len(texts)

    def export_app_backup(self, out: Path, records: list[dict] | None = None) -> int:
        """轉成單字 App 的備份格式 version 2。
        v2 和單字 App 約定（2026-10-16）：多了 reading（讀音）、example（例句＝記錄時的原句）、jlpt（N5～N1，其他空字串）、
        uid（固定編號，App 匯入時用來認同一個字：已有的字只更新重點、補空的欄位）；comment 留給使用者自己的筆記
        （v1 時放讀音）；isMarked＝重點單字。records：只匯出這些；None＝全部。回傳單字數。"""
        words = []
        for i, r in enumerate(self.records() if records is None else records, 1):
            try:
                ms = int(datetime.fromisoformat(r["saved_at"]).timestamp() * 1000)
            except (KeyError, ValueError):
                ms = 0
            mean = lang.rec_meanings(r)[0]
            definition = ("；" if _HAS_CJK.search("".join(mean)) else "; ").join(mean)
            # 也讀：App 的格式沒有這個欄位（version 不能升），讀音接在 reading 後面、意思接在解釋後面；uid 不變（App 才認得是同一個字）
            definition = "；".join(([definition] if definition else []) + self.also_meanings(r))
            words.append({
                "id": i,
                "term": r["word"],
                "type": _short_pos(r.get("pos", "")),
                "definition": definition,
                "comment": r.get("comment", ""),  # 從單字 App 匯入的筆記照樣帶回去
                "isMarked": 1 if r.get("marked") else 0,
                "date": ms,
                "reading": "" if r.get("reading") == r["word"] else self.reading_with_also(r),
                "example": one_line(r.get("sentence", "")),
                "jlpt": r.get("jlpt") if r.get("jlpt") in APP_LEVELS else "",
                "uid": uid_of(r),
            })
        iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        text = json.dumps({"version": 2, "date": iso, "words": words}, ensure_ascii=False, indent=2)
        out.write_bytes(text.encode("utf-8"))  # UTF-8 無 BOM、LF、檔尾無換行
        return len(words)
