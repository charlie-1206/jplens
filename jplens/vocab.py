"""單字本：使用者在結果視窗按「記錄」的單字，存在 vocab/saved_words.jsonl（一行一個 JSON）；
按「記錄原句」的句子存在 vocab/saved_sentences.jsonl（查完單字還是看不懂、想之後再查的句子）。

每筆保留查到的所有資訊（中英文意思、級數、出現的句子、時間），單字另有單字本視窗（manager.py）改的
marked（重點單字）、learned（已學會），舊的紀錄沒有這兩個欄位＝False。
需要時可以用 export_app_backup() 轉成使用者單字 App 的備份格式。

單字本視窗改紀錄時整個檔案重寫（先寫暫存檔再取代）；每次啟動後第一次重寫前，先把原檔複製成 .bak。
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT
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
    if pos.startswith("名詞"):  # 名詞（する動詞）→ n.；名詞／な形容詞 → adj.
        return "adj." if "形容" in pos else "n."
    if "動詞" in pos and "助動詞" not in pos:
        return "v."
    if "形容" in pos:
        return "adj."
    if "副詞" in pos:
        return "adv."
    if "名詞" in pos:
        return "n."
    return ""


class VocabStore:
    def __init__(self, path: Path = SAVED):
        self.listeners = []  # 紀錄改變時呼叫（單字本視窗用來重新整理）
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
            raise ValueError(f"單字本裡已經有「{new.get('word')}【{new.get('reading')}】」")
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
            definition = "；".join(r.get("meaning_zh") or []) or "; ".join(r.get("meaning_en") or [])
            words.append({
                "id": i,
                "term": r["word"],
                "type": _short_pos(r.get("pos", "")),
                "definition": definition,
                "comment": "",
                "isMarked": 1 if r.get("marked") else 0,
                "date": ms,
                "reading": "" if r.get("reading") == r["word"] else r.get("reading", ""),
                "example": one_line(r.get("sentence", "")),
                "jlpt": r.get("jlpt") if r.get("jlpt") in APP_LEVELS else "",
                "uid": uid_of(r),
            })
        iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        text = json.dumps({"version": 2, "date": iso, "words": words}, ensure_ascii=False, indent=2)
        out.write_bytes(text.encode("utf-8"))  # UTF-8 無 BOM、LF、檔尾無換行
        return len(words)
