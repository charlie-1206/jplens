"""Gemini：整句翻譯（結果視窗）、句子講解（單字本的句子頁）、API Key 的存放。

- API Key 存在 Windows 的認證管理員（keyring，服務名稱 jplens），不寫進 config.json、不寫進日誌；
  沒存的話讀環境變數 GEMINI_API_KEY。
- 翻譯用 gemini-3.5-flash-lite（2026-10-17 使用者決定：這裡重視即時反應速度）；講解的模型可選（Config.explain_model）。
- 翻譯結果快取在 ROOT/cache/translations.jsonl（同一句不重翻、不重複花額度）；講解存在句子紀錄裡（vocab.update_sentence）。
- 呼叫都在背景執行緒（run_async），結果用 Qt 訊號送回主執行緒。
"""

import json
import logging
import os
import re
import threading
import unicodedata

from PySide6.QtCore import QObject, Signal

from .config import ROOT
from .i18n import t

log = logging.getLogger("jplens")

SERVICE = "jplens"
KEYS = {  # 種類 → (認證管理員裡的名稱, 備援的環境變數)
    "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
    "openai": ("openai_api_key", "OPENAI_API_KEY"),  # 自訂的 OpenAI 相容服務（Groq、OpenRouter、DeepSeek、Ollama…）
}
TRANSLATE_MODEL = "gemini-3.5-flash-lite"
EXPLAIN_MODELS = {  # 模型 ID → 設定視窗顯示的名稱
    "gemini-3.8-flash": t("Gemini 3.8 Flash（講解較好，免費額度較少）"),
    "gemini-3.5-flash-lite": t("Gemini 3.5 Flash-Lite（較快，免費額度較多）"),
}
CACHE = ROOT / "cache" / "translations.jsonl"
TIMEOUT_MS = 30_000

TRANSLATE_PROMPT = (
    "把下面的日文翻成自然、口語的繁體中文（台灣用語）。\n"  # i18n: 不翻
    "只輸出譯文：不要解釋、不要加引號、不要附原文。原文有幾行就輸出幾行，逐行對應。\n"
    "原文可能是遊戲台詞、動畫字幕、漫畫對話或小說，可能有文字辨識的錯字，照最可能的意思翻。\n"
    "\n"
    "{text}"
)

TRANSLATE_PROMPT_EN = """Translate the following Japanese into natural, conversational English.
Output only the translation: no explanations, no quotation marks, no original text. Output the same number of lines as the original, line by line.
The text may be game dialogue, anime subtitles, manga speech or a novel, and may contain OCR errors; translate the most likely meaning.

{text}"""

EXPLAIN_PROMPT = (
    "你是日文老師，學生的母語是繁體中文（台灣），程度約 N4～N2。請講解下面這句日文，用繁體中文回答，\n"  # i18n: 不翻
    "格式用簡單的 Markdown（小標題、條列），不要寒暄、不要重複題目：\n"
    "\n"
    "1. **翻譯**：自然的中文翻譯；如果直譯和意譯差很多，兩個都給。\n"
    "2. **拆解**：依序列出句子裡的詞和文法句型（寫法、讀音、意思、在句中的作用）。基本的助詞簡短帶過，重點放在學生可能不懂的地方。\n"
    "3. **為什麼這樣說**：語氣、省略的主語或受詞、敬語、口語或方言的說法、說話者的態度。\n"
    "4. 如果原文看起來有文字辨識的錯字，指出最可能的正確寫法。\n"
    "\n"
    "句子：\n"
    "{text}"
)

EXPLAIN_PROMPT_EN = """You are a Japanese teacher. The student is an English speaker at about JLPT N4–N2. Explain the following Japanese sentence in English,
using simple Markdown (short headings, bullet points). No greetings, don't repeat the question:

1. **Translation**: a natural English translation; if a literal and a free translation differ a lot, give both.
2. **Breakdown**: list the words and grammar patterns in order (spelling, reading, meaning, role in the sentence). Keep basic particles brief; focus on what the student may not know.
3. **Why it's said this way**: tone, omitted subject or object, keigo, colloquial or dialect forms, the speaker's attitude.
4. If the text looks like it has OCR errors, point out the most likely correct spelling.

Sentence:
{text}"""


class AIError(Exception):
    """給使用者看的錯誤訊息（沒有 Key、額度用完、網路問題）。"""


# ---------------------------------------------------------------- API Key
def _keyring():
    try:
        import keyring
        return keyring
    except Exception:  # noqa: BLE001（沒裝或後端壞掉：只能用環境變數）
        log.exception("keyring 無法使用")
        return None


def get_key(kind: str = "gemini") -> tuple[str, str]:
    """回傳 (Key, 來源)；來源是 "keyring"／"env"／""（沒有）。"""
    name, env = KEYS[kind]
    kr = _keyring()
    if kr is not None:
        try:
            k = kr.get_password(SERVICE, name)
            if k:
                return k, "keyring"
        except Exception:  # noqa: BLE001
            log.exception("讀取認證管理員失敗")
    k = os.environ.get(env, "").strip()
    return (k, "env") if k else ("", "")


def set_key(key: str, kind: str = "gemini"):
    kr = _keyring()
    if kr is None:
        raise AIError(t("這台電腦不能用 Windows 認證管理員存 Key，請改設環境變數 {env}。", env=KEYS[kind][1]))
    kr.set_password(SERVICE, KEYS[kind][0], key.strip())
    _clients.clear()


def clear_key(kind: str = "gemini"):
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE, KEYS[kind][0])
        except Exception:  # noqa: BLE001（本來就沒有）
            pass
    _clients.clear()


# 常見 API Key 的樣子（Gemini、OpenAI、Groq、xAI、OpenRouter、Hugging Face…）：寫進日誌、顯示錯誤前遮掉
_KEY_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|(?:sk|gsk|xai|pplx|nvapi|sk-or|sk-proj|sk-ant)[-_][0-9A-Za-z_\-]{10,}|hf_[0-9A-Za-z]{20,}|r8_[0-9A-Za-z]{20,})")


def redact(text) -> str:
    """把看起來像 API Key 的字串換成「…末四碼」（服務回傳的錯誤訊息有時會原樣帶回 Key）。"""
    return _KEY_RE.sub(lambda m: "…" + m.group(0)[-4:], str(text))


def looks_like_key(text: str) -> bool:
    return bool(_KEY_RE.search(text or ""))


def masked(key: str) -> str:
    return f"…{key[-4:]}" if len(key) > 8 else t("很短，可能貼錯了")


# ---------------------------------------------------------------- 呼叫
_clients: dict[str, object] = {}
_BUSY = t("Gemini 現在很忙，等一下再試。")
_lock = threading.Lock()


def _client():
    key, _ = get_key()
    if not key:
        raise AIError(t("還沒有設定 Gemini API Key：系統匣 → 設定 → 翻譯與 AI，貼上 Key。"))
    with _lock:
        if key not in _clients:
            from google import genai
            from google.genai import types
            _clients.clear()
            _clients[key] = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        return _clients[key]


def _generate(model: str, prompt: str, retries: int = 0) -> str:
    """retries：伺服器忙（503）時等幾秒再試幾次（講解用；翻譯要快，不重試）。"""
    import time
    for i in range(retries):
        try:
            return _generate(model, prompt)
        except AIError as e:
            if str(e) != _BUSY:
                raise
            time.sleep(3 * (i + 1))
    from google.genai import errors
    try:
        resp = _client().models.generate_content(model=model, contents=prompt)
    except AIError:
        raise
    except errors.APIError as e:
        code = getattr(e, "code", None)
        log.warning("Gemini %s 失敗：%s %s", model, code, redact((getattr(e, "message", "") or "")[:200]))
        if code in (401, 403) or (code == 400 and "API key" in str(e)):
            raise AIError(t("API Key 無效或沒有權限，請到設定 → 翻譯與 AI 重新貼上。")) from None
        if code == 429:
            raise AIError(t("{model} 的免費額度暫時用完了，等一下再試（或換別的模型）。", model=model)) from None
        if code in (500, 502, 503, 504):
            raise AIError(_BUSY) from None
        raise AIError(t("Gemini 回傳錯誤（{code}）。", code=code)) from None
    except Exception as e:  # noqa: BLE001（網路斷線、逾時）
        log.warning("Gemini %s 連線失敗：%s", model, redact(e))
        raise AIError(t("連不上 Gemini（網路有問題或逾時）。")) from None
    text = (resp.text or "").strip()
    if not text:
        raise AIError(t("Gemini 沒有回傳內容（可能被安全過濾擋下），換個說法再試。"))
    return text


_cache: dict[tuple[str, str], str] | None = None  # (語言, cache_key) → 譯文
_QUOTES = "「」『』（）()【】〔〕［］[]〈〉《》\"'“”‘’"
_ELLIPSIS = re.compile(r"…+|\.{2,}|・{2,}|‥+")  # i18n: 不翻


def cache_key(text: str) -> str:
    """快取用的鍵：只差在頭尾的引號括號（包括只有半邊的）、空白、全半形、刪節號寫法、句尾的「。、…」的，算同一句。
    「？」保留（疑問句的翻譯不一樣）；差一兩個字的不算同一句（行ける／行けない 意思相反）。"""
    t = unicodedata.normalize("NFKC", text)
    t = "".join(t.split())  # 空白、換行
    t = _ELLIPSIS.sub("…", t)
    prev = None
    while prev != t:
        prev = t
        t = t.strip(_QUOTES).rstrip("。、,.…")
    return t or text


def text_length(text: str) -> int:
    """原文有幾個字（只算文字、數字，不算標點、空白）：背景預先翻譯的最少字數用。"""
    return sum(1 for ch in unicodedata.normalize("NFKC", text) if unicodedata.category(ch)[0] in "LN")


def _load_cache() -> dict[tuple[str, str], str]:
    """快取檔一行一句：{"text", "zh"（譯文；欄位名稱是舊的，英文譯文也放這裡）, "model", "lang"（沒有＝繁體中文）}。"""
    global _cache
    if _cache is None:
        _cache = {}
        try:
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    _cache[(r.get("lang", "zh-TW"), cache_key(r["text"]))] = r["zh"]  # 舊的快取也用同樣的規則整理
                except (ValueError, KeyError):
                    continue
        except FileNotFoundError:
            pass
    return _cache


def cached_translation(text: str, lang: str = "zh-TW") -> str | None:
    with _lock:
        return _load_cache().get((lang, cache_key(text)))


def translate_lang(cfg) -> str:
    """翻成哪種語言（Config.explain_lang）：English 或繁體中文（快取分開存）。"""
    return "en" if getattr(cfg, "explain_lang", "") == "en" else "zh-TW"


def openai_chat(base_url: str, model: str, prompt: str) -> str:
    """OpenAI 相容的 /chat/completions（Groq、OpenRouter、DeepSeek、OpenAI、本機的 Ollama／LM Studio…）。
    只用標準函式庫（urllib），不用另外裝套件。"""
    import time
    import urllib.error
    import urllib.request
    if not base_url or not model:
        raise AIError(t("自訂翻譯服務還沒設定好：設定 → 翻譯與 AI，填網址和模型名稱。"))
    url = base_url.strip().rstrip("/")
    from urllib.parse import urlparse
    if not urlparse(url).path:  # 只填了主機（http://localhost:11434、https://api.openai.com）：OpenAI 相容的路徑在 /v1 底下
        url += "/v1"
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    key, _ = get_key("openai")
    # 一定要有 User-Agent：放在 Cloudflare 後面的服務（Groq 等）會擋 Python 預設的「Python-urllib」
    headers = {"Content-Type": "application/json", "User-Agent": "jplens/0.1 (+https://github.com/charlie-1206/jplens)"}
    if key:  # 本機服務（Ollama）可以不用 Key
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.2}).encode("utf-8")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=TIMEOUT_MS / 1000) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = redact(e.read().decode("utf-8", "replace")[:200])
        log.warning("自訂翻譯服務 %s 失敗：%s %s", model, e.code, detail)
        if e.code == 401:
            raise AIError(t("自訂翻譯服務的 Key 無效。")) from None
        if e.code == 403:
            raise AIError(t("自訂翻譯服務拒絕存取（403）：Key 沒有權限，或被服務的防護擋下。{detail}", detail=detail[:80])) from None
        if e.code == 404:
            raise AIError(t("自訂翻譯服務回 404：網址或模型名稱可能打錯。")) from None
        if e.code == 429:
            raise AIError(t("自訂翻譯服務的額度暫時用完了，等一下再試。")) from None
        raise AIError(t("自訂翻譯服務回傳錯誤（{code}）。", code=e.code)) from None
    except Exception as e:  # noqa: BLE001（網址錯、連不上、逾時）
        log.warning("自訂翻譯服務連線失敗：%s", redact(e))
        raise AIError(t("連不上自訂翻譯服務（網址錯誤、網路有問題或逾時）。")) from None
    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        raise AIError(t("自訂翻譯服務回傳的格式看不懂（不是 OpenAI 相容格式？）。")) from None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # 會輸出思考過程的模型（DeepSeek-R1、Qwen3）
    if not text:
        raise AIError(t("自訂翻譯服務沒有回傳內容。"))
    log.info("自訂翻譯服務 %s：%.2f 秒", model, time.perf_counter() - t0)
    return text


def translate(text: str, cfg=None, force: bool = False) -> str:
    """整句翻譯（同一句只翻一次）。在背景執行緒呼叫。cfg 決定翻譯來源（Config.translate_provider），沒給就用 Gemini。
    force：不看快取、重新翻譯（結果視窗的「重新翻譯」），新的譯文取代快取裡舊的（檔案裡後面的行蓋前面的）。
    翻成哪種語言看 cfg.explain_lang（translate_lang）。"""
    lang = translate_lang(cfg)
    hit = None if force else cached_translation(text, lang)
    if hit is not None:
        return hit
    prompt = (TRANSLATE_PROMPT_EN if lang == "en" else TRANSLATE_PROMPT).format(text=text)
    if cfg is not None and cfg.translate_provider == "openai":
        model = cfg.openai_model
        zh = openai_chat(cfg.openai_base_url, model, prompt)
    else:
        model = TRANSLATE_MODEL
        zh = _generate(model, prompt)
    with _lock:
        _load_cache()[(lang, cache_key(text))] = zh
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE, "a", encoding="utf-8", newline="\n") as f:
                row = {"text": text, "zh": zh, "model": model} | ({"lang": lang} if lang != "zh-TW" else {})
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("寫入翻譯快取失敗")
    log.info("翻譯（%s）：%s → %s", model, text[:40], zh[:40])
    return zh


def explain(text: str, model: str, lang: str = "zh-TW") -> str:
    """lang：用哪種語言講解（Config.explain_lang）。"""
    if model not in EXPLAIN_MODELS:
        model = next(iter(EXPLAIN_MODELS))
    out = _generate(model, (EXPLAIN_PROMPT_EN if lang == "en" else EXPLAIN_PROMPT).format(text=text), retries=2)
    log.info("講解（%s）：%s（%d 字）", model, text[:40], len(out))
    return out


# ---------------------------------------------------------------- 背景執行
class _Relay(QObject):
    done = Signal(object, object, str)  # (callback, 結果, 錯誤訊息)


_relay = None


def run_async(fn, callback):
    """在背景執行緒跑 fn()，完成後在主執行緒呼叫 callback(結果, 錯誤訊息)（成功時錯誤訊息是空字串）。"""
    global _relay
    if _relay is None:
        _relay = _Relay()
        _relay.done.connect(lambda cb, res, err: cb(res, err))

    def work():
        try:
            res, err = fn(), ""
        except AIError as e:
            res, err = None, str(e)
        except Exception as e:  # noqa: BLE001
            import traceback
            log.error("Gemini 呼叫失敗\n%s", redact(traceback.format_exc()))
            res, err = None, t("發生錯誤：{err}", err=redact(e))
        _relay.done.emit(callback, res, err)

    threading.Thread(target=work, name="gemini", daemon=True).start()
