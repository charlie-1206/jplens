"""朗讀日文（單字本展開的卡片：單字、原句）。用 Windows 內建的日文語音（Qt TextToSpeech），不連網、不另外帶模型。

Windows 預設不一定有日文語音（要在「設定 → 時間與語言 → 語音 → 新增語音」加「日本語」）；沒有時 speak 回報 INSTALL_HINT。
引擎依序試 winrt（新版語音，Windows 設定裝的在這裡）、sapi（舊版）。語音引擎建立後要等事件迴圈跑過才會就緒，
所以第一次按的時候先記著，就緒後再唸。
"""

import logging

from PySide6.QtCore import QLocale
from PySide6.QtTextToSpeech import QTextToSpeech, QVoice  # noqa: F401（模組層 import：PyInstaller 才會帶 Qt 的語音外掛）

from .i18n import t

log = logging.getLogger("jplens")
INSTALL_HINT = t("這台電腦沒有日文語音：到 Windows「設定 → 時間與語言 → 語音 → 新增語音」加入「日本語」，裝好後再按一次")
ENGINES = ("winrt", "sapi")

_tts: QTextToSpeech | None = None
_tried: list[str] = []
_ready: bool | None = None  # None＝準備中；True＝有日文語音；False＝沒有
_pending: tuple[str, object] | None = None  # 準備好之前按的：(要唸的字, 失敗時的回呼)


def _japanese(voice) -> bool:
    return voice.locale().language() == QLocale.Japanese


def _next_engine():
    global _tts
    rest = [e for e in ENGINES if e in QTextToSpeech.availableEngines() and e not in _tried]
    if not rest:
        _finish(False)
        return
    _tried.append(rest[0])
    _tts = QTextToSpeech(rest[0])
    _tts.stateChanged.connect(_on_state)
    _on_state(_tts.state())  # 有的引擎一建立就是 Ready，不會再發 stateChanged


def _on_state(state):
    if _ready is not None or state not in (QTextToSpeech.State.Ready, QTextToSpeech.State.Error):
        return
    if state == QTextToSpeech.State.Ready:
        _tts.setLocale(QLocale(QLocale.Japanese, QLocale.Japan))
        voices = [v for v in _tts.availableVoices() if _japanese(v)]
        if voices:
            _tts.setVoice(voices[0])
            log.info("朗讀：%s 引擎、%s", _tried[-1], voices[0].name())
            _finish(True)
            return
    else:
        log.info("朗讀：%s 引擎無法使用：%s", _tried[-1], _tts.errorString())
    _tts.stateChanged.disconnect(_on_state)
    _next_engine()


def _finish(ok: bool):
    global _ready, _pending
    _ready = ok
    if not ok:
        log.info("朗讀：沒有日文語音")
    pending, _pending = _pending, None
    if pending:  # 準備好之前按的：直接唸或回報（不能再呼叫 speak：沒有語音時它會重新找，變成一直找下去）
        text, on_error = pending
        if ok:
            _say(text)
        elif on_error:
            on_error(INSTALL_HINT)


def prepare():
    """先建立語音引擎（單字本打開時呼叫，第一次按朗讀時就不用等）。之前沒找到日文語音就重新找
    （使用者照提示裝好語音後，不用重開程式；新裝的語音要建新的引擎才看得到）。"""
    global _tts, _ready
    if _ready is False:
        if _tts is not None:
            _tts.deleteLater()
        _tts, _ready = None, None
        _tried.clear()
    if _tts is None and _ready is None:
        _next_engine()


def speak(text: str, on_error=None):
    """唸 text（正在唸別的就先停掉）。沒有日文語音時呼叫 on_error(INSTALL_HINT)。"""
    global _pending
    text = (text or "").strip()
    if not text:
        return
    prepare()
    if _ready is None:
        _pending = (text, on_error)
        return
    if not _ready:
        if on_error:
            on_error(INSTALL_HINT)
        return
    _say(text)


def _say(text: str):
    _tts.stop()
    _tts.say(text)


def word_text(rec: dict) -> str:
    """單字要唸的字：讀音的假名（和卡片上的讀音一致：辛い 照記錄唸 からい 或 つらい，不讓語音自己猜）。"""
    return rec.get("reading") or rec.get("word", "")
