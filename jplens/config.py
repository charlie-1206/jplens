"""使用者設定（config.json，不進版控）。沒有檔案時用預設值，改過的值會寫回。"""

import json
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)  # PyInstaller 打包的執行檔（packaging/build.py）
# APP_DIR：程式附的唯讀資料（模型、字典）；ROOT：使用者的資料（config.json、vocab/、logs/）。
# 開發時兩個都是專案資料夾；打包後 APP_DIR 是 _internal/，ROOT 是 jplens.exe 所在的資料夾。
APP_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(sys.executable).resolve().parent if FROZEN else APP_DIR
CONFIG_PATH = ROOT / "config.json"
# 第一次使用：還沒有設定檔、也沒啟動過（logs/ 是啟動時建的；打包時自我測試留下的 logs/ 會清掉）。
# 要在 import 時判斷：app.main() 會先建 logs/ 才讀設定。App 啟動時把設定存下來，之後就看設定檔
NEW_USER = not CONFIG_PATH.exists() and not (ROOT / "logs").exists()
LANGS = ("zh-TW", "en")


def system_lang() -> str:
    """Windows 的顯示語言：中文（繁體、簡體都算，介面只有繁體）→ zh-TW，其他 → en。"""
    try:
        import ctypes
        return "zh-TW" if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF == 0x04 else "en"  # LANG_CHINESE
    except Exception:  # noqa: BLE001（不是 Windows）
        return "en"

# 框選時要按住的鍵（Windows 虛擬鍵碼）；全部按住時拖曳滑鼠左鍵才會框選
HOTKEYS = {
    "alt": [0x12],
    "ctrl+shift": [0x11, 0x10],
    "ctrl+alt": [0x11, 0x12],
}
HOTKEY_NAMES = {"alt": "Alt", "ctrl+shift": "Ctrl＋Shift", "ctrl+alt": "Ctrl＋Alt"}
FONT_SCALES = {"s": 0.88, "m": 1.0, "l": 1.22}       # 結果視窗字級
POPUP_WIDTHS = {"narrow": 380, "std": 440, "wide": 520}  # 結果視窗寬度（邏輯像素）


@dataclass
class Config:
    hotkey: str = "alt"
    min_level: int = 3        # 只顯示 N{min_level} 以上的單字（5 = 全部）；級外一律顯示
    show_proper: bool = False  # 是否顯示專有名詞（人名、地名）
    show_text: bool = True     # 結果視窗是否顯示辨識出的原文
    select_mode: str = "text"  # text = 反白（沿著文字拖曳選字）、box = 框選（拖曳出矩形）、manga = 漫畫
    ink_columns: bool = True   # 直排小說的框選、反白用墨水投影找欄（jplens/inkcols.py）
    layout_rules: bool = False  # 偵測框的舊補丁規則（拆框、切振假名、濾振假名、漫畫重切欄），2026-10-15 起預設關
    # 以下由設定視窗（settings_ui.py）修改
    mode_key: bool = True      # 按住熱鍵＋右鍵切換選取模式
    show_spinner: bool = True  # 按下熱鍵到版面偵測好之前，游標旁轉圈
    font_size: str = "m"       # 結果視窗字級：s／m／l（FONT_SCALES）
    popup_width: str = "std"   # 結果視窗寬度：narrow／std／wide（POPUP_WIDTHS）
    preload_manga: bool = False  # 啟動時就載入漫畫模型（多用約 160 MB 記憶體）
    run_as_admin: bool = False  # 以系統管理員身分執行（以系統管理員身分執行的遊戲裡才收得到按鍵，elevate.py）
    vocab_dir: str = ""        # 單字本資料夾；空字串＝專案的 vocab/
    export_scope: str = "all"  # 匯出單字時預設的範圍：all／marked
    export_learned: bool = True  # 匯出單字時已學會的也匯出
    auto_translate: bool = False  # 舊設定（2026-10-17 前）：True 載入時轉成 translate_timing="auto"
    # 整句翻譯的時機：manual＝按「翻譯」才翻；prefetch＝原文夠長就在背景先翻好，按了才顯示；auto＝翻好直接顯示
    translate_timing: str = "manual"
    prefetch_min_chars: int = 5  # 背景預先翻譯／自動翻譯：原文（去掉標點）至少幾個字才翻
    # 翻譯來源：gemini（預設，ai.TRANSLATE_MODEL）或 openai（任何 OpenAI 相容的服務：網址＋模型；Key 存在認證管理員）
    translate_provider: str = "gemini"
    openai_base_url: str = ""
    openai_model: str = ""
    explain_model: str = "gemini-3.8-flash"  # 單字本句子頁「講解」用的模型（ai.EXPLAIN_MODELS）
    # 無字幕模式（listen.py）：按住熱鍵＋這個字母開關；結果視窗的位置（邏輯像素，空的＝螢幕右下角）
    listen_key: str = "L"
    listen_partial: bool = True  # 邊講邊顯示（講話中每 0.8 秒重新辨識一次，CPU 約多用單核 10%）
    listen_pos: list = None
    listen_speed: float = 1.0  # 影片的播放速度（倍速）：辨識前放慢回原速（listen.SPEEDS）
    src_lang: str = "ja"  # 辨識的語言（要學的語言）：預留，只有日文
    # 解釋的語言（lang.py）、介面語言（i18n.py）：預設看 Windows 的顯示語言，中文 → 繁體中文，其他 → 英文（使用者 10-23）；
    # 改預設以前就在用的人照舊中文（load()）
    explain_lang: str = field(default_factory=system_lang)
    ui_lang: str = field(default_factory=system_lang)

    @classmethod
    def load(cls) -> "Config":
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in names})
        for k in ("explain_lang", "ui_lang"):
            if k not in data and not NEW_USER:  # 預設改看 Windows 語言（10-23）以前就在用、設定檔沒寫語言（沒改過設定）的人：照舊中文
                setattr(cfg, k, "zh-TW")
            if getattr(cfg, k) not in LANGS:
                setattr(cfg, k, system_lang())
        if cfg.hotkey not in HOTKEYS:
            cfg.hotkey = "alt"
        if cfg.select_mode not in ("text", "box", "manga"):
            cfg.select_mode = "text"
        if cfg.font_size not in FONT_SCALES:
            cfg.font_size = "m"
        if cfg.popup_width not in POPUP_WIDTHS:
            cfg.popup_width = "std"
        if cfg.export_scope not in ("all", "marked"):
            cfg.export_scope = "all"
        if cfg.auto_translate and "translate_timing" not in data:
            cfg.translate_timing = "auto"
        if cfg.translate_timing not in ("manual", "prefetch", "auto"):
            cfg.translate_timing = "manual"
        if cfg.translate_provider not in ("gemini", "openai"):
            cfg.translate_provider = "gemini"
        try:
            cfg.prefetch_min_chars = max(1, int(cfg.prefetch_min_chars))
        except (TypeError, ValueError):
            cfg.prefetch_min_chars = 5
        if not (isinstance(cfg.listen_key, str) and len(cfg.listen_key) == 1 and "A" <= cfg.listen_key.upper() <= "Z"):
            cfg.listen_key = "L"
        cfg.listen_key = cfg.listen_key.upper()
        if not (isinstance(cfg.listen_pos, list) and len(cfg.listen_pos) == 2
                and all(isinstance(v, int) for v in cfg.listen_pos)):
            cfg.listen_pos = None
        if cfg.listen_speed not in (1.0, 1.25, 1.5, 1.75, 2.0):  # listen.SPEEDS（不在這裡 import，避免載入 numpy）
            cfg.listen_speed = 1.0
        return cfg

    def save(self):
        CONFIG_PATH.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def hotkey_vks(self) -> list[int]:
        return HOTKEYS[self.hotkey]

    @property
    def hotkey_name(self) -> str:
        return HOTKEY_NAMES[self.hotkey]

    @property
    def listen_name(self) -> str:
        return f"{self.hotkey_name}＋{self.listen_key}"

    @property
    def vocab_path(self) -> Path:
        return Path(self.vocab_dir) if self.vocab_dir else ROOT / "vocab"
