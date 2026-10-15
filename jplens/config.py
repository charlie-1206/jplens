"""使用者設定（config.json，不進版控）。沒有檔案時用預設值，改過的值會寫回。"""

import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)  # PyInstaller 打包的執行檔（packaging/build.py）
# APP_DIR：程式附的唯讀資料（模型、字典）；ROOT：使用者的資料（config.json、vocab/、logs/）。
# 開發時兩個都是專案資料夾；打包後 APP_DIR 是 _internal/，ROOT 是 jplens.exe 所在的資料夾。
APP_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(sys.executable).resolve().parent if FROZEN else APP_DIR
CONFIG_PATH = ROOT / "config.json"

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
    # 預留（還沒實作）：辨識的語言（要學的語言）、解釋的語言、介面語言
    src_lang: str = "ja"
    explain_lang: str = "zh-TW"
    ui_lang: str = "zh-TW"

    @classmethod
    def load(cls) -> "Config":
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in names})
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
    def vocab_path(self) -> Path:
        return Path(self.vocab_dir) if self.vocab_dir else ROOT / "vocab"
