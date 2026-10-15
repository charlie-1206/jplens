"""單字本視窗、設定視窗共用的外觀：顏色、樣式表、線條圖示、深色標題列。

顏色和結果視窗（ui.STYLE）同一套：深色底、金色強調、淺藍讀音。
級數色塊比結果視窗的深一點，白字才看得清楚（對比 4.5:1 以上）。
"""

import ctypes

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap

C = {
    "bg": "#1c1e24", "title": "#16181d", "panel": "#20232a", "card": "#252830", "card_bd": "#2f333d",
    "card_open": "#4a5160", "input": "#262a33", "border": "#3a3f4b", "line": "#343842",
    "text": "#e8e8e8", "sub": "#a0a6b0", "dim": "#8f96a3", "reading": "#9fd0ff", "jp": "#cfd6e6",
    "accent": "#ffd27a", "accent_bg": "#5a4c26", "accent_dim": "#6b5a2c",
    "danger": "#ff9a8f", "danger_bd": "#6a3a36", "ok": "#8fd18f", "ok_bd": "#4a6a4a",
    "btn": "#3a3f4b", "btn_hover": "#4a5160", "btn_bd": "#555b66",
}
LEVEL_BG = {"N5": "#4f7a4f", "N4": "#3d7071", "N3": "#46699c", "N2": "#7a5296", "N1": "#a3463f",
            "級外": "#61656e", "專有名詞": "#7d6430", "文法": "#61656e"}
JP_FONTS = '"Yu Gothic UI", "Meiryo UI", "Meiryo", "MS UI Gothic"'

# 線條圖示（viewBox 0 0 20 20，stroke＝currentColor 由 icon() 換成指定顏色）
_PATHS = {
    "menu": '<path d="M3 5h14M3 10h14M3 15h14"/>',
    "search": '<circle cx="8.5" cy="8.5" r="5.5"/><path d="M13 13l4 4"/>',
    "filter": '<path d="M3 4h14l-5.5 6.5V16l-3 1.5v-7z"/>',
    "chev_down": '<path d="M5 8l5 5 5-5"/>',
    "chev_up": '<path d="M5 12l5-5 5 5"/>',
    "close": '<path d="M5 5l10 10M15 5L5 15"/>',
    "check": '<path d="M4 10.5l4 4 8-9"/>',
    "star": '<path d="M10 2.5l2.3 4.9 5.2.6-3.9 3.6 1.1 5.2L10 14.2l-4.7 2.6 1.1-5.2-3.9-3.6 5.2-.6z"/>',
    "eye": '<path d="M1.5 10S4.5 4 10 4s8.5 6 8.5 6-3 6-8.5 6S1.5 10 1.5 10z"/><circle cx="10" cy="10" r="2.5"/>',
}


def icon(name: str, color: str, fill: str = "none", size: int = 20, width: float = 1.6) -> QIcon:
    from PySide6.QtSvg import QSvgRenderer
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="{fill}" stroke="{color}" '
           f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{_PATHS[name]}</svg>')
    r = QSvgRenderer(QByteArray(svg.encode()))
    ic = QIcon()
    for scale in (1, 2):  # 125%／150% 縮放的螢幕也清楚
        pm = QPixmap(size * scale, size * scale)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        r.render(p, QRectF(0, 0, size * scale, size * scale))
        p.end()
        pm.setDevicePixelRatio(scale)
        ic.addPixmap(pm)
    return ic


def dark_title_bar(w):
    """Windows 10 20H1 以後的深色標題列（DWMWA_USE_IMMERSIVE_DARK_MODE＝20）；不支援的版本靜默略過。"""
    try:
        val = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(int(w.winId())), 20,
                                                   ctypes.byref(val), ctypes.sizeof(val))
    except Exception:  # noqa: BLE001
        pass


QSS = f"""
QWidget#window {{ background: {C['bg']}; }}
QWidget {{ color: {C['text']}; font-size: 14px; }}
QDialog {{ background: {C['bg']}; }}
QLabel {{ background: transparent; }}
QLabel#sub {{ color: {C['sub']}; font-size: 13px; }}
QLabel#dim {{ color: {C['dim']}; font-size: 12px; }}
QLabel#desc {{ color: {C['sub']}; font-size: 12px; }}
QLabel#note {{ color: {C['accent']}; font-size: 12px; }}
QLabel#section {{ color: {C['accent']}; font-size: 13px; font-weight: bold; }}
QLabel#h1 {{ color: #ffffff; font-size: 20px; font-weight: bold; }}
QLabel#word {{ color: #ffffff; font-size: 20px; font-weight: bold; font-family: {JP_FONTS}; }}
QLabel#reading {{ color: {C['reading']}; font-size: 14px; font-family: {JP_FONTS}; }}
QLabel#jp {{ color: {C['jp']}; font-size: 16px; font-family: {JP_FONTS}; }}
QLabel#summary {{ color: #d6d9de; font-size: 14px; }}
QLabel#meaning {{ font-size: 14px; }}
QLabel#en {{ color: #a8a8a8; font-size: 12px; }}
QLabel#learned {{ color: {C['ok']}; border: 1px solid {C['ok_bd']}; border-radius: 3px; padding: 0px 5px; font-size: 11px; }}
QLabel#reserved {{ color: {C['sub']}; border: 1px solid {C['card_open']}; border-radius: 3px; padding: 0px 4px; font-size: 11px; }}
QLabel#tag {{ color: {C['jp']}; background: #2f333d; border-radius: 10px; padding: 1px 9px; font-size: 12px; font-family: {JP_FONTS}; }}
QLabel#toast {{ font-size: 13px; }}
QFrame#card {{ background: {C['card']}; border: 1px solid {C['card_bd']}; border-radius: 8px; }}
QFrame#card[open="true"] {{ border-color: {C['card_open']}; }}
QFrame#cardDetail {{ border-top: 1px solid {C['line']}; }}
QFrame#quote {{ background: rgba(255,255,255,13); border-radius: 6px; }}
QFrame#group {{ background: {C['card']}; border: 1px solid {C['card_bd']}; border-radius: 8px; }}
QFrame#rowLine {{ border-top: 1px solid {C['card_bd']}; }}
QFrame#drawer {{ background: {C['panel']}; border-right: 1px solid {C['border']}; }}
QFrame#popupPanel {{ background: #23262e; border: 1px solid {C['card_open']}; border-radius: 8px; }}
QFrame#toastBox {{ background: #343a46; border: 1px solid {C['card_open']}; border-radius: 6px; }}
QFrame#hint {{ background: rgba(255,210,122,20); border: 1px solid {C['accent_dim']}; border-radius: 6px; }}
QFrame#sep {{ background: {C['line']}; max-height: 1px; min-height: 1px; }}
QWidget#scrim {{ background: rgba(0,0,0,115); }}
QLineEdit, QPlainTextEdit {{ background: {C['input']}; border: 1px solid {C['border']}; border-radius: 6px;
    padding: 6px 8px; color: {C['text']}; selection-background-color: {C['btn_hover']}; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {C['card_open']}; }}
QLineEdit:disabled {{ color: {C['dim']}; }}
QLineEdit#search {{ padding: 8px 8px; }}
QLineEdit#path {{ background: {C['bg']}; color: {C['jp']}; font-size: 13px; }}
QPushButton {{ background: {C['btn']}; border: 1px solid {C['btn_bd']}; border-radius: 4px; padding: 6px 12px;
    color: {C['text']}; font-size: 13px; }}
QPushButton:hover {{ background: {C['btn_hover']}; }}
QPushButton:disabled {{ color: {C['dim']}; background: #2c3038; }}
QPushButton#primary {{ background: {C['accent']}; color: {C['bg']}; border: none; font-weight: bold; padding: 6px 18px; }}
QPushButton#primary:hover {{ background: #ffdd99; }}
QPushButton#primary:disabled {{ background: {C['accent_dim']}; color: {C['bg']}; }}
QPushButton#danger {{ background: transparent; color: {C['danger']}; border-color: {C['danger_bd']}; }}
QPushButton#danger:hover {{ background: #3a2624; }}
QPushButton#learnedBtn {{ background: transparent; color: {C['ok']}; border-color: {C['ok_bd']}; }}
QPushButton#link {{ background: transparent; border: none; color: {C['accent']}; font-weight: bold; padding: 4px 8px; }}
QPushButton#plain {{ background: transparent; border: none; color: {C['sub']}; }}
QPushButton#plain:hover {{ color: {C['text']}; }}
QPushButton#nav {{ background: transparent; border: none; border-radius: 6px; text-align: left; padding: 9px 10px;
    font-size: 14px; }}
QPushButton#nav:hover {{ background: #2b2f38; }}
QPushButton#nav:checked {{ background: #343a46; color: {C['accent']}; }}
QPushButton#chip {{ background: transparent; border: 1px solid {C['card_open']}; border-radius: 14px; padding: 4px 12px; }}
QPushButton#chip:hover {{ background: #2b2f38; }}
QPushButton#chip:checked {{ background: {C['accent_bg']}; border-color: {C['accent']}; color: {C['accent']}; }}
QPushButton#activeChip {{ background: #3a3220; border: 1px solid {C['accent_dim']}; border-radius: 11px;
    padding: 2px 10px; color: {C['accent']}; font-size: 12px; }}
QPushButton#seg {{ background: transparent; border: none; border-left: 1px solid {C['card_open']}; border-radius: 0px;
    padding: 6px 14px; }}
QPushButton#seg[first="true"] {{ border-left: none; }}
QPushButton#seg:checked {{ background: {C['accent_bg']}; color: {C['accent']}; }}
QPushButton#seg:disabled {{ color: {C['dim']}; background: transparent; }}
QFrame#segBox {{ border: 1px solid {C['card_open']}; border-radius: 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; }}
QToolButton:hover {{ background: #2b2f38; }}
QToolButton:checked {{ background: {C['accent_bg']}; border-color: {C['accent']}; }}
QToolButton#active {{ border-color: {C['accent']}; }}
QComboBox {{ background: {C['input']}; border: 1px solid {C['card_open']}; border-radius: 4px; padding: 4px 10px;
    font-size: 13px; min-width: 120px; }}
QComboBox:disabled {{ color: {C['dim']}; }}
QComboBox::drop-down {{ border: none; background: transparent; width: 26px; }}
QComboBox::down-arrow {{ image: url({{ARROW}}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: #2b2e36; border: 1px solid {C['card_open']}; color: {C['text']};
    selection-background-color: {C['btn_hover']}; outline: none; }}
QRadioButton, QCheckBox {{ font-size: 14px; spacing: 8px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #4a505b; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #5a616d; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QToolTip {{ background: #2b2e36; color: {C['text']}; border: 1px solid {C['card_open']}; padding: 4px; }}
"""


def _arrow_file() -> str:
    """下拉選單的箭頭：樣式表只能用檔案路徑指定圖片，寫一個小 SVG 到暫存資料夾。"""
    import tempfile
    from pathlib import Path
    p = Path(tempfile.gettempdir()) / "jplens_combo_arrow.svg"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" stroke="{C["sub"]}" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_PATHS["chev_down"]}</svg>')
    try:
        if not p.exists() or p.read_text(encoding="utf-8") != svg:
            p.write_text(svg, encoding="utf-8")
    except OSError:
        return ""
    return p.as_posix()


QSS = QSS.replace("{ARROW}", _arrow_file())


def badge_style(level: str) -> str:
    return (f"background:{LEVEL_BG.get(level, '#61656e')}; color:white; border-radius:3px; padding:0px 6px;"
            "font-size:11px;")
