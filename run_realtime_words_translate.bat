@echo off
cd /d "%~dp0"
rem 用哪個 Python：環境變數 JPLENS_PYTHONW（pythonw.exe 的完整路徑）；沒設就用 PATH 上的 pythonw
if defined JPLENS_PYTHONW (start "" "%JPLENS_PYTHONW%" -m jplens) else (start "" pythonw -m jplens)
