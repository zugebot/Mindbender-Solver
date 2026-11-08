@echo off
rem Serves web/ at http://localhost:8770/ in the background (pythonw, no window; see serve.py).
rem Double-click it, or run it from any folder. Stop it in Task Manager (pythonw.exe).
start "" pythonw "%~dp0serve.py" %*
