@echo off
pyinstaller ^
  --name="FightAnalyzer" ^
  --onefile ^
  --windowed ^
  --hidden-import=ultralytics ^
  --hidden-import=cv2 ^
  --hidden-import=matplotlib ^
  --add-data="yolov8n.pt;." ^
  --icon=icon.ico ^
  app.py
pause