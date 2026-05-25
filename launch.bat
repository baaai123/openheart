@echo off
if exist "D:\electron-l2d\package.json" (
    start "" /min cmd /c "cd /d D:\electron-l2d && npm start"
)
exit
