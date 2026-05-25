@echo off
title OpenHeart

if exist "D:\electron-l2d\package.json" (
    start "OpenHeart" /d "D:\electron-l2d" cmd /k npm start
) else (
    echo electron-l2d not found at D:\electron-l2d
    pause
)
