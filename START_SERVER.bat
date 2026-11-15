@echo off
echo =====================================================
echo  NASA Astronaut Health CDSS - Startup Script
echo =====================================================
cd /d "c:\Users\HP\Documents\nasa ai hackhathon"
echo Starting server on http://127.0.0.1:8080 ...
start "" "http://127.0.0.1:8080"
.\.venv\Scripts\python.exe src/server.py 8080
pause
