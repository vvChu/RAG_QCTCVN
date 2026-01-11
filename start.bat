@echo off
echo ================================================
echo   CCBA RAG System Launcher
echo ================================================
echo.
echo Starting API Server (port 8000)...
start "CCBA RAG API" cmd /k "python server.py"

echo Waiting for API to initialize (60s)...
timeout /t 60 /nobreak

echo Starting Streamlit UI (port 8501)...
start "CCBA RAG UI" cmd /k "python -m streamlit run app.py"

echo.
echo ================================================
echo   Services Started:
echo   - API:  http://127.0.0.1:8000/docs
echo   - UI:   http://127.0.0.1:8501
echo ================================================
echo.
pause
