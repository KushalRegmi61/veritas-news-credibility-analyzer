@echo off
REM Script to run FastAPI server with Uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
