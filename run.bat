@echo off
REM TrackMyPDB - start the web app (run install.bat once first)
if not exist .venv\Scripts\activate.bat (echo Run install.bat first & pause & exit /b 1)
call .venv\Scripts\activate.bat
streamlit run streamlit_app.py
