@echo off
REM TrackMyPDB - Windows installer (creates a local virtual environment)
python --version >nul 2>&1 || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
python -m venv .venv || (echo Could not create virtual environment & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[app,agent]" || (echo Installation failed & pause & exit /b 1)
python -c "import trackmypdb, rdkit; print('TrackMyPDB', trackmypdb.__version__, 'installed with RDKit', rdkit.__version__)"
echo.
echo Done. Start the app with run.bat
pause
