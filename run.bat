@echo off
echo ============================================
echo   Cricket Performance Predictor
echo ============================================
echo.

:MENU
echo.
echo Select an option:
echo.
echo 1. Generate Data
echo 2. Run EDA
echo 3. Train Model
echo 4. Start FastAPI Server
echo 5. Start Streamlit UI
echo 6. Run Complete Pipeline
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto GENERATE
if "%choice%"=="2" goto EDA
if "%choice%"=="3" goto TRAIN
if "%choice%"=="4" goto API
if "%choice%"=="5" goto STREAMLIT
if "%choice%"=="6" goto COMPLETE
if "%choice%"=="7" goto END

:GENERATE
echo.
echo Generating cricket data...
python src\data_generation.py
echo.
pause
goto MENU

:EDA
echo.
echo Running EDA...
python src\eda.py
echo.
pause
goto MENU

:TRAIN
echo.
echo Training model...
python src\model_training.py
echo.
pause
goto MENU

:API
echo.
echo Starting FastAPI server...
echo API Docs: http://localhost:8000/docs
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
cd ..
goto MENU

:STREAMLIT
echo.
echo Starting Streamlit UI...
streamlit run app\streamlit_app.py
goto MENU

:COMPLETE
echo.
echo Running complete pipeline...
echo.
echo Step 1: Generating data...
python src\data_generation.py
echo.
echo Step 2: Running EDA...
python src\eda.py
echo.
echo Step 3: Training model...
python src\model_training.py
echo.
echo ============================================
echo Pipeline Complete!
echo ============================================
echo.
echo You can now:
echo - Start FastAPI: python -m uvicorn app.main:app --reload
echo - Start Streamlit: streamlit run app\streamlit_app.py
echo.
pause
goto MENU

:END
echo.
echo Goodbye!
exit
