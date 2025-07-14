@echo off
echo.
echo ===============================================
echo  Precipitation Data Preprocessing - Windows Setup
echo ===============================================
echo.

REM Check if conda is installed
conda --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda is not installed or not in PATH
    echo.
    echo Please install Miniconda or Anaconda first:
    echo 1. Download from: https://docs.conda.io/en/latest/miniconda.html
    echo 2. Run the installer and follow the instructions
    echo 3. Make sure to check "Add Miniconda to PATH" during installation
    echo 4. Restart Command Prompt and run this script again
    echo.
    pause
    exit /b 1
)

echo [INFO] Conda found! Version:
conda --version

echo.
echo [INFO] Creating conda environment 'precipitation_data'...
conda env create -f environment.yml
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment
    echo Check that environment.yml exists in the current directory
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Setup completed successfully!
echo.
echo To run the preprocessing pipeline:
echo.
echo Option 1 - Command Prompt:
echo   autorun_preprocessing.bat
echo.
echo Option 2 - PowerShell:
echo   .\autorun_preprocessing.ps1
echo.
echo For help with options:
echo   autorun_preprocessing.bat --help
echo   .\autorun_preprocessing.ps1 -Help
echo.
pause 