@echo off
setlocal enabledelayedexpansion

REM Precipitation Data Preprocessing Pipeline - Windows Batch Version
REM Usage: autorun_preprocessing.bat [options]

REM Color codes for Windows (requires ANSI support)
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "NC=[0m"

REM Default options
set "SKIP_BMP=false"
set "SKIP_FILTER=false"
set "FORCE=false"
set "PATTERN="
set "GRAYSCALE=false"
set "CHANNEL="
set "IN_MEMORY=false"
set "ENV_NAME=precipitation_data"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto start_processing
if "%~1"=="--skip-bmp" (
    set "SKIP_BMP=true"
    shift
    goto parse_args
)
if "%~1"=="--skip-filter" (
    set "SKIP_FILTER=true"
    shift
    goto parse_args
)
if "%~1"=="--force" (
    set "FORCE=true"
    shift
    goto parse_args
)
if "%~1"=="--pattern" (
    set "PATTERN=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--grayscale" (
    set "GRAYSCALE=true"
    shift
    goto parse_args
)
if "%~1"=="--channel" (
    set "CHANNEL=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--in-memory" (
    set "IN_MEMORY=true"
    shift
    goto parse_args
)
if "%~1"=="--help" (
    goto show_help
)
echo %RED%Unknown option: %~1%NC%
goto show_help

:show_help
echo.
echo %BLUE%Precipitation Data Preprocessing Pipeline - Windows Version%NC%
echo.
echo %YELLOW%Usage:%NC%
echo   autorun_preprocessing.bat [options]
echo.
echo %YELLOW%Options:%NC%
echo   --skip-bmp      Skip BMP to TIFF conversion
echo   --skip-filter   Skip non-local means filtering
echo   --force         Force reprocessing of existing files
echo   --pattern FILE  Process only files matching pattern
echo   --grayscale     Convert to grayscale TIFF
echo   --channel N     Extract specific channel (0=Red, 1=Green, 2=Blue, 3=Alpha, 4=Alpha/Green fallback)
echo   --in-memory     Use in-memory workflow (BMP to filtered TIFF, no intermediate files)
echo   --help          Show this help message
echo.
echo %YELLOW%Examples:%NC%
echo   autorun_preprocessing.bat
echo   autorun_preprocessing.bat --skip-bmp
echo   autorun_preprocessing.bat --grayscale --force
echo   autorun_preprocessing.bat --in-memory
echo   autorun_preprocessing.bat --channel 1 --pattern "sample"
echo.
exit /b 0

:start_processing
echo.
echo %BLUE%==========================================%NC%
echo %BLUE%  Precipitation Data Preprocessing Pipeline%NC%
echo %BLUE%==========================================%NC%
echo.

REM Check if conda is available
conda --version >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: Conda is not installed or not in PATH%NC%
    echo Please install Miniconda or Anaconda first
    exit /b 1
)

REM Check if environment exists
conda env list | findstr /C:"%ENV_NAME%" >nul
if errorlevel 1 (
    echo %YELLOW%Creating conda environment '%ENV_NAME%'...%NC%
    conda env create -f environment.yml
    if errorlevel 1 (
        echo %RED%Failed to create conda environment%NC%
        exit /b 1
    )
    echo %GREEN%Environment created successfully%NC%
) else (
    echo %GREEN%Environment '%ENV_NAME%' already exists%NC%
)

REM Activate conda environment
echo %YELLOW%Activating conda environment...%NC%
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo %RED%Failed to activate conda environment%NC%
    exit /b 1
)

REM Build command arguments
set "BMP_ARGS="
set "FILTER_ARGS="

if "%FORCE%"=="true" (
    set "BMP_ARGS=!BMP_ARGS! --force"
    set "FILTER_ARGS=!FILTER_ARGS! --force"
)

if not "%PATTERN%"=="" (
    set "BMP_ARGS=!BMP_ARGS! --pattern %PATTERN%"
    set "FILTER_ARGS=!FILTER_ARGS! --pattern %PATTERN%"
)

if "%GRAYSCALE%"=="true" (
    set "BMP_ARGS=!BMP_ARGS! --grayscale"
)

if not "%CHANNEL%"=="" (
    set "BMP_ARGS=!BMP_ARGS! --channel %CHANNEL%"
)

if "%IN_MEMORY%"=="true" (
    echo.
    echo %BLUE%Using in-memory workflow (BMP to filtered TIFF, no intermediate files)%NC%
    
    REM Build command for in-memory workflow
    set "MEMORY_CMD=python bmp_to_filtered_workflow.py 3mM 6mM"
    if "%FORCE%"=="true" set "MEMORY_CMD=!MEMORY_CMD! --force"
    if not "%PATTERN%"=="" set "MEMORY_CMD=!MEMORY_CMD! --pattern %PATTERN%"
    if "%GRAYSCALE%"=="true" set "MEMORY_CMD=!MEMORY_CMD! --grayscale"
    if not "%CHANNEL%"=="" set "MEMORY_CMD=!MEMORY_CMD! --channel %CHANNEL%"
    
    echo %YELLOW%Command: !MEMORY_CMD!%NC%
    echo.
    
    !MEMORY_CMD!
    if errorlevel 1 (
        echo %RED%In-memory workflow failed%NC%
        exit /b 1
    )
    echo %GREEN%In-memory workflow completed successfully%NC%
) else (
    REM Traditional two-step workflow
    REM Step 1: BMP to TIFF conversion
    if "%SKIP_BMP%"=="false" (
        echo.
        echo %BLUE%Step 1: Converting BMP files to TIFF...%NC%
        echo %YELLOW%Command: python bmp_to_tiff_converter.py 3mM 6mM !BMP_ARGS!%NC%
        echo.
        
        python bmp_to_tiff_converter.py 3mM 6mM !BMP_ARGS!
        if errorlevel 1 (
            echo %RED%BMP to TIFF conversion failed%NC%
            exit /b 1
        )
        echo %GREEN%BMP to TIFF conversion completed successfully%NC%
    ) else (
        echo %YELLOW%Skipping BMP to TIFF conversion (--skip-bmp specified)%NC%
    )

    REM Step 2: Non-local means filtering
    if "%SKIP_FILTER%"=="false" (
        echo.
        echo %BLUE%Step 2: Applying non-local means filtering...%NC%
        echo %YELLOW%Command: python nonlocal_means_filter.py 3mM 6mM !FILTER_ARGS!%NC%
        echo.
        
        python nonlocal_means_filter.py 3mM 6mM !FILTER_ARGS!
        if errorlevel 1 (
            echo %RED%Non-local means filtering failed%NC%
            exit /b 1
        )
        echo %GREEN%Non-local means filtering completed successfully%NC%
    ) else (
        echo %YELLOW%Skipping non-local means filtering (--skip-filter specified)%NC%
    )
)

echo.
echo %GREEN%========================================%NC%
echo %GREEN%  Processing completed successfully!%NC%
echo %GREEN%========================================%NC%
echo.
echo %YELLOW%Summary:%NC%
if "%SKIP_BMP%"=="false" echo - BMP to TIFF conversion: %GREEN%COMPLETED%NC%
if "%SKIP_BMP%"=="true" echo - BMP to TIFF conversion: %YELLOW%SKIPPED%NC%
if "%SKIP_FILTER%"=="false" echo - Non-local means filtering: %GREEN%COMPLETED%NC%
if "%SKIP_FILTER%"=="true" echo - Non-local means filtering: %YELLOW%SKIPPED%NC%
echo.
echo %BLUE%Next steps:%NC%
echo - Use image_quality_inspector.py to compare BMP vs TIFF quality
echo - Use raw_vs_filtered_inspector.py to analyze filtering results
echo.

endlocal 