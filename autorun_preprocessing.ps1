# Precipitation Data Preprocessing Pipeline - PowerShell Version
# Usage: .\autorun_preprocessing.ps1 [options]

param(
    [switch]$SkipBmp,
    [switch]$SkipFilter,
    [switch]$Force,
    [string]$Pattern = "",
    [switch]$Grayscale,
    [string]$Channel = "",
    [switch]$InMemory,
    [switch]$Help
)

# Color functions for better output
function Write-ColorText {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

function Write-Success { param([string]$Text) Write-ColorText $Text "Green" }
function Write-Warning { param([string]$Text) Write-ColorText $Text "Yellow" }
function Write-Error { param([string]$Text) Write-ColorText $Text "Red" }
function Write-Info { param([string]$Text) Write-ColorText $Text "Cyan" }
function Write-Header { param([string]$Text) Write-ColorText $Text "Blue" }

# Show help
if ($Help) {
    Write-Host ""
    Write-Header "Precipitation Data Preprocessing Pipeline - PowerShell Version"
    Write-Host ""
    Write-Warning "Usage:"
    Write-Host "  .\autorun_preprocessing.ps1 [options]"
    Write-Host ""
    Write-Warning "Options:"
    Write-Host "  -SkipBmp        Skip BMP to TIFF conversion"
    Write-Host "  -SkipFilter     Skip non-local means filtering"
    Write-Host "  -Force          Force reprocessing of existing files"
    Write-Host "  -Pattern FILE   Process only files matching pattern"
    Write-Host "  -Grayscale      Convert to grayscale TIFF"
    Write-Host "  -Channel N      Extract specific channel (0=Red, 1=Green, 2=Blue, 3=Alpha, 4=Alpha/Green fallback)"
    Write-Host "  -InMemory       Use in-memory workflow (BMP to filtered TIFF, no intermediate files)"
    Write-Host "  -Help           Show this help message"
    Write-Host ""
    Write-Warning "Examples:"
    Write-Host "  .\autorun_preprocessing.ps1"
    Write-Host "  .\autorun_preprocessing.ps1 -SkipBmp"
    Write-Host "  .\autorun_preprocessing.ps1 -Grayscale -Force"
    Write-Host "  .\autorun_preprocessing.ps1 -InMemory"
    Write-Host "  .\autorun_preprocessing.ps1 -Channel 1 -Pattern `"sample`""
    Write-Host ""
    exit 0
}

# Configuration
$EnvName = "precipitation_data"

Write-Host ""
Write-Header "=========================================="
Write-Header "  Precipitation Data Preprocessing Pipeline"
Write-Header "=========================================="
Write-Host ""

# Check if conda is available
try {
    $condaVersion = conda --version 2>$null
    if (-not $condaVersion) { throw "Conda not found" }
} catch {
    Write-Error "Error: Conda is not installed or not in PATH"
    Write-Host "Please install Miniconda or Anaconda first"
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

# Check if environment exists
$envExists = conda env list | Select-String $EnvName
if (-not $envExists) {
    Write-Warning "Creating conda environment '$EnvName'..."
    conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create conda environment"
        exit 1
    }
    Write-Success "Environment created successfully"
} else {
    Write-Success "Environment '$EnvName' already exists"
}

# Activate conda environment (note: this is tricky in PowerShell)
Write-Warning "Activating conda environment..."
Write-Info "Note: If activation fails, run this command first:"
Write-Info "conda init powershell"
Write-Info "Then restart PowerShell and run this script again."

# Try to initialize conda for PowerShell if needed
try {
    & conda init powershell 2>$null
} catch {
    # Ignore errors - init might already be done
}

# Set up conda activation
$condaBase = conda info --base
if ($condaBase) {
    $activateScript = Join-Path $condaBase "shell\condabin\conda-hook.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        conda activate $EnvName
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to activate conda environment"
            Write-Warning "Try running: conda init powershell"
            Write-Warning "Then restart PowerShell and run this script again"
            exit 1
        }
    } else {
        Write-Warning "Conda PowerShell integration not found. Using fallback method."
        Write-Warning "You may need to run: conda init powershell"
    }
}

# Build command arguments
$BmpArgs = @()
$FilterArgs = @()

if ($Force) {
    $BmpArgs += "--force"
    $FilterArgs += "--force"
}

if ($Pattern) {
    $BmpArgs += "--pattern", $Pattern
    $FilterArgs += "--pattern", $Pattern
}

if ($Grayscale) {
    $BmpArgs += "--grayscale"
}

if ($Channel) {
    $BmpArgs += "--channel", $Channel
}

if ($InMemory) {
    Write-Host ""
    Write-Header "Using in-memory workflow (BMP -> filtered TIFF, no intermediate files)"
    
    # Build command for in-memory workflow
    $MemoryArgs = @("bmp_to_filtered_workflow.py", "3mM", "6mM")
    
    if ($Force) { $MemoryArgs += "--force" }
    if ($Pattern) { $MemoryArgs += "--pattern", $Pattern }
    if ($Grayscale) { $MemoryArgs += "--grayscale" }
    if ($Channel) { $MemoryArgs += "--channel", $Channel }
    
    $cmd = "python " + ($MemoryArgs -join " ")
    Write-Warning "Command: $cmd"
    Write-Host ""
    
    & python $MemoryArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "In-memory workflow failed"
        exit 1
    }
    Write-Success "In-memory workflow completed successfully"
} else {
    # Traditional two-step workflow
    # Step 1: BMP to TIFF conversion
    if (-not $SkipBmp) {
        Write-Host ""
        Write-Header "Step 1: Converting BMP files to TIFF..."
        $cmd = "python bmp_to_tiff_converter.py 3mM 6mM " + ($BmpArgs -join " ")
        Write-Warning "Command: $cmd"
        Write-Host ""
        
        $args = @("bmp_to_tiff_converter.py", "3mM", "6mM") + $BmpArgs
        & python $args
        if ($LASTEXITCODE -ne 0) {
            Write-Error "BMP to TIFF conversion failed"
            exit 1
        }
        Write-Success "BMP to TIFF conversion completed successfully"
    } else {
        Write-Warning "Skipping BMP to TIFF conversion (-SkipBmp specified)"
    }

    # Step 2: Non-local means filtering
    if (-not $SkipFilter) {
        Write-Host ""
        Write-Header "Step 2: Applying non-local means filtering..."
        $cmd = "python nonlocal_means_filter.py 3mM 6mM " + ($FilterArgs -join " ")
        Write-Warning "Command: $cmd"
        Write-Host ""
        
        $args = @("nonlocal_means_filter.py", "3mM", "6mM") + $FilterArgs
        & python $args
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Non-local means filtering failed"
            exit 1
        }
        Write-Success "Non-local means filtering completed successfully"
    } else {
        Write-Warning "Skipping non-local means filtering (-SkipFilter specified)"
    }
}

Write-Host ""
Write-Success "=== PIPELINE COMPLETED SUCCESSFULLY ==="
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - Output HDF5 files now include Na2CO3, CaCl, and replicate attributes if parsed from folder name." -ForegroundColor Cyan
Write-Host "  - Use analyze_timeseries.py or investigate_streaks.py for quality checks" -ForegroundColor Cyan
Write-Host "  - Use image_quality_inspector.py to compare BMP vs TIFF quality (if needed)" -ForegroundColor Cyan
Write-Host ""
Write-Header "Next steps:"
Write-Host "- Use image_quality_inspector.py to compare BMP vs TIFF quality"
Write-Host "- Use raw_vs_filtered_inspector.py to analyze filtering results"
Write-Host "" 