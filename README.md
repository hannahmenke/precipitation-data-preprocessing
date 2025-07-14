# Precipitation Data Preprocessing Pipeline

An automated pipeline for preprocessing precipitation imaging data from BMP files to filtered TIFF files using non-local means filtering with Avizo-matched parameters.

## 🚀 Cross-Platform Support
- ✅ **Windows**: Batch scripts (`.bat`) and PowerShell (`.ps1`)
- ✅ **macOS/Linux**: Bash scripts (`.sh`)
- ✅ **One-click setup** for all platforms

## Prerequisites

### Windows
1. **Install Miniconda/Anaconda** (if not already installed):
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - During installation, check "Add Miniconda to PATH"
   - Restart Command Prompt after installation

2. **For PowerShell users** (recommended):
   - Windows 10/11 includes PowerShell by default
   - For conda integration, run: `conda init powershell`

### macOS/Linux
- **Conda/Miniconda**: Install from https://docs.conda.io/en/latest/miniconda.html
- **Git**: Usually pre-installed

## Quick Start

### Windows Users

#### 1. One-Click Setup
```cmd
setup_windows.bat
```

This will check for conda installation and set up the environment automatically.

#### 2. Run the Complete Pipeline

**Option A: Command Prompt**
```cmd
autorun_preprocessing.bat
```

**Option B: PowerShell (Recommended)**
```powershell
.\autorun_preprocessing.ps1
```

#### 3. Run with Options

**Command Prompt:**
```cmd
autorun_preprocessing.bat --grayscale
autorun_preprocessing.bat --pattern "6mM"
autorun_preprocessing.bat --skip-bmp
autorun_preprocessing.bat --in-memory
autorun_preprocessing.bat --in-memory --grayscale --force
autorun_preprocessing.bat --help
```

**PowerShell:**
```powershell
.\autorun_preprocessing.ps1 -Grayscale
.\autorun_preprocessing.ps1 -Pattern "6mM"
.\autorun_preprocessing.ps1 -SkipBmp
.\autorun_preprocessing.ps1 -InMemory
.\autorun_preprocessing.ps1 -InMemory -Grayscale -Force
.\autorun_preprocessing.ps1 -Help
```

### macOS/Linux Users

#### 1. Run the Complete Pipeline
```bash
./autorun_preprocessing.sh
```

This will:
- Set up the conda environment automatically
- Convert all BMP files to TIFF format
- Apply non-local means filtering with your Avizo settings
- Generate a processing summary and log

#### 2. Run with Options
```bash
# Process only 6mM files
./autorun_preprocessing.sh --pattern "6mM"

# Skip BMP conversion, only filter existing TIFF files
./autorun_preprocessing.sh --skip-bmp

# Force reprocessing of all files
./autorun_preprocessing.sh --force

# Convert BMP files to grayscale TIFF
./autorun_preprocessing.sh --grayscale

# Extract specific channel (channel 4 = alpha or green fallback)
./autorun_preprocessing.sh --channel 4

# Use in-memory workflow (no intermediate TIFF files, saves 50% storage)
./autorun_preprocessing.sh --in-memory

# In-memory workflow with grayscale conversion
./autorun_preprocessing.sh --in-memory --grayscale --force

# Get help
./autorun_preprocessing.sh --help
```

## Pipeline Components

### 1. BMP to TIFF Converter (`bmp_to_tiff_converter.py`)
- Converts .bmp files to .tiff format with LZW compression
- Handles large scientific images safely
- Preserves original file structure
- **Grayscale conversion** option for reducing file size
- **Channel extraction** (R, G, B, Alpha, or custom channel 4)
- **Smart reconversion**: Automatically detects if existing TIFF files match requested format

### 2. Non-Local Means Filter (`nonlocal_means_filter.py`)
- Applies denoising filter with Avizo-matched settings:
  - **Search window**: 10 px
  - **Local neighborhood**: 3 pixels 
  - **Similarity value**: 0.8 (h=6.0)
- Uses tile-based processing for large images
- Maintains data type and dynamic range

### 3. In-Memory Workflow (`bmp_to_filtered_workflow.py`)
- **Single-step processing**: BMP → Filtered TIFF (no intermediate files)
- **Storage efficient**: Saves ~50% disk space by eliminating intermediate TIFF files
- **Memory optimized**: Uses tile-based processing for large scientific images
- **Full feature support**: Grayscale conversion, channel extraction, pattern filtering
- **Performance**: Processes 275MB images in ~3 seconds

#### When to Use In-Memory Workflow
- **Production environments** where storage is limited
- **Large datasets** where intermediate files consume too much space
- **Automated pipelines** where only final results are needed
- **Fast processing** where minimizing I/O is important

#### When to Use Traditional Workflow
- **Quality inspection** where intermediate files help with debugging
- **Development** where you want to examine conversion quality
- **Archival purposes** where intermediate steps need preservation

### 4. Quality Inspection Tools

#### Compare Raw vs Filtered Images
```bash
python raw_vs_filtered_inspector.py
```
Shows side-by-side comparisons with noise metrics.

#### Compare BMP vs TIFF Quality
```bash
python image_quality_inspector.py
```
Compares conversion quality between formats.

## Environment Setup

The pipeline uses a conda environment with compatible package versions:

### Automatic Setup
The autorun script handles environment creation automatically.

### Manual Setup
```bash
# Create environment from file
conda env create -f environment.yml

# Or create manually
conda create -n precipitation_data python=3.11 numpy=1.26 pillow matplotlib opencv -y

# Activate environment
conda activate precipitation_data
```

## File Structure

```
Precipitation_Data_test/
├── 3mM/                              # Dataset folders
│   ├── *.bmp                         # Original BMP files
│   ├── *.tiff                        # Converted TIFF files
│   └── *_nlm_filtered.tiff          # Filtered TIFF files
├── 6mM/
│   └── ...
├── autorun_preprocessing.sh          # Main pipeline script (macOS/Linux)
├── autorun_preprocessing.bat         # Main pipeline script (Windows CMD)
├── autorun_preprocessing.ps1         # Main pipeline script (Windows PowerShell)
├── setup_windows.bat                 # One-click Windows setup
├── bmp_to_tiff_converter.py         # BMP→TIFF converter
├── nonlocal_means_filter.py         # NLM filter
├── bmp_to_filtered_workflow.py      # In-memory BMP→Filtered TIFF workflow
├── raw_vs_filtered_inspector.py     # Quality comparison
├── image_quality_inspector.py       # Format comparison
├── environment.yml                   # Conda environment
└── preprocessing_YYYYMMDD_HHMMSS.log # Processing logs
```

## Processing Parameters

### Non-Local Means Filter Settings (Avizo Compatible)
- **Filter strength (h)**: 6.0 (equivalent to similarity 0.8)
- **Template size**: 3 pixels (local neighborhood)
- **Search window**: 10 pixels
- **Tile size**: 2048 pixels (for memory management)

### Customization
All scripts accept command-line parameters for fine-tuning:

```bash
# Custom filter strength
python nonlocal_means_filter.py --h 8.0

# Different window sizes
python nonlocal_means_filter.py --template-size 5 --search-size 8

# Process specific patterns
python nonlocal_means_filter.py --pattern "AfterFPN"

# BMP to TIFF conversion options
python bmp_to_tiff_converter.py --grayscale
python bmp_to_tiff_converter.py --channel 4

# In-memory workflow (no intermediate TIFF files)
python bmp_to_filtered_workflow.py 3mM 6mM --grayscale
python bmp_to_filtered_workflow.py --channel 4 --pattern "sample"
python bmp_to_filtered_workflow.py --h 8.0 --force
```

## Output

### Processing Logs
Each run generates a timestamped log file with:
- Processing times
- File counts and sizes
- Error messages and warnings
- Parameter settings used

### File Naming Convention
- Original: `dataset_file.bmp`
- Converted: `dataset_file.tiff` (traditional workflow only)
- Filtered: `dataset_file_nlm_filtered.tiff`

**Note**: The in-memory workflow (`--in-memory`) skips creating intermediate `.tiff` files and goes directly from BMP to filtered TIFF, saving approximately 50% disk space.

### Smart Conversion Logic
The BMP to TIFF converter intelligently handles existing files:

- **Format matching**: Checks if existing TIFF matches requested format (color vs grayscale)
- **Automatic reconversion**: If you run with `--grayscale` but existing TIFF is color, it will reconvert
- **Channel extraction**: If you run with `--channel 4` but existing TIFF is multi-channel, it will reconvert
- **Timestamp checking**: Always reconverts if source BMP is newer than existing TIFF

Examples:
```bash
# First run: creates color TIFF files
./autorun_preprocessing.sh

# Second run with --grayscale: detects color TIFFs and reconverts to grayscale
./autorun_preprocessing.sh --grayscale --force
```

## Troubleshooting

### Windows-Specific Issues

#### Conda Not Found
If you get "conda is not recognized":
1. Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
2. During installation, check "Add Miniconda to PATH"
3. Restart Command Prompt/PowerShell
4. Run `setup_windows.bat` again

#### PowerShell Execution Policy
If PowerShell blocks script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Conda Environment Activation Issues
If conda activate fails in PowerShell:
```powershell
conda init powershell
```
Then restart PowerShell and try again.

### General Issues

#### Environment Issues
If you encounter NumPy compatibility errors:
```bash
conda activate precipitation_data
conda install numpy=1.26 -y
```

#### Memory Issues
For very large images, adjust tile size:
```bash
python nonlocal_means_filter.py --tile-size 1024
```

#### Permission Issues (macOS/Linux)
Make sure the autorun script is executable:
```bash
chmod +x autorun_preprocessing.sh
```

## Performance

Processing times depend on image size and hardware:
- **Small images** (< 10 MB): ~10-30 seconds
- **Medium images** (10-100 MB): ~1-5 minutes  
- **Large images** (> 100 MB): ~5-30 minutes

The pipeline uses tile-based processing to handle images larger than available memory. 