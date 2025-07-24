# Precipitation Data Analysis Pipeline

A comprehensive pipeline for preprocessing precipitation imaging data from BMP files to filtered TIFF files using non-local means filtering with Avizo-matched parameters, plus machine learning classification of precipitation particle types using XGBoost.

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

# Keep BMP files in color (override default grayscale)
./autorun_preprocessing.sh --color

# Extract specific channel (channel 4 = alpha or green fallback)
./autorun_preprocessing.sh --channel 4

# Use in-memory workflow (no intermediate TIFF files, saves 50% storage)
./autorun_preprocessing.sh --in-memory

# In-memory workflow with color images (override default grayscale)
./autorun_preprocessing.sh --in-memory --color --force

# Get help
./autorun_preprocessing.sh --help
```

## Pipeline Components

### 1. BMP to TIFF Converter (`bmp_to_tiff_converter.py`)
- Converts .bmp files to .tiff format with LZW compression
- Handles large scientific images safely
- Preserves original file structure
- **Grayscale conversion by default** for optimal file size and processing
- **Color preservation** option (--color) when needed
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

### 4. Image Normalization (`image_normalization.py`)
- Normalizes images to match reference histogram characteristics
- **Multiple normalization methods**:
  - **`histogram_matching`**: Full histogram distribution matching (default)
  - **`peak_align`**: Align histogram peaks only (preserves spread)
  - **`peak_spread_align`**: Align both peaks and spread
  - **`min_max`**: Min-max range normalization
  - **`z_score`**: Mean and standard deviation matching
- **Visual comparison plots** with overlay histograms showing:
  - Peak positions and standard deviations
  - Before/after image comparison
  - Combined histogram view for easy comparison
- **Batch processing** of entire directories
- **Reference image selection** for consistent normalization across datasets

#### Usage Examples

```bash
# Basic histogram matching normalization (default)
python image_normalization.py

# Peak alignment only (preserves original spread characteristics)
python image_normalization.py --method peak_align

# Align both peaks and spread for complete histogram control
python image_normalization.py --method peak_spread_align

# Custom input/output directories and reference image
python image_normalization.py \
    --input_dir raw_images \
    --output_dir normalized_images \
    --reference_image reference.tiff \
    --method histogram_matching

# Other normalization methods
python image_normalization.py --method min_max      # Simple range scaling
python image_normalization.py --method z_score      # Statistical normalization
```

#### When to Use Different Methods

**Peak Alignment (`peak_align`)**
- Best for: Correcting lighting differences while preserving image contrast
- Preserves: Original histogram spread and shape
- Changes: Only the brightness levels to align peaks
- Use case: Batch processing where you want consistent brightness but natural contrast

**Peak + Spread Alignment (`peak_spread_align`)**
- Best for: Complete histogram standardization across all images
- Preserves: Overall histogram shape 
- Changes: Both brightness and contrast to match reference
- Use case: Machine learning preprocessing where consistent intensity distributions are critical

**Full Histogram Matching (`histogram_matching`)**
- Best for: Exact visual matching to reference image characteristics
- Preserves: Nothing (complete transformation)
- Changes: Entire intensity distribution to precisely match reference
- Use case: Visual processing where exact histogram replication is required

#### Output

The normalization process generates:
- **Normalized images** with `normalized_` prefix
- **Reference image copy** in output directory
- **Histogram comparison plots** showing transformation effects
- **Processing logs** with normalization statistics

### 5. Quality Inspection Tools

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

### 5.1 Streak/Artifact Investigation and Sensor Issues

#### Investigate Streaks and Regional Artifacts (`investigate_streaks.py`)

If you observe streaks, bands, or regional intensity artifacts in your processed data, use the `investigate_streaks.py` tool to diagnose the source:

```bash
python investigate_streaks.py --h5_file path/to/problematic.h5 --bmp_file path/to/sample.bmp --output_dir streak_analysis_output
```

- The tool analyzes the reference image, raw BMP, filtered, and normalized images for regional intensity changes (e.g. left/right or top/bottom differences).
- It saves comprehensive diagnostic plots in the specified output directory.
- It reports statistics such as 'Left vs Right difference' and highlights if the artifact is present in the raw data or introduced by processing.

**Typical findings:**
- If the reference image is uniform but the raw BMP shows a large left/right intensity difference, the artifact is likely due to sensor non-uniformity, lighting gradients, or calibration issues—not the pipeline.
- Artifacts present in the raw data will persist through filtering and normalization.

**Recommendations if artifacts are detected:**
1. **Check imaging system calibration** (sensor flat-field, gain, offset)
2. **Verify illumination uniformity** (avoid gradients)
3. **Consider flat-field correction** before running the pipeline
4. **Try a different reference image** for normalization
5. **Use the diagnostic plots** to guide troubleshooting

See the `streak_analysis_output/` directory for example output plots and statistics.

---

### 6. BMP to Filtered+Normalized HDF5 Pipeline (`bmp_to_filtered_normalized_hdf5.py`)
- **Purpose**: Recursively finds all BMP images in all folders under a specified root directory (use `--root`), applies non-local means filtering, then normalizes them using the 'peak_align' (peak shift) method with the default reference image, and saves the results as an HDF5 file per folder. No intermediate TIFFs are created.
- **Parallel Processing:** All BMPs in all folders are processed in parallel, maximizing CPU usage. The script displays a live progress bar for the overall processing. The reference image is loaded only once per worker process for efficiency, reducing I/O and memory usage (especially important for large reference images and datasets).
- **Automatic Metadata Extraction:** Output HDF5 files now include `Na2CO3_mM`, `CaCl_mM`, and `replicate` attributes if these can be parsed from the folder name (e.g., `2`5+100mM-0628-5-mix` → `Na2CO3_mM=2.5`, `CaCl_mM=100`, `replicate=5`).
- **Usage Example:**

```bash
python bmp_to_filtered_normalized_hdf5.py --input_root image \
    --reference_image /path/to/reference.bmp \
    --h 6.0 --template_size 3 --search_size 10 --max_workers 12
```
- **Options:**
  - `--input_root`: Root directory containing subfolders of BMP images (default: `image`)
  - `--reference_image`: Reference image for normalization (default: path to a BMP)
  - `--h`: Non-local means filter strength (default: 6.0)
  - `--template_size`: Template window size (default: 3)
  - `--search_size`: Search window size (default: 10)
  - `--max_workers`: Number of parallel workers (set to number of high-performance cores; default: all cores)
- **Output:**
  - For each folder, an HDF5 file named `<folder>_filtered_normalized_timeseries.h5` containing the filtered and normalized image stack, timestamps, and metadata.
- **Performance Note:** The script is optimized for high-performance systems and can take advantage of many CPU cores and large RAM. Use `--max_workers` to control CPU usage and avoid overloading your system.

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

## Additional Scripts

Below are additional scripts included in this repository, organized by use case. These tools support data analysis, model evaluation, feature engineering, quality inspection, and advanced investigation.

---

### 1. Data Analysis & Visualization

#### `analyze_timeseries.py`
- **Purpose:** Analyze HDF5 timeseries files, visualizing the first, middle, and last images, difference images, and histograms.
- **Features:**
  - Plots first, middle, last, and difference images
  - Shows histograms and mean/std intensity over time
  - Saves or displays comprehensive visualizations
- **Usage:**
  ```sh
  python analyze_timeseries.py --file path/to/timeseries.h5 [--save output.png]
  ```

#### `compare_feature_distributions.py`
- **Purpose:** Compare feature distributions between datasets or classes.
- **Features:**
  - Plots histograms, boxplots, or violin plots for selected features
  - Useful for data exploration and class comparison
- **Usage:**
  ```sh
  python compare_feature_distributions.py --file data.xlsx --feature Area
  ```

#### `create_simple_train_test_plot.py`, `create_train_test_plots.py`, `create_validation_plots.py`, `create_validation_visualizations.py`
- **Purpose:** Generate various plots for train/test splits and validation results.
- **Features:**
  - Visualize data splits, model performance, and validation metrics
  - Save plots as PNG files for reporting
- **Usage:**
  ```sh
  python create_simple_train_test_plot.py
  python create_train_test_plots.py
  python create_validation_plots.py
  python create_validation_visualizations.py
  ```

#### `clustering_error_analysis.py`
- **Purpose:** Perform clustering-based error analysis for XGBoost predictions.
- **Features:**
  - Identifies error patterns using clustering (KMeans, DBSCAN, etc.)
  - Visualizes clusters and error rates
  - Exports results to Excel and PNG
- **Usage:**
  ```sh
  python clustering_error_analysis.py
  ```

---

### 2. Model Training & Evaluation

#### `test_enhanced_model.py`
- **Purpose:** Test the enhanced model on test data and report performance.
- **Features:**
  - Loads trained model and test data
  - Reports accuracy, confusion matrix, and feature importance
- **Usage:**
  ```sh
  python test_enhanced_model.py
  ```

#### `improve_model_with_clustering.py`
- **Purpose:** Enhance model performance using clustering-based techniques.
- **Features:**
  - Integrates clustering into the training pipeline
  - May improve generalization and error analysis
- **Usage:**
  ```sh
  python improve_model_with_clustering.py
  ```

#### `excel_predictor.py`
- **Purpose:** Make predictions on new Excel data using a trained model.
- **Features:**
  - Loads model and scaler
  - Outputs predictions and probabilities to Excel
- **Usage:**
  ```sh
  python excel_predictor.py --data new_data.xlsx
  ```

---

### 3. Feature Engineering & Improvement

#### `improve_features.py`
- **Purpose:** Engineer and improve features for model training.
- **Features:**
  - Adds new features or transforms existing ones
  - Saves improved datasets for downstream use
- **Usage:**
  ```sh
  python improve_features.py --input data.xlsx --output improved_data.xlsx
  ```

---

### 4. Quality Inspection & Investigation

#### `raw_vs_filtered_inspector.py`
- **Purpose:** Compare raw and filtered images side by side for quality assessment.
- **Features:**
  - Displays or saves side-by-side comparisons
  - Supports custom subset extraction and batch mode
- **Usage:**
  ```sh
  python raw_vs_filtered_inspector.py --auto
  ```

#### `investigate_streaks.py`
- **Purpose:** Investigate streak artifacts in normalized precipitation data.
- **Features:**
  - Examines raw images, reference quality, filtering, and normalization steps
  - Visualizes each processing stage
- **Usage:**
  ```sh
  python investigate_streaks.py --h5_file path/to/problematic.h5
  python investigate_streaks.py --bmp_folder path/to/folder --reference_image path/to/ref.bmp
  ```

#### `uncertainty_quantification.py`
- **Purpose:** Quantify prediction uncertainty for model outputs.
- **Features:**
  - Computes and visualizes uncertainty metrics
  - Useful for model evaluation and reporting
- **Usage:**
  ```sh
  python uncertainty_quantification.py
  ```

---

### 5. Utility Scripts

These scripts are already documented in detail above, but for completeness:
- `check_and_clean_h5.py`: HDF5 integrity checker and cleaner
- `inspect_hdf5.py`: HDF5 structure/content inspector
- `images_to_hdf5.py`: Converts filtered TIFFs to HDF5 timeseries
- `image_quality_inspector.py`: Compares BMP and TIFF image quality
- `bmp_to_tiff_converter.py`, `nonlocal_means_filter.py`, `bmp_to_filtered_workflow.py`, `image_normalization.py`: Core pipeline utilities

---

For more details on each script, see the script's docstring or run with `--help` for command-line options.

## File Structure

```
Precipitation_Data_test/
├── 3mM/                              # Dataset folders
│   ├── *.bmp                         # Original BMP files
│   ├── *.tiff                        # Converted TIFF files
│   └── *_nlm_filtered.tiff          # Filtered TIFF files
├── 6mM/
│   └── ...
├── training_data/                    # ML training data
│   ├── 6mM-10-label-train.xlsx     # Enhanced labeled particle data (6mM)
│   └── 3mM-4-label-train.xlsx      # Enhanced labeled particle data (3mM)
├── models/                           # Saved trained models
│   ├── latest_model.joblib          # Most recent XGBoost model
│   ├── latest_scaler.joblib         # Feature scaler
│   ├── latest_label_encoder.joblib  # Class label encoder
│   ├── latest_feature_names.joblib  # Feature names
│   └── latest_metadata.joblib       # Model metadata
├── autorun_preprocessing.sh          # Main pipeline script (macOS/Linux)
├── autorun_preprocessing.bat         # Main pipeline script (Windows CMD)
├── autorun_preprocessing.ps1         # Main pipeline script (Windows PowerShell)
├── setup_windows.bat                 # One-click Windows setup
├── bmp_to_tiff_converter.py         # BMP→TIFF converter
├── nonlocal_means_filter.py         # NLM filter
├── bmp_to_filtered_workflow.py      # In-memory BMP→Filtered TIFF workflow
├── excel_xgboost_classifier.py      # XGBoost particle classifier
├── excel_predictor.py               # Prediction script for new data
├── run_classification_demo.py       # ML classification demo script
├── image_normalization.py           # Image histogram normalization
├── images_for_normalisation/        # Input images for normalization
├── normalized_images/               # Normalized output images
├── raw_vs_filtered_inspector.py     # Quality comparison
├── image_quality_inspector.py       # Format comparison
├── images_to_hdf5.py                # Time series HDF5 converter
├── inspect_hdf5.py                  # HDF5 file inspector
├── environment.yml                   # Conda environment
├── requirements.txt                  # Python dependencies
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

# BMP to TIFF conversion options (grayscale is now default)
python bmp_to_tiff_converter.py --color          # Keep in color
python bmp_to_tiff_converter.py --channel 4      # Extract specific channel

# In-memory workflow (no intermediate TIFF files, grayscale by default)
python bmp_to_filtered_workflow.py 3mM 6mM --color     # Keep in color
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

## 🧠 Machine Learning Classification

### XGBoost Precipitation Particle Classification

The pipeline includes a machine learning component that uses XGBoost to classify precipitation particle types based on morphological and intensity features extracted from the processed images.

#### Features
- **Multi-class classification** of precipitation particle types (classes 1-4)
- **Comprehensive feature engineering** from particle properties:
  - **Shape descriptors**: Area, Major/Minor axes, Eccentricity, Circularity, ConvexArea, Extent, Perimeter, Major_Minor_ratio
  - **Intensity statistics**: Gray_ave, Gray_var, MeanIntensity, Gray_skew, Gray_kur (skewness, kurtosis)
  - **Distance measures**: dis, dis_normal
  - **Smart exclusions**: Removes spatial coordinates, pixel-level data, and redundant features
- **Hyperparameter tuning** with GridSearchCV for optimal performance
- **Anti-overfitting measures** to ensure robust generalization:
  - Automatic feature selection using ANOVA F-test (reduces features from 17→12 by default)
  - Regularized hyperparameters (L1/L2 regularization, min samples per leaf)
  - Conservative model architecture (fewer trees, shallower depth, lower learning rates)
  - Aggressive subsampling for better generalization
- **Automatic class balancing** with multiple resampling techniques:
  - SMOTE oversampling (default)
  - Random undersampling
  - SMOTE + Edited Nearest Neighbours
  - Class weight balancing
- **Comprehensive evaluation** with confusion matrices and feature importance
- **Visualization** of results with publication-ready plots
- **Cross-validation** for robust model assessment using balanced accuracy
- **Model persistence** - automatically saves trained models for future use
- **Prediction pipeline** - easy-to-use script for classifying new data

#### Quick Start

```bash
# Activate environment
conda activate precipitation_data

# Option 1: Run the demo script (recommended for first-time users)
python run_classification_demo.py

# Option 2: Run XGBoost classification directly (uses SMOTE balancing by default)
python excel_xgboost_classifier.py --save-plots

# Use different balancing methods
python excel_xgboost_classifier.py --balance-method smote          # SMOTE oversampling (default)
python excel_xgboost_classifier.py --balance-method class_weights  # Class weight balancing
python excel_xgboost_classifier.py --balance-method undersampling  # Random undersampling
python excel_xgboost_classifier.py --balance-method none           # No balancing

# Control anti-overfitting features
python excel_xgboost_classifier.py --max-features 15              # Keep more features (default: 12)
python excel_xgboost_classifier.py --no-feature-selection         # Use all 17 features
python excel_xgboost_classifier.py --max-features 8               # More aggressive feature reduction

# Use custom Excel files
python excel_xgboost_classifier.py --files data1.xlsx data2.xlsx

# Customize training parameters
python excel_xgboost_classifier.py --test-size 0.3 --target-column "particle_type"
```

#### Input Data Format

The classifier expects Excel files with tabular data containing:
- **Feature columns**: Numeric measurements (Area, Eccentricity, etc.)
- **Target column**: Class labels for particle types (default: "type")
- **Automatic handling** of missing values and feature scaling

#### Output

The script provides:
- **Model performance metrics** (accuracy, precision, recall, F1-score)
- **Feature importance ranking** showing which measurements are most predictive
- **Confusion matrix** for detailed classification analysis
- **Visualization plots** saved as high-resolution PNG files
- **Cross-validation scores** for model reliability assessment

#### Example Output

```
============================================================
CLASSIFICATION COMPLETE!
============================================================
✓ Final Test Accuracy: 0.9234
✓ Model successfully trained on 541 samples  
✓ Used 15+ features including: ['Area', 'Major', 'Minor', 'Eccentricity', 'Circularity', 'ConvexArea', 'Extent', 'Perimeter', 'MeanIntensity', 'Gray_ave', 'Gray_var', 'Gray_skew', 'Gray_kur', 'dis', 'dis_normal', 'Major_Minor_ratio']
✓ Class mapping: {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3}
✓ Model components saved to: models/

📋 To use this model for predictions:
  python excel_predictor.py --data new_data.xlsx
```

#### Advanced Usage

```bash
# Custom hyperparameters and evaluation
python excel_xgboost_classifier.py \
    --files training_data/6mM-label.xlsx training_data/3mM-label.xlsx \
    --target-column "type" \
    --test-size 0.25 \
    --random-seed 123 \
    --save-plots

# Help and options
python excel_xgboost_classifier.py --help
```

#### Making Predictions on New Data

Once you've trained a model, you can easily classify new, unlabeled data:

```bash
# Basic prediction on new Excel file
python excel_predictor.py --data new_particles.xlsx

# Custom output file and show more results
python excel_predictor.py --data new_data.xlsx --output my_predictions.xlsx --show-top 20

# Use a specific trained model
python excel_predictor.py --data new_data.xlsx --model-dir custom_models/
```

**The prediction script will:**
- Load your trained model automatically
- Preprocess the new data using the same steps as training
- Generate predictions with confidence scores
- Create probability estimates for each class
- Save results to an Excel file with detailed output
- Display the most confident predictions

**Output includes:**
- Original data plus prediction columns
- `predicted_class`: The predicted particle type
- `confidence`: Prediction confidence (0-1 scale)
- `prob_class_X`: Probability for each class
- Data sorted by confidence (most confident first)

#### Class Balancing Methods

The classifier automatically handles class imbalance using several techniques:

**SMOTE (Synthetic Minority Oversampling Technique)** - *Default*
- Creates synthetic samples for minority classes
- Preserves original data while adding balanced synthetic examples
- Best for: Most cases, especially with continuous features
- Pros: Maintains data patterns, no information loss
- Cons: Slightly increases dataset size

**Class Weights**
- Assigns higher weights to minority classes during training
- No data modification, just changes learning emphasis
- Best for: When you want to keep original data unchanged
- Pros: Fast, no synthetic data, maintains dataset size
- Cons: May not work as well with severe imbalance

**Random Undersampling**
- Removes samples from majority classes to balance dataset
- Reduces dataset size to match minority class
- Best for: Large datasets where losing data is acceptable
- Pros: Fast, reduces memory usage
- Cons: Information loss, may hurt performance

**SMOTE + Edited Nearest Neighbours (SMOTEENN)**
- Combines SMOTE oversampling with cleaning techniques
- Removes borderline/noisy samples after SMOTE
- Best for: Noisy datasets with class overlap
- Pros: High-quality balanced data
- Cons: More processing time, may remove useful samples

**None**
- No balancing applied, uses original class distribution
- Best for: Already balanced datasets or when imbalance is desired
- Use this if: Your dataset is naturally balanced or represents true population

#### Overfitting Prevention

The classifier includes comprehensive anti-overfitting measures to ensure models generalize well to new data:

**Automatic Feature Selection**
- Uses ANOVA F-test to rank feature importance for classification
- Reduces features from 17 to 12 by default (retains most informative features)
- Removes redundant features like Gray_skew_abs, MeanIntensity, Minor axis
- Command-line control: `--max-features N` or `--no-feature-selection`

**Regularized Model Architecture**
- Conservative hyperparameter ranges to prevent overfitting:
  - Fewer trees: 50-150 estimators (vs 100-1000 in standard setups)
  - Shallower depth: 2-4 levels (vs 3-10)
  - Lower learning rates: 0.01-0.1 (vs 0.1-0.3)
  - L1/L2 regularization: alpha=0-1, lambda=1-5
  - Minimum samples per leaf: 1-5
- Aggressive subsampling: 60-90% of samples and features per tree

**Cross-Validation Strategy**
- 5-fold stratified cross-validation maintains class balance in each fold
- Balanced accuracy metric better handles class imbalance
- Grid search explores 6,561 parameter combinations systematically

**Early Stopping Protection**
- Removes early stopping from GridSearchCV to prevent validation dataset conflicts
- Relies on regularization parameters instead of training iteration limits
- More stable hyperparameter tuning process

### Data Requirements

For machine learning classification, ensure your Excel files contain:
1. **Consistent column names** across all files
2. **Numeric feature columns** (no text except in identifier columns)
3. **Complete target labels** (no missing values in classification column)
4. **Sufficient samples** per class (minimum 10-20 samples recommended)

### Integration with Image Processing

The classification component is designed to work with particle measurement data extracted from the processed images:

1. **Image Processing**: Use the main pipeline to convert and filter images
2. **Feature Extraction**: Extract particle measurements using image analysis software
3. **Data Export**: Save measurements to Excel format
4. **Classification**: Run the XGBoost classifier to predict particle types 

# Automated Training & Validation Pipeline

## Enhanced Features

- **Advanced Feature Engineering**: 9 new engineered features to improve classification accuracy
- **Clustering-Based Model Enhancement**: Uses KMeans clustering to identify error-prone regions and creates 8 cluster-aware features
- **Ensemble Learning**: Combines multiple models (XGBoost Enhanced, Random Forest Enhanced, XGBoost Conservative) with soft voting
- **Smart Data Augmentation**: Applies SMOTEENN resampling focused on confused class pairs (especially Class 1 vs Class 4)
- **Comprehensive Validation**: Tests on ALL datasets in `data_for_classification/` with detailed metrics and visualizations
- **Enhanced Logging System**: Comprehensive logging with TeeLogger class that captures all terminal output to timestamped log files
- **Results-Only Model Saving**: Enhanced models save only to results directory to avoid duplication
- **Windows Compatibility**: Emoji-free output using bracket notation for cross-platform compatibility
- **Direct Method Processing**: Refactored validation script generation to use direct method calls for better debugging

## Overview

The pipeline now includes a fully automated end-to-end training, validation, and reporting workflow for precipitation data classification. This is orchestrated by `autorun_training.py` and associated scripts. The workflow covers:
- Data discovery (auto-detects all Excel files in `training_data/` and `data_for_classification/`)
- Feature engineering (adds 9 new features for improved classification)
- Enhanced ensemble model training (with clustering, resampling, and soft-voting ensemble)
- Automated validation on **all** datasets in `data_for_classification/` (not just val3/val6)
- Comprehensive, publication-ready visualizations
- Timestamped results folders for reproducibility
- **Comprehensive logging** with all terminal output captured to timestamped log files

## Quick Start

1. **Prepare your data**: Place Excel files in appropriate directories:
   - Training data: `training_data/` (e.g., train3.xlsx, train6.xlsx)
   - Test/validation data: `data_for_classification/` (e.g., val3.xlsx, val6.xlsx, etc.)

2. **Run the automated pipeline**:
   ```bash
   python autorun_training.py
   ```

3. **Check results**: Look in the timestamped results folder (e.g., `results_20240115_143022/`) for:
   - Enhanced model: `models/enhanced_ensemble_model.joblib`
   - Validation plots: `plots/enhanced_model_validation_results.png`
   - Performance report: `reports/training_pipeline_report.txt`
   - **Comprehensive logs**: `logs/` directory with timestamped logs for all operations

## How It Works

1. **Run the pipeline**:
   ```bash
   conda activate precipitation_data
   python autorun_training.py
   # or with options:
   python autorun_training.py --skip-feature-eng --skip-training
   ```
2. **All results are saved in a new timestamped folder** (e.g., `results_YYYYMMDD_HHMMSS/`).
3. **Validation is fully automated**:
   - Every Excel file in `data_for_classification/` is processed and evaluated.
   - No need to manually specify validation files; new files are automatically included.
4. **All terminal output is logged**: Comprehensive logging captures all output to timestamped files for debugging and record-keeping.

## Pipeline Components

### 1. Feature Engineering (`improve_features.py`)
Creates 9 new engineered features:
- **Shape_Complexity**: Measures shape intricacy combining normalized extent and circularity
- **Area_Perimeter_Ratio**: Normalized compactness measure
- **Convex_Efficiency**: Efficiency of shape filling its convex hull
- **Eccentricity_Robust**: Clipped eccentricity to handle outliers
- **Distance_Ratio**: Ratio of raw to normalized distance
- **Distance_Interaction**: Interaction between distances
- **Intensity_Stability**: Stability of intensity relative to variance
- **Comprehensive_Shape**: Combined shape metric
- **Class4_Discriminator**: Feature to separate Class 4 from Class 1

**Note**: Files are correctly named with `test_data_improved.xlsx` (not validation) since this data comes from the `data_for_classification/` folder which contains test datasets.

### 2. Enhanced Model Training (`improve_model_with_clustering.py`)
- **Clustering Analysis**: Uses KMeans with 2 clusters to identify error-prone regions
- **8 Cluster-Aware Features**:
  - Distance to cluster centers (2 features)
  - Cluster membership probabilities (2 features) 
  - Class 4 vs Class 1 discriminators (3 features)
  - High-error region indicator (1 feature)
- **Targeted Resampling**: Applies SMOTEENN to balance classes
- **Ensemble Model**: Soft-voting ensemble of 3 models with different hyperparameters
- **Results-Only Saving**: Enhanced models save only to results directory to avoid duplication

### 3. Comprehensive Validation Testing
- Tests enhanced model on **all test datasets** in `data_for_classification/`
- Generates individual performance plots for each dataset
- Creates comprehensive overview plot combining all results
- Computes detailed metrics: accuracy, precision, recall, F1, confidence statistics
- **Direct Method Processing**: Uses direct class method calls instead of dynamic script generation for better debugging

### 4. Automated Pipeline (`autorun_training.py`)
- **Data Discovery**: Automatically finds and validates Excel files
- **Sequential Processing**: Runs feature engineering → model training → validation testing
- **Results Management**: Creates timestamped output directories
- **Report Generation**: Comprehensive text reports with all metrics and analysis
- **Enhanced Logging**: Comprehensive logging system with TeeLogger class captures all terminal output
- **Windows Compatibility**: Uses bracket notation (e.g., [START], [SUCCESS]) instead of emojis for cross-platform compatibility

### Individual Scripts

#### `improve_features.py`
Standalone feature engineering script:
```bash
python improve_features.py
```
Generates `train_improved.xlsx` and `test_data_improved.xlsx` (correctly named for test data)

#### `improve_model_with_clustering.py`
Enhanced model training with clustering:
```bash
python improve_model_with_clustering.py
```
Requires existing base model components in `models/` directory

#### `create_simple_train_test_plot.py`
Create training vs test performance visualization:
```bash
python create_simple_train_test_plot.py
```
Generates `train_test_performance.png` with Windows-compatible output formatting

## Key Improvements Over Standard Approach

1. **Cluster-Aware Feature Engineering**: Identifies and targets error-prone regions
2. **Ensemble Learning**: Combines multiple model strategies for robust predictions
3. **Class-Specific Optimization**: Focuses on Class 4 vs Class 1 confusion (main error source)
4. **Comprehensive Validation**: Tests on all available datasets with detailed analysis
5. **Automated Pipeline**: End-to-end automation with proper results management
6. **Advanced Resampling**: Uses SMOTEENN for intelligent data augmentation
7. **Feature Selection**: Automatically selects most informative features
8. **Enhanced Logging System**: Comprehensive logging with TeeLogger class captures all terminal output to timestamped files
9. **Results-Only Model Saving**: Enhanced models save only to results directory to avoid duplication in base models folder
10. **Direct Method Processing**: Refactored validation processing from dynamic script generation to direct method calls for better debugging
11. **Cross-Platform Compatibility**: Windows-compatible output using bracket notation instead of emojis
12. **Improved File Naming**: Clear distinction between training, test, and validation data with appropriate file names

## Validation Visualizations

- **Main overview plot**: `enhanced_model_validation_results.png`
  - Shows validation accuracy for **all** datasets and the combined set
  - Per-class accuracy, confidence distributions, correct vs incorrect confidence
  - Class distribution pie chart
  - **Dataset size comparison bar chart**
  - Performance overview and summary
  - **No confusion matrices** (these are only in the individual dataset plots)
  - **Plots are only saved to file, not displayed**

- **Individual plots**: For each validation dataset (e.g., `enhanced_model_3mM-5_finalData_results.png`), a dedicated plot is generated with:
  - Confusion matrix
  - Confidence distribution
  - Class-wise performance

- **Train/test performance plot**: `train_test_performance.png`

## Outputs

All outputs are saved in the timestamped results folder:
- **models/**: Enhanced ensemble model, cluster model, scaler, selector, and metadata
- **plots/**:
  - `enhanced_model_validation_results.png` (overview for all validation datasets)
  - `enhanced_model_<dataset>_results.png` (individual plots for each validation set; one for every validation file, automatically discovered)
  - `train_test_performance.png` (train/test split performance)
- **reports/**: Pipeline report, test results Excel, predictions for each validation set
- **data/**: Improved train and validation datasets
- **logs/** - **Comprehensive logging output**:
  - `feature_engineering_YYYYMMDD_HHMMSS.log` - Feature engineering process logs
  - `model_training_YYYYMMDD_HHMMSS.log` - Model training logs
  - `validation_testing_YYYYMMDD_HHMMSS.log` - Validation testing logs
  - `train_test_plotting_YYYYMMDD_HHMMSS.log` - Plotting process logs
  - Corresponding `_errors.log` files for any error output

**Note:** Plots are only saved to file and are not displayed interactively.

## Full Workflow Details

For a detailed, step-by-step explanation of the pipeline, feature engineering, model, and validation process, see [`autorun_training_workflow.md`](./autorun_training_workflow.md). 

## HDF5 File Integrity Checker: `check_and_clean_h5.py`

This utility script helps you maintain the integrity of your HDF5 dataset folders by:

- **Automatically deleting all `._*.h5` files** (macOS metadata) it finds during the scan.
- **Recursively checking all `.h5` files** in a specified root directory (default: `image/`) for corruption (cannot be opened or missing the `images` dataset).
- **Listing all problematic/corrupted files** at the end of the scan.
- **Optionally deleting problematic/corrupted files** if the `--delete` flag is provided.

### Usage

- **List problematic files (no deletion except ._ files):**
  ```sh
  python check_and_clean_h5.py --root image
  ```
- **List and delete problematic files:**
  ```sh
  python check_and_clean_h5.py --root image --delete
  ```

### Output
- The script prints a summary of how many files were checked, how many were good, how many were problematic, and how many `._*.h5` files were deleted.
- If the `--delete` flag is used, it will also delete and report each problematic `.h5` file.
- A list of all problematic files is printed at the end for your review.

### Notes
- The script is safe to run multiple times; it will only delete `._*.h5` files and, if requested, corrupted `.h5` files.
- It is recommended to run this script after copying or generating new HDF5 files, especially if you use macOS or external drives. 

## HDF5 Inspector: `inspect_hdf5.py`

This utility script allows you to inspect the structure and contents of HDF5 files created by the pipeline (e.g., *_filtered_timeseries.h5 or *_filtered_normalized_timeseries.h5 files).

- **Lists all file-level attributes** (e.g., chemistry, replicate, creation time).
- **Displays all datasets** in the file, including:
  - Dataset name (e.g., /images, /times, /timestamps, /filenames)
  - Shape, data type, and number of elements
  - For images: memory size and pixel value range
  - For times: time range, duration, and time intervals
  - For timestamps: first and last timestamp
  - For filenames: all included file names
- **Shows file size, uncompressed size, and compression ratio**

### Usage

- **Inspect all *_filtered_timeseries.h5 files in the current directory and subdirectories:**
  ```sh
  python inspect_hdf5.py
  ```
- **Inspect a specific HDF5 file:**
  ```sh
  python inspect_hdf5.py path/to/your_file_filtered_timeseries.h5
  ```
- **Inspect multiple files at once:**
  ```sh
  python inspect_hdf5.py file1.h5 file2.h5
  ```

### Output
- The script prints a detailed summary for each file, including all attributes and datasets, and a summary of file size and compression.
- Useful for verifying data integrity, structure, and metadata before further analysis or sharing. 