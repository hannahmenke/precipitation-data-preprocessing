# Automated Training Pipeline Workflow

## Recent Enhancements

This document reflects the latest improvements made to the automated training pipeline:

### Key Improvements
- **Enhanced Logging System**: Comprehensive logging with TeeLogger class captures all terminal output to timestamped log files
- **Results-Only Model Saving**: Enhanced models save only to results directory to avoid duplication
- **Direct Method Processing**: Refactored validation script generation to use direct method calls for better debugging
- **Windows Compatibility**: Emoji-free output using bracket notation for cross-platform compatibility
- **Improved File Naming**: Clear distinction between training and test data with appropriate file names
- **Better Error Handling**: Comprehensive error logging and improved debugging capabilities

## Overview
The `autorun_training.py` script automates the end-to-end training pipeline for classifying precipitation data. It handles data discovery, feature engineering, model training with an enhanced cluster-based ensemble, validation testing, and report generation. The pipeline uses improved features and an ensemble model to achieve better performance, especially on challenging classes.

## Inputs (Command-Line Arguments)
The script accepts the following optional arguments:

- `--max-features N` (default: 20): Maximum number of features to select during training.
- `--random-seed N` (default: 42): Random seed for reproducibility across the pipeline.
- `--output-dir DIR` (default: timestamped folder like 'results_YYYYMMDD_HHMMSS'): Directory where all outputs (models, plots, reports, logs) will be saved. If not specified, a new timestamped folder is created for each run.
- `--skip-feature-eng`: If set, skips feature engineering if improved data already exists.
- `--skip-training`: If set, skips model training if enhanced model already exists.
- `--skip-validation`: If set, skips validation testing.

### Data Inputs
- Training data: Excel files in `training_data/` directory (e.g., train3.xlsx, train6.xlsx).
- Test data: Excel files in `data_for_classification/` directory (e.g., val3.xlsx, val6.xlsx, 3mM-5_finalData.xlsx, etc.).

**Note**: Files in `data_for_classification/` are correctly referred to as test data since they are used for final model evaluation, not validation during training.

The data should contain features like Area, Major, Eccentricity, etc., and a 'type' column for labels (classes 1-4).

## Pipeline Steps
1. **Setup Output Directory**: Creates a timestamped results folder (or uses the specified one) with subdirectories for models, plots, reports, logs, and data. **Enhanced logging system** is initialized to capture all terminal output to timestamped log files.

2. **Data Discovery**: Scans training and test directories for Excel files, validates them, and reports sample counts, classes, and columns. Uses Windows-compatible bracket notation for all output messages.

3. **Feature Engineering** (via `improve_features.py`):
   - Loads raw data.
   - Creates 9 new engineered features to improve classification (see 'New Features' section below).
   - Analyzes feature importance and suggests improvements.
   - Saves improved datasets: `train_improved.xlsx` and `test_data_improved.xlsx` (correctly named since data comes from `data_for_classification/` test folder).

4. **Enhanced Model Training** (via `improve_model_with_clustering.py`):
   - Loads improved training data and base model components.
   - Performs clustering (KMeans with 2 clusters) to identify error-prone regions.
   - Creates 8 cluster-aware features (distances to clusters, soft probabilities, class discriminators, error indicators).
   - Applies targeted resampling (SMOTEENN) to balance classes, focusing on confused pairs like Class 1 and 4.
   - Trains an ensemble model (see 'Ensemble Model' section below).
   - **Saves the enhanced model only to results directory** (`results*/models/enhanced_ensemble_model.joblib`) to avoid duplication in base models folder.

5. **Validation Testing & Visualization** (using direct method calls for better debugging):
   - Tests the enhanced model on **all test datasets** found in `data_for_classification/` (auto-discovered, not hardcoded).
   - Uses **direct class method calls** instead of dynamic script generation for better debugging and maintainability.
   - Computes metrics: accuracy, precision, recall, F1, confidence statistics.
   - Generates predictions and saves to Excel files.
   - **Creates a comprehensive validation overview plot** (`enhanced_model_validation_results.png`) that includes ALL test datasets (not just val3/val6/combined). The plot now features:
     - Validation accuracy for each dataset and the combined set
     - Per-class accuracy, confidence distributions, correct vs incorrect confidence
     - Class distribution pie chart
     - **Dataset size comparison bar chart**
     - Performance overview and summary
     - **No confusion matrices** (these are only in the individual dataset plots)
   - **Creates individual plots** for each test dataset (e.g., `enhanced_model_<dataset>_results.png` for every test file), each including a confusion matrix, confidence distribution, and class-wise performance. All are automatically copied to the results folder.
   - **Plots are only saved to file, not displayed in the viewer** (for batch/automated use).
   - Also creates train/test performance plots (via `create_simple_train_test_plot.py`) with Windows-compatible output formatting.
   - **Enhanced models save only to results directory** to avoid duplication in base models folder.

6. **Report Generation**: Creates a detailed text report summarizing all steps, data info, validation results, and performance improvements. Saved as `training_pipeline_report.txt` in the output directory.

7. **Comprehensive Logging**: All terminal output is captured using TeeLogger class and saved to timestamped log files in the `logs/` directory for debugging and record-keeping.

8. **Windows Compatibility**: All output uses bracket notation (e.g., [START], [SUCCESS], [ERROR]) instead of emojis for cross-platform compatibility.

## New Features
The feature engineering is implemented in `improve_features.py` via the `create_improved_features` function. It adds 9 new features to the dataset based on error analysis. Below are the exact calculations and code snippets for each:

1. **Shape_Complexity**: Measures shape intricacy by combining normalized extent and circularity.
   - Formula: `extent_norm * (1 - circularity_norm)`
   - Code:
     ```python
     extent_norm = (df['Extent'] - df['Extent'].min()) / (df['Extent'].max() - df['Extent'].min())
     circularity_norm = (df['Circularity'] - df['Circularity'].min()) / (df['Circularity'].max() - df['Circularity'].min())
     df['Shape_Complexity'] = extent_norm * (1 - circularity_norm)
     ```

2. **Area_Perimeter_Ratio**: Normalized compactness measure.
   - Formula: `Area / (Perimeter + epsilon)` (epsilon=1e-6 to avoid division by zero)
   - Code:
     ```python
     df['Area_Perimeter_Ratio'] = df['Area'] / (df['Perimeter'] + 1e-6)
     ```

3. **Convex_Efficiency**: Efficiency of shape filling its convex hull.
   - Formula: `Area / (ConvexArea + epsilon)`
   - Code:
     ```python
     df['Convex_Efficiency'] = df['Area'] / (df['ConvexArea'] + 1e-6)
     ```

4. **Eccentricity_Robust**: Clipped eccentricity to handle outliers.
   - Formula: `clip(Eccentricity, 0, 1)`
   - Code:
     ```python
     df['Eccentricity_Robust'] = np.clip(df['Eccentricity'], 0, 1)
     ```

5. **Distance_Ratio**: Ratio of raw to normalized distance.
   - Formula: `dis / (dis_normal + epsilon)`
   - Code:
     ```python
     df['Distance_Ratio'] = df['dis'] / (df['dis_normal'] + 1e-6)
     ```

6. **Distance_Interaction**: Interaction between distances.
   - Formula: `dis * dis_normal`
   - Code:
     ```python
     df['Distance_Interaction'] = df['dis'] * df['dis_normal']
     ```

7. **Intensity_Stability**: Stability of intensity relative to variance.
   - Formula: `MeanIntensity / (Gray_var + epsilon)` (uses 'MeanIntensity' or fallback to 'Gray_ave')
   - Code:
     ```python
     gray_mean = df['MeanIntensity'] if 'MeanIntensity' in df.columns else df.get('Gray_ave', 0)
     df['Intensity_Stability'] = gray_mean / (df['Gray_var'] + 1e-6)
     ```

8. **Comprehensive_Shape**: Combined shape metric.
   - Formula: `Major_Minor_ratio * Circularity * (1 - Eccentricity_Robust)`
   - Code:
     ```python
     df['Comprehensive_Shape'] = (df['Major_Minor_ratio'] * df['Circularity'] * (1 - df['Eccentricity_Robust']))
     ```

9. **Class4_Discriminator**: Feature to separate Class 4 from Class 1.
   - Formula: `Extent * Circularity * Perimeter`
   - Code:
     ```python
     df['Class4_Discriminator'] = (df['Extent'] * df['Circularity'] * df['Perimeter'])
     ```

These features are added to the DataFrame after loading raw data, and missing values are filled with medians.

## Ensemble Model
The enhanced model is created in `improve_model_with_clustering.py` using a `ClusterBasedModelImprover` class. It involves clustering, feature augmentation, resampling, and ensemble training. Here's the detailed breakdown:

### 1. Clustering
- Uses KMeans with 2 clusters on selected features.
- Code:
  ```python
  self.cluster_model = KMeans(n_clusters=2, random_state=self.random_state)
  cluster_labels = self.cluster_model.fit_predict(X_train_selected)
  ```

### 2. Cluster-Aware Features (8 features)
These are created in `create_cluster_aware_features`:
- **Distance to cluster centers** (2 features): Euclidean distance to each cluster center.
  - Formula: `sqrt(sum((x - center)^2))` for each center.
- **Cluster membership probabilities** (2 features): Soft probabilities from GaussianMixture.
  - Code:
    ```python
    gmm = GaussianMixture(n_components=2, random_state=self.random_state)
    gmm.fit(X_train_selected)
    cluster_probs = gmm.predict_proba(X_train_selected)
    ```
- **Class 4 vs Class 1 discriminators** (3 features): Distances to class means and their difference.
  - Formula: `dist_to_class_4 = sqrt(sum((x - class_4_mean)^2))`, similarly for class_1, and difference.
  - Code:
    ```python
    class_4_mean = np.mean(X_train_selected[class_4_mask], axis=0)
    dist_to_class_4 = np.sqrt(np.sum((X_train_selected - class_4_mean) ** 2, axis=1))
    ```
- **High-error region indicator** (1 feature): Binary indicator for error-prone cluster (cluster 0).
  - Code:
    ```python
    error_indicator = (cluster_labels == 0).astype(float)
    ```

### 3. Data Augmentation
- Combines selected features with cluster features.
- Applies SMOTEENN for resampling.
- Code:
  ```python
  X_enhanced = np.column_stack([X_train_selected, cluster_features])
  smote_enn = SMOTEENN(random_state=self.random_state)
  X_resampled, y_resampled = smote_enn.fit_resample(X_enhanced, y_train)
  ```

### 4. Ensemble Components
Soft-voting ensemble of three models trained on enhanced data:
- **XGBoost Enhanced**:
  ```python
  XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, min_child_weight=3, random_state=42, eval_metric='mlogloss')
  ```
- **Random Forest Enhanced**:
  ```python
  RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42, class_weight='balanced')
  ```
- **XGBoost Conservative**:
  ```python
  XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.5, reg_lambda=2, min_child_weight=5, random_state=42, eval_metric='mlogloss')
  ```
- Ensemble:
  ```python
  VotingClassifier(estimators=[('xgb_enhanced', xgb_enhanced), ('rf_enhanced', rf_enhanced), ('xgb_conservative', xgb_conservative)], voting='soft')
  ```

### 5. Evaluation
- Cross-validation on enhanced data.
- Comparison with base model on test set.
- **All models save only to results directory** to avoid duplication in base models folder.

This setup can be reproduced by following the code structure in the respective scripts, ensuring the base model and improved data are available.

## Outputs
All outputs are saved in the timestamped results folder (e.g., `results_YYYYMMDD_HHMMSS`):

- **models/**: Enhanced ensemble model, cluster model, metadata.
- **plots/**: 
  - **Train/test performance plot** (`train_test_performance.png`)
  - **Validation overview plot** (`enhanced_model_validation_results.png`):
    - Includes ALL validation datasets (auto-discovered)
    - Shows accuracy, per-class accuracy, confidence, class distribution, dataset sizes, and summary
    - **No confusion matrices** (see below)
  - **Individual validation plots** (e.g., `enhanced_model_<dataset>_results.png` for every validation file):
    - Each includes a confusion matrix, confidence distribution, and class-wise performance for that dataset. All are automatically discovered and copied.
- **reports/**: Pipeline report (`training_pipeline_report.txt`), test results Excel (`enhanced_model_test_results.xlsx`), predictions (`val3_predictions.xlsx`, etc.).
- **data/**: Improved train and test datasets.
- **logs/**: **Comprehensive logging output** with timestamped files:
  - `feature_engineering_YYYYMMDD_HHMMSS.log` - Feature engineering process logs
  - `model_training_YYYYMMDD_HHMMSS.log` - Model training logs  
  - `validation_testing_YYYYMMDD_HHMMSS.log` - Validation testing logs
  - `train_test_plotting_YYYYMMDD_HHMMSS.log` - Plotting process logs
  - Corresponding `_errors.log` files for any error output

**Note:** Plots are only saved to file and are not displayed interactively.

The pipeline ensures reproducible runs with timestamped isolation of results. 