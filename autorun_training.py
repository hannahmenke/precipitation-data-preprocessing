#!/usr/bin/env python3
"""
Automated Training Pipeline for Precipitation Data Classification.

This script automates the complete training pipeline:
1. Discovers training data in training_data folder
2. Runs feature engineering to create improved features
3. Trains the enhanced cluster-based ensemble model
4. Validates on data in data_for_classification folder
5. Generates comprehensive reports and visualizations

Usage:
    python autorun_training.py
    
Optional arguments:
    --max-features N    : Maximum number of features to select (default: 20)
    --random-seed N     : Random seed for reproducibility (default: 42)
    --skip-feature-eng  : Skip feature engineering if improved data exists
    --output-dir DIR    : Output directory for results (default: results/)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AutoTrainingPipeline:
    """
    Automated training pipeline for precipitation data classification.
    """
    
    def __init__(self, max_features=20, random_seed=42, output_dir=None, skip_feature_eng=False, skip_training=False):
        self.max_features = max_features
        self.random_seed = random_seed
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'results_{timestamp}'
        self.output_dir = Path(output_dir)
        self.skip_feature_eng = skip_feature_eng
        self.skip_training = skip_training
        
        # Key directories
        self.training_data_dir = Path("training_data")
        self.validation_data_dir = Path("data_for_classification")
        self.models_dir = Path("models")
        
        # Pipeline status
        self.pipeline_status = {
            'data_discovery': False,
            'feature_engineering': False,
            'model_training': False,
            'validation_testing': False,
            'report_generation': False
        }
        
        # Results storage
        self.results = {}
        
    def setup_output_directory(self):
        """Setup output directory structure."""
        print("📁 Setting up output directory...")
        
        # Create main output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = ['models', 'plots', 'reports', 'logs', 'data']
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Copy any existing important files
        if self.models_dir.exists():
            for model_file in self.models_dir.glob("latest_*.joblib"):
                shutil.copy2(model_file, self.output_dir / 'models')
        
        print(f"✓ Output directory created: {self.output_dir}")
        
    def discover_training_data(self):
        """Discover and validate training data files."""
        print("\n🔍 STEP 1: Discovering Training Data")
        print("=" * 50)
        
        if not self.training_data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {self.training_data_dir}")
        
        # Look for training files
        training_files = list(self.training_data_dir.glob("*.xlsx"))
        training_files = [f for f in training_files if not f.name.startswith('.') and not f.name.startswith('~')]
        
        print(f"Found {len(training_files)} Excel files:")
        
        training_data_info = {}
        total_samples = 0
        
        for file_path in training_files:
            try:
                df = pd.read_excel(file_path)
                samples = len(df)
                total_samples += samples
                
                # Check for target column
                has_target = 'type' in df.columns
                classes = df['type'].unique() if has_target else []
                
                training_data_info[file_path.name] = {
                    'path': file_path,
                    'samples': samples,
                    'has_target': has_target,
                    'classes': sorted(classes) if has_target else [],
                    'columns': len(df.columns)
                }
                
                print(f"  📄 {file_path.name}: {samples} samples, {len(df.columns)} columns")
                if has_target:
                    print(f"      Classes: {sorted(classes)}")
                else:
                    print(f"      ⚠️  No 'type' column found")
                    
            except Exception as e:
                print(f"  ❌ Error reading {file_path.name}: {e}")
        
        print(f"\n✓ Total training samples discovered: {total_samples}")
        
        # Store results
        self.results['training_data_info'] = training_data_info
        self.results['total_training_samples'] = total_samples
        self.pipeline_status['data_discovery'] = True
        
        return training_data_info
    
    def discover_validation_data(self):
        """Discover and validate validation data files."""
        print("\n🔍 Discovering Validation Data")
        print("-" * 30)
        
        if not self.validation_data_dir.exists():
            print(f"⚠️  Validation data directory not found: {self.validation_data_dir}")
            return {}
        
        # Look for validation files
        validation_files = list(self.validation_data_dir.glob("*.xlsx"))
        validation_files = [f for f in validation_files if not f.name.startswith('.') and not f.name.startswith('~')]
        
        print(f"Found {len(validation_files)} validation files:")
        
        validation_data_info = {}
        total_val_samples = 0
        
        for file_path in validation_files:
            try:
                df = pd.read_excel(file_path)
                samples = len(df)
                total_val_samples += samples
                
                # Check for target column
                has_target = 'type' in df.columns
                classes = df['type'].unique() if has_target else []
                
                validation_data_info[file_path.name] = {
                    'path': file_path,
                    'samples': samples,
                    'has_target': has_target,
                    'classes': sorted(classes) if has_target else [],
                    'columns': len(df.columns)
                }
                
                print(f"  📄 {file_path.name}: {samples} samples, {len(df.columns)} columns")
                if has_target:
                    print(f"      Classes: {sorted(classes)}")
                    
            except Exception as e:
                print(f"  ❌ Error reading {file_path.name}: {e}")
        
        print(f"✓ Total validation samples: {total_val_samples}")
        
        # Store results
        self.results['validation_data_info'] = validation_data_info
        self.results['total_validation_samples'] = total_val_samples
        
        return validation_data_info
    
    def run_feature_engineering(self):
        """Run feature engineering pipeline."""
        print("\n🔧 STEP 2: Feature Engineering")
        print("=" * 50)
        
        # Check if improved data already exists
        improved_train_path = Path('training_data/train_improved.xlsx')
        improved_val_path = Path('validation_data_improved.xlsx')
        
        if self.skip_feature_eng and improved_train_path.exists() and improved_val_path.exists():
            print("✓ Skipping feature engineering - improved data already exists")
            self.pipeline_status['feature_engineering'] = True
            return True
        
        print("Running feature engineering script...")
        
        try:
            # Run the improve_features.py script
            result = subprocess.run([
                sys.executable, 'improve_features.py'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print("✓ Feature engineering completed successfully")
                print("Output:")
                print(result.stdout)
                
                # Copy improved data to output directory
                if improved_train_path.exists():
                    shutil.copy2(improved_train_path, self.output_dir / 'data' / 'train_improved.xlsx')
                if improved_val_path.exists():
                    shutil.copy2(improved_val_path, self.output_dir / 'data' / 'validation_improved.xlsx')
                
                self.pipeline_status['feature_engineering'] = True
                return True
            else:
                print("❌ Feature engineering failed")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Feature engineering timed out")
            return False
        except Exception as e:
            print(f"❌ Error running feature engineering: {e}")
            return False
    
    def run_model_training(self):
        """Run enhanced model training."""
        print("\n🎯 STEP 3: Enhanced Model Training")
        print("=" * 50)
        
        print("Running cluster-based model improvement...")
        
        try:
            # Run the improve_model_with_clustering.py script
            result = subprocess.run([
                sys.executable, 'improve_model_with_clustering.py'
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                print("✓ Enhanced model training completed successfully")
                print("Training output:")
                print(result.stdout)
                
                # Copy trained models to output directory
                models_to_copy = [
                    'enhanced_ensemble_model.joblib',
                    'cluster_model.joblib',
                    'enhanced_model_metadata.joblib'
                ]
                
                for model_file in models_to_copy:
                    src_path = self.models_dir / model_file
                    if src_path.exists():
                        shutil.copy2(src_path, self.output_dir / 'models' / model_file)
                
                self.pipeline_status['model_training'] = True
                return True
            else:
                print("❌ Model training failed")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Model training timed out")
            return False
        except Exception as e:
            print(f"❌ Error running model training: {e}")
            return False
    
    def run_validation_testing(self):
        """Run validation testing on all validation datasets."""
        print("\n🧪 STEP 4: Validation Testing")
        print("=" * 50)
        
        if self.skip_training:
            # Process each validation file separately
            return self.run_individual_validation_testing()
        else:
            # Original combined validation testing + individual processing
            print("Running enhanced model validation...")
            
            success_combined = True
            try:
                # Run the test_enhanced_model.py script (without visualization)
                result = subprocess.run([
                    sys.executable, 'test_enhanced_model.py'
                ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
                
                if result.returncode == 0:
                    print("✓ Combined validation testing completed successfully")
                    print("Combined validation output:")
                    print(result.stdout)
                    
                    # Run train/test performance visualization (using enhanced features)
                    train_test_result = subprocess.run([
                        sys.executable, 'create_simple_train_test_plot.py'
                    ], capture_output=True, text=True, timeout=300)
                    
                    if train_test_result.returncode == 0:
                        print("✓ Created train vs test performance plots")
                    else:
                        print("⚠️  Train/test visualization failed")
                    
                    # Run PURE validation-focused visualization (no comparisons)
                    viz_result = subprocess.run([
                        sys.executable, 'create_validation_visualizations.py'
                    ], capture_output=True, text=True, timeout=300)
                    
                    if viz_result.returncode == 0:
                        print("✓ Created pure validation visualizations (NO comparisons)")
                    else:
                        print("⚠️  Pure validation visualization failed")
                    
                    # Copy validation results to output directory (PURE validation only)
                    results_to_copy = [
                        'enhanced_model_test_results.xlsx',
                        'train_test_performance.png',  # Train vs test performance (enhanced features)
                        'enhanced_model_validation_results.png',  # Pure validation (no comparisons)
                        'enhanced_model_val3_results.png',
                        'enhanced_model_val6_results.png',
                        'val3_predictions.xlsx',
                        'val6_predictions.xlsx',
                        'combined_predictions.xlsx'
                    ]
                    
                    for result_file in results_to_copy:
                        src_path = Path(result_file)
                        if src_path.exists():
                            if result_file.endswith('.png'):
                                shutil.copy2(src_path, self.output_dir / 'plots' / result_file)
                            else:
                                shutil.copy2(src_path, self.output_dir / 'reports' / result_file)
                    
                    # Parse validation results for summary
                    self.parse_validation_results()
                    
                else:
                    print("❌ Combined validation testing failed")
                    print("Error output:")
                    print(result.stderr)
                    success_combined = False
                    
            except subprocess.TimeoutExpired:
                print("❌ Combined validation testing timed out")
                success_combined = False
            except Exception as e:
                print(f"❌ Error running combined validation testing: {e}")
                success_combined = False
            
            # ALWAYS run individual validation testing for ALL files in data_for_classification
            print("\n🔄 Running individual validation testing for all files...")
            success_individual = self.run_individual_validation_testing()
            
            # Pipeline succeeds if either combined or individual testing succeeds
            if success_combined or success_individual:
                self.pipeline_status['validation_testing'] = True
                return True
            else:
                return False

    def run_individual_validation_testing(self):
        """Process each validation file separately."""
        print("Processing each validation file individually...")
        
        # Find all Excel files in data_for_classification
        validation_files = list(self.validation_data_dir.glob("*.xlsx"))
        validation_files = [f for f in validation_files if not f.name.startswith('.') and not f.name.startswith('~')]
        
        if not validation_files:
            print("❌ No validation files found in data_for_classification/")
            return False
        
        print(f"Found {len(validation_files)} validation files to process")
        
        all_results = []
        successful_files = 0
        
        for val_file in validation_files:
            print(f"\n📊 Processing {val_file.name}...")
            
            try:
                # Process this individual file
                success = self.process_individual_validation_file(val_file)
                if success:
                    successful_files += 1
                    all_results.append(val_file.name)
                
            except Exception as e:
                print(f"❌ Error processing {val_file.name}: {e}")
        
        print(f"\n✓ Successfully processed {successful_files}/{len(validation_files)} validation files")
        
        if successful_files > 0:
            self.pipeline_status['validation_testing'] = True
            return True
        else:
            return False

    def process_individual_validation_file(self, val_file):
        """Process a single validation file and generate individual results."""
        file_stem = val_file.stem
        
        try:
            # Create individual validation script for this file
            individual_script = f"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter

def create_improved_features(df):
    \"\"\"Create improved features based on error analysis.\"\"\"
    df_improved = df.copy()
    
    # 1. Shape Complexity Score
    extent_norm = (df['Extent'] - df['Extent'].min()) / (df['Extent'].max() - df['Extent'].min())
    circularity_norm = (df['Circularity'] - df['Circularity'].min()) / (df['Circularity'].max() - df['Circularity'].min())
    df_improved['Shape_Complexity'] = extent_norm * (1 - circularity_norm)
    
    # 2. Normalized Shape Ratios
    df_improved['Area_Perimeter_Ratio'] = df['Area'] / (df['Perimeter'] + 1e-6)
    df_improved['Convex_Efficiency'] = df['Area'] / (df['ConvexArea'] + 1e-6)
    
    # 3. Robust Eccentricity
    df_improved['Eccentricity_Robust'] = np.clip(df['Eccentricity'], 0, 1)
    
    # 4. Distance Feature Combinations
    df_improved['Distance_Ratio'] = df['dis'] / (df['dis_normal'] + 1e-6)
    df_improved['Distance_Interaction'] = df['dis'] * df['dis_normal']
    
    # 5. Intensity Stability Score
    gray_mean = df['MeanIntensity'] if 'MeanIntensity' in df.columns else df.get('Gray_ave', 0)
    df_improved['Intensity_Stability'] = gray_mean / (df['Gray_var'] + 1e-6)
    
    # 6. Comprehensive Shape Score
    df_improved['Comprehensive_Shape'] = (
        df_improved['Major_Minor_ratio'] * 
        df_improved['Circularity'] * 
        (1 - df_improved['Eccentricity_Robust'])
    )
    
    # 7. Class4 Discriminator
    df_improved['Class4_Discriminator'] = (
        df['Extent'] * df['Circularity'] * df['Perimeter']
    )
    
    return df_improved

# Load enhanced model
models_dir = Path("models")
model = joblib.load(models_dir / "enhanced_ensemble_model.joblib")
cluster_model = joblib.load(models_dir / "cluster_model.joblib")
scaler = joblib.load(models_dir / "latest_scaler.joblib")
selector = joblib.load(models_dir / "latest_feature_selector.joblib")
feature_names = joblib.load(models_dir / "latest_feature_names.joblib")

# Load and preprocess validation data
df = pd.read_excel('{val_file}')
print(f"Processing {{len(df)}} samples from {val_file.name}")

# Apply feature engineering to create improved features
df_improved = create_improved_features(df)

# Extract features and prepare data using the same features as training
exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
all_features = [col for col in df_improved.columns if col not in exclude_cols and df_improved[col].dtype in ['float64', 'int64']]

# Get the exact features that were used for training the scaler
X_all = df_improved[all_features].fillna(df_improved[all_features].median())
X_scaled = scaler.transform(X_all)
X_selected = selector.transform(X_scaled)

# Create cluster features
cluster_labels = cluster_model.predict(X_selected)
cluster_centers = cluster_model.cluster_centers_
cluster_features = []

# Distance to centers
for center in cluster_centers:
    dist = np.linalg.norm(X_selected - center, axis=1)
    cluster_features.append(dist)

# Gaussian mixture probabilities
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_selected)
cluster_probs = gmm.predict_proba(X_selected)
for i in range(2):
    cluster_features.append(cluster_probs[:, i])

# Placeholder features
cluster_features.extend([np.zeros(len(X_selected)), np.zeros(len(X_selected)), np.zeros(len(X_selected))])

# Error indicator
error_indicator = (cluster_labels == 0).astype(float)
cluster_features.append(error_indicator)

# Combine features
X_enhanced = np.hstack((X_selected, np.column_stack(cluster_features)))

# Make predictions
y_pred_raw = model.predict(X_enhanced)
label_mapping = {{0: 1, 1: 2, 2: 3, 3: 4}}
y_pred = [label_mapping[pred] for pred in y_pred_raw]
y_pred_proba = model.predict_proba(X_enhanced)
confidence = np.max(y_pred_proba, axis=1)

# Save predictions
results_df = pd.DataFrame({{
    'sample_id': range(len(df)),
    'predicted_class': y_pred,
    'confidence': confidence
}})

# Add original data columns
for col in df.columns:
    if col not in results_df.columns:
        results_df[col] = df[col].values

results_df.to_excel('{file_stem}_predictions.xlsx', index=False)

# Create visualization if we have true labels
if 'type' in df.columns:
    y_true = df['type'].values
    accuracy = accuracy_score(y_true, y_pred)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Enhanced Model Results - {file_stem.upper()} (Accuracy: {{accuracy:.1%}})', fontsize=14, fontweight='bold')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=sorted(np.unique(y_true)), yticklabels=sorted(np.unique(y_true)))
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # Confidence distribution
    axes[0,1].hist(confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].set_title('Confidence Distribution')
    axes[0,1].set_xlabel('Confidence')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(np.mean(confidence), color='red', linestyle='--', label=f'Mean: {{np.mean(confidence):.3f}}')
    axes[0,1].legend()
    
    # Class distribution - both true and predicted
    unique_classes, true_counts = np.unique(y_true, return_counts=True)
    pred_counts = [np.sum(np.array(y_pred) == cls) for cls in unique_classes]
    
    x = np.arange(len(unique_classes))
    width = 0.35
    
    axes[1,0].bar(x - width/2, true_counts, width, label='True', alpha=0.8, color='lightblue', edgecolor='black')
    axes[1,0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='orange', edgecolor='black')
    
    axes[1,0].set_title('True vs Predicted Class Distribution')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_xlabel('Class')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([f'Class {{c}}' for c in unique_classes])
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Per-class accuracy
    class_accs = []
    for cls in unique_classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], np.array(y_pred)[mask])
            class_accs.append(class_acc)
        else:
            class_accs.append(0)
    
    bars = axes[1,1].bar([f'Class {{c}}' for c in unique_classes], class_accs, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(unique_classes)])
    axes[1,1].set_title('Per-Class Accuracy')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accs):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{{acc:.1%}}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('{file_stem}_validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed results
    print(f"\\n📊 RESULTS for {file_stem}:")
    print(f"✓ Accuracy: {{accuracy:.1%}}")
    print(f"✓ Mean confidence: {{np.mean(confidence):.3f}}")
    print(f"✓ Total samples: {{len(df)}}")
    
    # Per-class breakdown
    for cls in unique_classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], np.array(y_pred)[mask])
            class_conf = np.mean(confidence[mask])
            print(f"   Class {{cls}}: {{class_acc:.1%}} accuracy, {{class_conf:.3f}} confidence ({{np.sum(mask)}} samples)")
    
    print(f"✓ Saved predictions: {file_stem}_predictions.xlsx")
    print(f"✓ Saved plot: {file_stem}_validation_results.png")
else:
    print(f"\\n📊 RESULTS for {file_stem}:")
    print(f"✓ Total samples: {{len(df)}}")
    print(f"✓ Mean confidence: {{np.mean(confidence):.3f}}")
    print(f"✓ Saved predictions: {file_stem}_predictions.xlsx (no true labels for accuracy)")
"""
            
            # Write and execute the individual script
            script_path = f"process_{file_stem}.py"
            with open(script_path, 'w') as f:
                f.write(individual_script)
            
            # Run the script
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✓ Individual validation completed")
                print(result.stdout)
                
                # Copy results to output directory
                pred_file = f"{file_stem}_predictions.xlsx"
                plot_file = f"{file_stem}_validation_results.png"
                
                if Path(pred_file).exists():
                    shutil.copy2(pred_file, self.output_dir / 'reports' / pred_file)
                if Path(plot_file).exists():
                    shutil.copy2(plot_file, self.output_dir / 'plots' / plot_file)
                
                # Clean up script
                Path(script_path).unlink()
                
                return True
            else:
                print("❌ Individual validation failed")
                print("Error:", result.stderr)
                # Clean up script
                if Path(script_path).exists():
                    Path(script_path).unlink()
                return False
                
        except Exception as e:
            print(f"❌ Error in individual validation: {e}")
            return False
    
    def parse_validation_results(self):
        """Parse validation results for summary reporting."""
        try:
            results_file = Path('enhanced_model_test_results.xlsx')
            if results_file.exists():
                df = pd.read_excel(results_file)
                
                # Extract key metrics
                validation_summary = {}
                
                for dataset in df['dataset'].unique():
                    dataset_data = df[df['dataset'] == dataset]
                    
                    original_row = dataset_data[dataset_data['model'] == 'original']
                    enhanced_row = dataset_data[dataset_data['model'] == 'enhanced']
                    
                    if not original_row.empty and not enhanced_row.empty:
                        original_acc = original_row['accuracy'].iloc[0]
                        enhanced_acc = enhanced_row['accuracy'].iloc[0]
                        improvement = enhanced_acc - original_acc
                        
                        validation_summary[dataset] = {
                            'original_accuracy': original_acc,
                            'enhanced_accuracy': enhanced_acc,
                            'improvement': improvement,
                            'original_confidence': original_row['avg_confidence'].iloc[0],
                            'enhanced_confidence': enhanced_row['avg_confidence'].iloc[0]
                        }
                
                self.results['validation_summary'] = validation_summary
                
        except Exception as e:
            print(f"⚠️  Could not parse validation results: {e}")
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n📋 STEP 5: Generating Final Report")
        print("=" * 50)
        
        report_content = self.create_report_content()
        
        # Save report
        report_path = self.output_dir / 'reports' / 'training_pipeline_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"✓ Final report saved to: {report_path}")
        
        self.pipeline_status['report_generation'] = True
        
        # Also print summary to console
        print(self.create_summary_report())
    
    def create_report_content(self):
        """Create detailed report content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
AUTOMATED TRAINING PIPELINE REPORT
{'='*50}
Generated: {timestamp}
Pipeline Configuration:
  - Max Features: {self.max_features}
  - Random Seed: {self.random_seed}
  - Output Directory: {self.output_dir}
  - Skip Feature Engineering: {self.skip_feature_eng}
  - Skip Training: {self.skip_training}

PIPELINE STATUS:
{'='*20}
"""
        
        for step, status in self.pipeline_status.items():
            status_symbol = "✅" if status else "❌"
            report += f"{status_symbol} {step.replace('_', ' ').title()}: {'COMPLETED' if status else 'FAILED'}\n"
        
        report += f"\nDATA DISCOVERY:\n{'='*20}\n"
        
        if 'training_data_info' in self.results:
            report += f"Training Data Files: {len(self.results['training_data_info'])}\n"
            report += f"Total Training Samples: {self.results.get('total_training_samples', 0)}\n"
            
            for filename, info in self.results['training_data_info'].items():
                report += f"  📄 {filename}: {info['samples']} samples, {info['columns']} columns\n"
                if info['has_target']:
                    report += f"      Classes: {info['classes']}\n"
        
        if 'validation_data_info' in self.results:
            report += f"\nValidation Data Files: {len(self.results['validation_data_info'])}\n"
            report += f"Total Validation Samples: {self.results.get('total_validation_samples', 0)}\n"
            
            for filename, info in self.results['validation_data_info'].items():
                report += f"  📄 {filename}: {info['samples']} samples, {info['columns']} columns\n"
        
        if 'validation_summary' in self.results:
            report += f"\nVALIDATION RESULTS:\n{'='*20}\n"
            
            for dataset, metrics in self.results['validation_summary'].items():
                report += f"\n{dataset.upper()} Dataset:\n"
                report += f"  Original Accuracy: {metrics['original_accuracy']:.1%}\n"
                report += f"  Enhanced Accuracy: {metrics['enhanced_accuracy']:.1%}\n"
                report += f"  Improvement: {metrics['improvement']:+.1%}\n"
                report += f"  Original Confidence: {metrics['original_confidence']:.3f}\n"
                report += f"  Enhanced Confidence: {metrics['enhanced_confidence']:.3f}\n"
        
        report += f"\nOUTPUT FILES:\n{'='*20}\n"
        report += f"📁 Models: {self.output_dir}/models/\n"
        report += f"📁 Plots: {self.output_dir}/plots/\n"
        report += f"📁 Reports: {self.output_dir}/reports/\n"
        report += f"📁 Data: {self.output_dir}/data/\n"
        report += f"📁 Logs: {self.output_dir}/logs/\n"
        
        return report
    
    def create_summary_report(self):
        """Create summary report for console output."""
        success_count = sum(self.pipeline_status.values())
        total_steps = len(self.pipeline_status)
        
        summary = f"\n{'='*60}\n"
        summary += f"🎯 TRAINING PIPELINE SUMMARY\n"
        summary += f"{'='*60}\n"
        summary += f"Pipeline Success: {success_count}/{total_steps} steps completed\n"
        
        if 'validation_summary' in self.results and self.results['validation_summary']:
            summary += f"\n📈 PERFORMANCE IMPROVEMENTS:\n"
            
            total_improvements = []
            for dataset, metrics in self.results['validation_summary'].items():
                improvement = metrics['improvement']
                total_improvements.append(improvement)
                summary += f"  {dataset:10}: {metrics['original_accuracy']:.1%} → {metrics['enhanced_accuracy']:.1%} ({improvement:+.1%})\n"
            
            if total_improvements:
                avg_improvement = np.mean(total_improvements)
                summary += f"\n🎉 Average Improvement: {avg_improvement:+.1%}\n"
                
                if avg_improvement > 0.01:
                    summary += "✅ ENHANCED MODEL PERFORMS BETTER!\n"
                elif avg_improvement < -0.01:
                    summary += "⚠️  Original model performed better\n"
                else:
                    summary += "➖ Performance similar\n"
        
        if success_count == total_steps:
            summary += f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!\n"
            summary += f"📁 Results saved to: {self.output_dir}\n"
        else:
            summary += f"\n⚠️  PIPELINE COMPLETED WITH ISSUES\n"
            failed_steps = [step for step, status in self.pipeline_status.items() if not status]
            summary += f"❌ Failed steps: {', '.join(failed_steps)}\n"
        
        summary += f"{'='*60}\n"
        
        return summary
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        start_time = datetime.now()
        
        print("🚀 AUTOMATED TRAINING PIPELINE STARTING")
        print("=" * 60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        print(f"  Max features: {self.max_features}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Skip feature engineering: {self.skip_feature_eng}")
        print(f"  Skip training: {self.skip_training}")
        
        try:
            # Setup output directory
            self.setup_output_directory()
            
            # Step 1: Data Discovery
            self.discover_training_data()
            self.discover_validation_data()
            
            # Step 2: Feature Engineering
            if not self.skip_training:
                if not self.run_feature_engineering():
                    print("⚠️  Feature engineering failed, but continuing...")
            
            # Step 3: Model Training
            if not self.skip_training:
                if not self.run_model_training():
                    print("❌ Model training failed - stopping pipeline")
                    return False
            
            # Step 4: Validation Testing
            if not self.run_validation_testing():
                print("⚠️  Validation testing failed, but continuing...")
            
            # Step 5: Generate Final Report
            self.generate_final_report()
            
            # Calculate total time
            end_time = datetime.now()
            total_time = end_time - start_time
            
            print(f"\n🎉 PIPELINE COMPLETED!")
            print(f"Total time: {total_time}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Automated Training Pipeline for Precipitation Data Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--max-features', type=int, default=20,
                       help='Maximum number of features to select')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results (defaults to timestamped folder)')
    parser.add_argument('--skip-feature-eng', action='store_true',
                       help='Skip feature engineering if improved data exists')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and only run validation on data_for_classification files')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = AutoTrainingPipeline(
        max_features=args.max_features,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        skip_feature_eng=args.skip_feature_eng,
        skip_training=args.skip_training
    )
    
    success = pipeline.run_full_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 