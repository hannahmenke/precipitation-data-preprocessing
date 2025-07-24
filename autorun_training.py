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
import seaborn as sns
import joblib
from pathlib import Path
import subprocess
import sys
import shutil
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TeeLogger:
    """Logger that writes to both console and file simultaneously."""
    
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log_file.write(f"=== AUTORUN TRAINING PIPELINE LOG ===\n")
        self.log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 50 + "\n\n")
        self.log_file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.write(f"\n\n=== LOG ENDED ===\n")
        self.log_file.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.close()


def setup_comprehensive_logging(output_dir):
    """Set up comprehensive logging that captures all output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_path = output_dir / 'logs' / f'autorun_training_{timestamp}.log'
    
    # Redirect stdout to our TeeLogger
    tee_logger = TeeLogger(main_log_path)
    sys.stdout = tee_logger
    
    return tee_logger

class AutoTrainingPipeline:
    """
    Automated training pipeline for precipitation data classification.
    """
    
    def __init__(self, max_features=20, random_seed=42, output_dir=None, skip_feature_eng=False, skip_training=False, model_dir=None):
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
        if model_dir is None:
            # Default to base models directory - will be updated after setup
            self.models_dir = Path("models")
        else:
            self.models_dir = Path(model_dir)
        
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
        
        # Setup output directory first
        self.setup_output_directory()
        
        # Setup comprehensive logging
        self.logger = setup_comprehensive_logging(self.output_dir)
        
        # Update models directory to point to results if not explicitly set and not skipping training
        if model_dir is None and not self.skip_training:
            self.models_dir = self.output_dir / "models"
    
    def cleanup_logging(self):
        """Clean up logging resources."""
        if hasattr(self, 'logger') and self.logger:
            self.logger.close()
            sys.stdout = self.logger.terminal  # Restore original stdout
    
    def _run_script_with_model_redirect(self, script_name, timeout=300):
        """Run a script with temporary model directory redirection."""
        original_models_dir = Path("models")
        models_backup_dir = Path("models_backup_temp_script")
        
        # Backup original models directory if it exists
        if original_models_dir.exists():
            if models_backup_dir.exists():
                shutil.rmtree(models_backup_dir)
            shutil.move(str(original_models_dir), str(models_backup_dir))
        
        # Create symlink to results models directory
        original_models_dir.symlink_to(self.output_dir / 'models', target_is_directory=True)
        
        try:
            result = subprocess.run([
                sys.executable, script_name
            ], capture_output=True, text=True, timeout=timeout)
            return result
        finally:
            # Clean up symlink
            if original_models_dir.is_symlink():
                original_models_dir.unlink()
            
            # Restore backup if it exists
            if models_backup_dir.exists():
                shutil.move(str(models_backup_dir), str(original_models_dir))
        
    def setup_output_directory(self):
        """Setup output directory structure."""
        print("Directory: Setting up output directory...")
        
        # Create main output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = ['models', 'plots', 'reports', 'logs', 'data']
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Copy any existing important files (only if not the same directory)
        if self.models_dir.exists() and self.models_dir != (self.output_dir / 'models'):
            # Copy all model files that might be needed
            important_patterns = [
                "latest_*.joblib",
                "enhanced_*.joblib", 
                "cluster_*.joblib"
            ]
            
            for pattern in important_patterns:
                for model_file in self.models_dir.glob(pattern):
                    dest_path = self.output_dir / 'models' / model_file.name
                    if not dest_path.exists():  # Don't overwrite existing files
                        shutil.copy2(model_file, dest_path)
                        print(f"Copied existing model: {model_file.name}")
        
        print(f"Success: Output directory created: {self.output_dir}")
    
    def save_subprocess_logs(self, result, log_name):
        """Save subprocess stdout and stderr to log files."""
        print(f"[LOG] Saving logs for {log_name}...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Always save a log file with basic info
        stdout_path = self.output_dir / 'logs' / f"{log_name}_{timestamp}.log"
        with open(stdout_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {log_name.upper()} OUTPUT ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write("=" * 50 + "\n\n")
            
            if result.stdout:
                f.write(result.stdout)
            else:
                f.write("(No stdout output)\n")
        
        print(f"Success: Saved log to {stdout_path}")
        
        # Save stderr separately if there are errors
        if result.stderr:
            stderr_path = self.output_dir / 'logs' / f"{log_name}_{timestamp}_errors.log"
            with open(stderr_path, 'w', encoding='utf-8') as f:
                f.write(f"=== {log_name.upper()} ERRORS ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result.stderr)
            print(f"Success: Saved error log to {stderr_path}")
        
    def discover_training_data(self):
        """Discover and validate training data files."""
        print("\nSTEP 1: Discovering Training Data")
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
                
                print(f"  File: {file_path.name}: {samples} samples, {len(df.columns)} columns")
                if has_target:
                    print(f"      Classes: {sorted(classes)}")
                else:
                    print(f"      Warning: No 'type' column found")
                    
            except Exception as e:
                print(f"Error: Error reading {file_path.name}: {e}")
        
        print(f"\nSuccess: Total training samples discovered: {total_samples}")
        
        # Store results
        self.results['training_data_info'] = training_data_info
        self.results['total_training_samples'] = total_samples
        self.pipeline_status['data_discovery'] = True
        
        return training_data_info
    
    def discover_validation_data(self):
        """Discover and validate validation data files."""
        print("\nSTEP: Discovering Validation Data")
        print("-" * 30)
        
        if not self.validation_data_dir.exists():
            print(f"Warning: Validation data directory not found: {self.validation_data_dir}")
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
                
                print(f"  File: {file_path.name}: {samples} samples, {len(df.columns)} columns")
                if has_target:
                    print(f"      Classes: {sorted(classes)}")
                    
            except Exception as e:
                print(f"Error: Error reading {file_path.name}: {e}")
        
        print(f"Success: Total validation samples: {total_val_samples}")
        
        # Store results
        self.results['validation_data_info'] = validation_data_info
        self.results['total_validation_samples'] = total_val_samples
        
        return validation_data_info
    
    def run_feature_engineering(self):
        """Run feature engineering pipeline."""
        print("\nSTEP 2: Feature Engineering")
        print("=" * 50)
        
        # Check if improved data already exists
        improved_train_path = Path('training_data/train_improved.xlsx')
        improved_test_path = Path('test_data_improved.xlsx')  # Renamed for clarity - this is test data
        
        if self.skip_feature_eng and improved_train_path.exists() and improved_test_path.exists():
            print("Success: Skipping feature engineering - improved data already exists")
            self.pipeline_status['feature_engineering'] = True
            return True
        
        print("Running feature engineering script...")
        
        try:
            # Run the improve_features.py script
            result = subprocess.run([
                sys.executable, 'improve_features.py'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            # Save logs regardless of success/failure
            self.save_subprocess_logs(result, "feature_engineering")
            
            if result.returncode == 0:
                print("Success: Feature engineering completed successfully")
                print("Output:")
                print(result.stdout)
                
                # Copy improved data to output directory
                if improved_train_path.exists():
                    shutil.copy2(improved_train_path, self.output_dir / 'data' / 'train_improved.xlsx')
                if improved_test_path.exists():
                    shutil.copy2(improved_test_path, self.output_dir / 'data' / 'test_improved.xlsx')
                
                self.pipeline_status['feature_engineering'] = True
                return True
            else:
                print("Error: Feature engineering failed")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("Error: Feature engineering timed out")
            return False
        except Exception as e:
            print(f"Error: Error running feature engineering: {e}")
            return False
    
    def run_model_training(self):
        """Run enhanced model training."""
        print("\nSTEP 3: Enhanced Model Training")
        print("=" * 50)
        
        print("Running cluster-based model improvement...")
        
        try:
            # Run the improve_model_with_clustering.py script normally
            result = subprocess.run([
                sys.executable, 'improve_model_with_clustering.py'
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            # Save logs regardless of success/failure
            self.save_subprocess_logs(result, "model_training")
            
            if result.returncode == 0:
                print("Success: Enhanced model training completed successfully")
                print("Training output:")
                print(result.stdout)
                
                # Move trained models from base directory to results directory
                models_to_move = [
                    'enhanced_ensemble_model.joblib',
                    'cluster_model.joblib',
                    'enhanced_model_metadata.joblib'
                ]
                
                print("Moving enhanced models to results directory...")
                base_models_dir = Path("models")
                results_models_dir = self.output_dir / 'models'
                
                for model_file in models_to_move:
                    src_path = base_models_dir / model_file
                    dest_path = results_models_dir / model_file
                    if src_path.exists():
                        shutil.move(str(src_path), str(dest_path))
                        print(f"Moved enhanced model: {model_file}")
                    else:
                        print(f"Warning: Enhanced model not found: {model_file}")
                
                print("Enhanced models saved only to results directory")
                
                self.pipeline_status['model_training'] = True
                return True
            else:
                print("Error: Model training failed")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("Error: Model training timed out")
            return False
        except Exception as e:
            print(f"Error: Error running model training: {e}")
            return False
    
    def run_validation_testing(self):
        """Run validation testing on all validation datasets."""
        print("\nSTEP 4: Validation Testing")
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
                # Use symlink redirection for model access
                result = self._run_script_with_model_redirect('test_enhanced_model.py', timeout=600)
                
                # Save logs regardless of success/failure
                self.save_subprocess_logs(result, "validation_testing")
                
                if result.returncode == 0:
                    print("Success: Combined validation testing completed successfully")
                    print("Combined validation output:")
                    print(result.stdout)
                    
                    # Run train/test performance visualization (using enhanced features)
                    train_test_result = self._run_script_with_model_redirect('create_simple_train_test_plot.py', timeout=300)
                    
                    # Save logs for train/test plotting
                    self.save_subprocess_logs(train_test_result, "train_test_plotting")
                    
                    if train_test_result.returncode == 0:
                        print("Success: Created train vs test performance plots")
                    else:
                        print("Warning: Train/test visualization failed")
                    
                    # Run PURE validation-focused visualization (no comparisons)
                    viz_result = self._run_script_with_model_redirect('create_validation_visualizations.py', timeout=300)
                    
                    # Save logs for validation visualization
                    self.save_subprocess_logs(viz_result, "validation_visualization")
                    
                    if viz_result.returncode == 0:
                        print("Success: Created pure validation visualizations (NO comparisons)")
                    else:
                        print("Warning: Pure validation visualization failed")
                    
                    # Copy validation results to output directory (PURE validation only)
                    results_to_copy = [
                        'enhanced_model_test_results.xlsx',
                        'train_test_performance.png',  # Train vs test performance (enhanced features)
                        'enhanced_model_validation_results.png',  # Pure validation (no comparisons)
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

                    # NEW: Copy all dynamically generated individual validation plots
                    for plot_file in Path('.').glob('enhanced_model_*_results.png'):
                        shutil.copy2(plot_file, self.output_dir / 'plots' / plot_file.name)
                    # ALSO copy all *_validation_results.png plots (from individual scripts)
                    for plot_file in Path('.').glob('*_validation_results.png'):
                        shutil.copy2(plot_file, self.output_dir / 'plots' / plot_file.name)
                    
                    # Parse validation results for summary
                    self.parse_validation_results()
                    
                else:
                    print("Error: Combined validation testing failed")
                    print("Error output:")
                    print(result.stderr)
                    success_combined = False
                    
            except subprocess.TimeoutExpired:
                print("Error: Combined validation testing timed out")
                success_combined = False
            except Exception as e:
                print(f"Error: Error running combined validation testing: {e}")
                success_combined = False
            
            # ALWAYS run individual validation testing for ALL files in data_for_classification
            print("\nSTEP: Running individual validation testing for all files...")
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
            print("Error: No validation files found in data_for_classification/")
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
                print(f"Error: Error processing {val_file.name}: {e}")
        
        print(f"\nSuccess: Successfully processed {successful_files}/{len(validation_files)} validation files")
        
        if successful_files > 0:
            self.pipeline_status['validation_testing'] = True
            return True
        else:
            return False

    def create_improved_features(self, df):
        """Create improved features based on error analysis."""
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
    
    def load_validation_models(self):
        """Load all required models and components for validation."""
        try:
            return {
                'model': joblib.load(self.models_dir / "enhanced_ensemble_model.joblib"),
                'cluster_model': joblib.load(self.models_dir / "cluster_model.joblib"),
                'scaler': joblib.load(self.models_dir / "latest_scaler.joblib"),
                'selector': joblib.load(self.models_dir / "latest_feature_selector.joblib"),
                'feature_names': joblib.load(self.models_dir / "latest_feature_names.joblib")
            }
        except Exception as e:
            print(f"Error loading models from {self.models_dir}: {e}")
            return None
    
    def create_validation_visualization(self, y_true, y_pred, confidence, file_stem):
        """Create validation visualization plots."""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Enhanced Model Results - {file_stem.upper()} (Accuracy: {accuracy:.1%})', 
                        fontsize=14, fontweight='bold')
            
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
            axes[0,1].axvline(np.mean(confidence), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(confidence):.3f}')
            axes[0,1].legend()
            
            # Class distribution - both true and predicted
            unique_classes, true_counts = np.unique(y_true, return_counts=True)
            pred_counts = [np.sum(np.array(y_pred) == cls) for cls in unique_classes]
            
            x = np.arange(len(unique_classes))
            width = 0.35
            
            axes[1,0].bar(x - width/2, true_counts, width, label='True', alpha=0.8, 
                         color='lightblue', edgecolor='black')
            axes[1,0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, 
                         color='orange', edgecolor='black')
            
            axes[1,0].set_title('True vs Predicted Class Distribution')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_xlabel('Class')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels([f'Class {c}' for c in unique_classes])
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
            
            bars = axes[1,1].bar([f'Class {c}' for c in unique_classes], class_accs, alpha=0.7, 
                                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(unique_classes)])
            axes[1,1].set_title('Per-Class Accuracy')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, class_accs):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                              f'{acc:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_file = f'{file_stem}_validation_results.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file, accuracy, unique_classes, class_accs
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None, None, None, None
    
    def process_individual_validation_file(self, val_file):
        """Process a single validation file and generate individual results."""
        file_stem = val_file.stem
        
        try:
            # Load models once
            models = self.load_validation_models()
            if models is None:
                return False
            
            # Load and preprocess validation data
            df = pd.read_excel(val_file)
            print(f"Processing {len(df)} samples from {val_file.name}")
            
            # Apply feature engineering
            df_improved = self.create_improved_features(df)
            
            # Extract features and prepare data
            exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
            all_features = [col for col in df_improved.columns 
                           if col not in exclude_cols and df_improved[col].dtype in ['float64', 'int64']]
            
            # Prepare features for prediction
            X_all = df_improved[all_features].fillna(df_improved[all_features].median())
            X_scaled = models['scaler'].transform(X_all)
            X_selected = models['selector'].transform(X_scaled)
            
            # Create cluster features
            cluster_labels = models['cluster_model'].predict(X_selected)
            cluster_centers = models['cluster_model'].cluster_centers_
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
            y_pred_raw = models['model'].predict(X_enhanced)
            label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
            y_pred = [label_mapping[pred] for pred in y_pred_raw]
            y_pred_proba = models['model'].predict_proba(X_enhanced)
            confidence = np.max(y_pred_proba, axis=1)
            
            # Save predictions
            results_df = pd.DataFrame({
                'sample_id': range(len(df)),
                'predicted_class': y_pred,
                'confidence': confidence
            })
            
            # Add original data columns
            for col in df.columns:
                if col not in results_df.columns:
                    results_df[col] = df[col].values
            
            pred_file = f"{file_stem}_predictions.xlsx"
            results_df.to_excel(pred_file, index=False)
            
            # Create visualization and print results
            if 'type' in df.columns:
                y_true = df['type'].values
                plot_file, accuracy, unique_classes, class_accs = self.create_validation_visualization(
                    y_true, y_pred, confidence, file_stem)
                
                # Print detailed results
                print(f"\n📊 RESULTS for {file_stem}:")
                print(f"Success: Accuracy: {accuracy:.1%}")
                print(f"Success: Mean confidence: {np.mean(confidence):.3f}")
                print(f"Success: Total samples: {len(df)}")
                
                # Per-class breakdown
                for cls in unique_classes:
                    mask = y_true == cls
                    if np.sum(mask) > 0:
                        class_acc = accuracy_score(y_true[mask], np.array(y_pred)[mask])
                        class_conf = np.mean(confidence[mask])
                        print(f"   Class {cls}: {class_acc:.1%} accuracy, {class_conf:.3f} confidence ({np.sum(mask)} samples)")
                
                print(f"Success: Saved predictions: {pred_file}")
                if plot_file:
                    print(f"Success: Saved plot: {plot_file}")
            else:
                plot_file = None
                print(f"\n📊 RESULTS for {file_stem}:")
                print(f"Success: Total samples: {len(df)}")
                print(f"Success: Mean confidence: {np.mean(confidence):.3f}")
                print(f"Success: Saved predictions: {pred_file} (no true labels for accuracy)")
            
            # Copy results to output directory
            if Path(pred_file).exists():
                shutil.copy2(pred_file, self.output_dir / 'reports' / pred_file)
            if plot_file and Path(plot_file).exists():
                shutil.copy2(plot_file, self.output_dir / 'plots' / plot_file)
            
            return True
                
        except Exception as e:
            print(f"Error: Error in individual validation: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"Warning: Could not parse validation results: {e}")
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\nSTEP 5: Generating Final Report")
        print("=" * 50)
        
        report_content = self.create_report_content()
        
        # Save report
        report_path = self.output_dir / 'reports' / 'training_pipeline_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Success: Final report saved to: {report_path}")
        
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
            status_symbol = "Success:" if status else "Error:"
            report += f"{status_symbol} {step.replace('_', ' ').title()}: {'COMPLETED' if status else 'FAILED'}\n"
        
        report += f"\nDATA DISCOVERY:\n{'='*20}\n"
        
        if 'training_data_info' in self.results:
            report += f"Training Data Files: {len(self.results['training_data_info'])}\n"
            report += f"Total Training Samples: {self.results.get('total_training_samples', 0)}\n"
            
            for filename, info in self.results['training_data_info'].items():
                report += f"  File: {filename}: {info['samples']} samples, {info['columns']} columns\n"
                if info['has_target']:
                    report += f"      Classes: {info['classes']}\n"
        
        if 'validation_data_info' in self.results:
            report += f"\nValidation Data Files: {len(self.results['validation_data_info'])}\n"
            report += f"Total Validation Samples: {self.results.get('total_validation_samples', 0)}\n"
            
            for filename, info in self.results['validation_data_info'].items():
                report += f"  File: {filename}: {info['samples']} samples, {info['columns']} columns\n"
        
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
        report += f"Directory: {self.output_dir}/models/\n"
        report += f"Directory: {self.output_dir}/plots/\n"
        report += f"Directory: {self.output_dir}/reports/\n"
        report += f"Directory: {self.output_dir}/data/\n"
        report += f"Directory: {self.output_dir}/logs/\n"
        
        return report
    
    def create_summary_report(self):
        """Create summary report for console output."""
        # If skip_training, treat feature_engineering and model_training as 'skipped', not failed
        skipped_steps = []
        pipeline_status = self.pipeline_status.copy()
        if self.skip_training:
            for step in ['feature_engineering', 'model_training']:
                if not pipeline_status[step]:
                    pipeline_status[step] = None  # Mark as skipped
                    skipped_steps.append(step)
        success_count = sum(1 for v in pipeline_status.values() if v)
        total_steps = len([v for v in pipeline_status.values() if v is not None])
        
        summary = f"\n{'='*60}\n"
        summary += f"STEP: TRAINING PIPELINE SUMMARY\n"
        summary += f"{'='*60}\n"
        summary += f"Pipeline Success: {success_count}/{total_steps} steps completed\n"
        
        if 'validation_summary' in self.results and self.results['validation_summary']:
            summary += f"\n📈 PERFORMANCE IMPROVEMENTS:\n"
            
            total_improvements = []
            for dataset, metrics in self.results['validation_summary'].items():
                improvement = metrics['improvement']
                total_improvements.append(improvement)
                summary += f"  {dataset:10}: {metrics['original_accuracy']:.1%} -> {metrics['enhanced_accuracy']:.1%} ({improvement:+.1%})\n"
            
            if total_improvements:
                avg_improvement = np.mean(total_improvements)
                summary += f"\nSuccess: Average Improvement: {avg_improvement:+.1%}\n"
                
                if avg_improvement > 0.01:
                    summary += "Success: ENHANCED MODEL PERFORMS BETTER!\n"
                elif avg_improvement < -0.01:
                    summary += "Warning: Original model performed better\n"
                else:
                    summary += "- Performance similar\n"
        
        # Only show failed steps that were not skipped
        failed_steps = [step for step, status in pipeline_status.items() if status is False]
        if success_count == total_steps:
            summary += f"\nSuccess: PIPELINE COMPLETED SUCCESSFULLY!\n"
            summary += f"Directory: Results saved to: {self.output_dir}\n"
        else:
            summary += f"\nWarning: PIPELINE COMPLETED WITH ISSUES\n"
            if failed_steps:
                summary += f"Error: Failed steps: {', '.join(failed_steps)}\n"
            if skipped_steps:
                summary += f"Note: Skipped steps: {', '.join(skipped_steps)} (due to --skip-training)\n"
        
        summary += f"{'='*60}\n"
        
        return summary
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        start_time = datetime.now()
        
        print("[START] AUTOMATED TRAINING PIPELINE STARTING")
        print("=" * 60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        print(f"  Max features: {self.max_features}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Skip feature engineering: {self.skip_feature_eng}")
        print(f"  Skip training: {self.skip_training}")
        print(f"  Model directory: {self.models_dir}")
        
        # Validate model directory when skipping training
        if self.skip_training:
            if not self.models_dir.exists():
                print(f"Error: Model directory '{self.models_dir}' does not exist!")
                print("Please ensure the model directory exists or don't use --skip-training")
                return False
            
            # Check for required model files
            required_models = [
                "enhanced_ensemble_model.joblib",
                "cluster_model.joblib", 
                "latest_scaler.joblib",
                "latest_feature_selector.joblib",
                "latest_feature_names.joblib"
            ]
            
            missing_models = []
            for model_file in required_models:
                if not (self.models_dir / model_file).exists():
                    missing_models.append(model_file)
            
            if missing_models:
                print(f"Error: Missing required model files in '{self.models_dir}':")
                for model in missing_models:
                    print(f"  - {model}")
                print("Please ensure all required models are present or don't use --skip-training")
                return False
            
            print(f"✓ All required models found in {self.models_dir}")
        
        try:
            # Setup output directory
            self.setup_output_directory()
            
            # Step 1: Data Discovery
            self.discover_training_data()
            self.discover_validation_data()
            
            # Step 2: Feature Engineering
            if not self.skip_training:
                if not self.run_feature_engineering():
                    print("Warning: Feature engineering failed, but continuing...")
            
            # Step 3: Model Training
            if not self.skip_training:
                if not self.run_model_training():
                    print("Error: Model training failed - stopping pipeline")
                    return False
            
            # Step 4: Validation Testing
            if not self.run_validation_testing():
                print("Warning: Validation testing failed, but continuing...")
            
            # Step 5: Generate Final Report
            self.generate_final_report()
            
            # Calculate total time
            end_time = datetime.now()
            total_time = end_time - start_time
            
            print(f"\nSuccess: PIPELINE COMPLETED!")
            print(f"Total time: {total_time}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f"\nError: PIPELINE FAILED: {e}")
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
    parser.add_argument('--model-dir', type=str, default=None, help='Custom model directory to use when skipping training (default: models/)')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = AutoTrainingPipeline(
        max_features=args.max_features,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        skip_feature_eng=args.skip_feature_eng,
        skip_training=args.skip_training,
        model_dir=args.model_dir
    )
    
    try:
        success = pipeline.run_full_pipeline()
    finally:
        # Always close the logger properly
        pipeline.cleanup_logging()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 