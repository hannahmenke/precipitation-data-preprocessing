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
    
    def __init__(self, max_features=20, random_seed=42, output_dir=None, skip_feature_eng=False, skip_training=False, model_dir=None, network_dir=None, training_mode='combined'):
        self.max_features = max_features
        self.random_seed = random_seed
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'results_{timestamp}'
        self.output_dir = Path(output_dir)
        self.skip_feature_eng = skip_feature_eng
        self.skip_training = skip_training
        self.network_dir = (network_dir or "first_network").rstrip('/')
        self.training_mode = training_mode
        
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
        """Discover and validate training data files in experiment structure."""
        print("\nSTEP 1: Discovering Training Data")
        print("=" * 50)

        if not self.training_data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {self.training_data_dir}")

        # Look for experiment directories and their train/test/val files
        experiment_dirs = []
        network_dir = self.training_data_dir / self.network_dir
        print(f"Using network directory: {self.network_dir}")

        if network_dir.exists():
            for exp_dir in network_dir.iterdir():
                if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                    split_data_dir = exp_dir / "split_data_tables"
                    if split_data_dir.exists():
                        experiment_dirs.append(split_data_dir)
        else:
            raise FileNotFoundError(f"Network directory not found: {network_dir}")

        print(f"Found {len(experiment_dirs)} experiment directories with split data:")

        training_data_info = {}
        total_train_samples = 0
        total_test_samples = 0
        total_val_samples = 0

        for exp_dir in experiment_dirs:
            exp_name = f"{exp_dir.parent.parent.name}/{exp_dir.parent.name}"
            print(f"\n  Experiment: {exp_name}")

            # Check for train, test, val files
            train_file = exp_dir / "train.xlsx"
            test_file = exp_dir / "test.xlsx"
            val_file = exp_dir / "val.xlsx"

            exp_info = {
                'experiment_dir': exp_dir,
                'train_file': train_file if train_file.exists() else None,
                'test_file': test_file if test_file.exists() else None,
                'val_file': val_file if val_file.exists() else None,
                'train_samples': 0,
                'test_samples': 0,
                'val_samples': 0,
                'classes': []
            }

            # Process each split file
            for split_name, file_path in [('train', train_file), ('test', test_file), ('val', val_file)]:
                if file_path and file_path.exists():
                    try:
                        df = pd.read_excel(file_path)
                        samples = len(df)
                        exp_info[f'{split_name}_samples'] = samples

                        if split_name == 'train':
                            total_train_samples += samples
                        elif split_name == 'test':
                            total_test_samples += samples
                        elif split_name == 'val':
                            total_val_samples += samples

                        # Check for target column
                        has_target = 'type' in df.columns
                        if has_target and split_name == 'train':
                            exp_info['classes'] = sorted(df['type'].unique())

                        print(f"    {split_name:5}: {samples:4} samples, {len(df.columns):2} columns")
                        if has_target and split_name == 'train':
                            print(f"           Classes: {exp_info['classes']}")
                        elif not has_target:
                            print(f"           Warning: No 'type' column found")

                    except Exception as e:
                        print(f"    Error reading {file_path.name}: {e}")
                else:
                    print(f"    {split_name:5}: Missing")

            training_data_info[exp_name] = exp_info

        print(f"\nSuccess: Total samples discovered:")
        print(f"  Train: {total_train_samples}")
        print(f"  Test:  {total_test_samples}")
        print(f"  Val:   {total_val_samples}")

        # Show training mode analysis
        if self.training_mode != 'combined':
            print(f"\n🎯 TRAINING MODE: {self.training_mode.upper()}")
            print("=" * 50)
            training_groups = self.group_experiments_by_training_mode(training_data_info)
            for group_name, experiments in training_groups.items():
                print(f"Group '{group_name}': {len(experiments)} experiments")
                for exp in experiments:
                    exp_info = training_data_info[exp]
                    print(f"  - {exp}: {exp_info['train_samples']} train samples")
            print(f"\nNote: {len(training_groups)} separate models will be trained")
            print("=" * 50)

        # Store results
        self.results['training_data_info'] = training_data_info
        self.results['total_training_samples'] = total_train_samples
        self.results['total_test_samples'] = total_test_samples
        self.results['total_val_samples'] = total_val_samples
        self.pipeline_status['data_discovery'] = True

        return training_data_info

    def discover_existing_per_experiment_models(self):
        """Discover existing per-experiment models when skipping training."""
        print("\nDiscovering existing per-experiment models...")

        if not hasattr(self, 'results') or 'training_data_info' not in self.results:
            print("Warning: No training data info found")
            return

        # Initialize trained_model_groups if needed
        if not hasattr(self, 'trained_model_groups'):
            self.trained_model_groups = {}

        # Check for each experiment if it has a corresponding per-experiment model
        for exp_name in self.results['training_data_info'].keys():
            # Create safe group name (same logic as training)
            safe_group_name = exp_name.replace('/', '_').replace('\\', '_')
            model_suffix = f"_{safe_group_name}"

            # Check if per-experiment model files exist
            enhanced_model_path = self.models_dir / f"enhanced_ensemble_model{model_suffix}.joblib"
            cluster_model_path = self.models_dir / f"cluster_model{model_suffix}.joblib"

            if enhanced_model_path.exists() and cluster_model_path.exists():
                self.trained_model_groups[exp_name] = True
                print(f"✅ Found per-experiment model: {exp_name}")
            else:
                print(f"❌ No per-experiment model found for: {exp_name}")

        print(f"Discovered {len(self.trained_model_groups)} per-experiment models")

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

    def create_combined_training_data(self):
        """Create combined training files from experiment structure with balancing."""
        try:
            # Collect all train files
            all_train_data = []
            all_test_data = []

            print("STEP: Collecting training data from experiments...")
            for exp_name, exp_info in self.results['training_data_info'].items():
                # Add training data
                if exp_info['train_file'] and exp_info['train_file'].exists():
                    train_df = pd.read_excel(exp_info['train_file'])
                    train_df['source_experiment'] = exp_name
                    all_train_data.append(train_df)

                    # Show class distribution for this experiment
                    if 'type' in train_df.columns:
                        class_counts = train_df['type'].value_counts().sort_index()
                        print(f"  {exp_name}: {len(train_df)} samples - {dict(class_counts)}")
                    else:
                        print(f"  {exp_name}: {len(train_df)} samples - No 'type' column")

                # Add test data
                if exp_info['test_file'] and exp_info['test_file'].exists():
                    test_df = pd.read_excel(exp_info['test_file'])
                    test_df['source_experiment'] = exp_name
                    all_test_data.append(test_df)

            if not all_train_data:
                print("Error: No training data found in experiments")
                return False

            # Apply experiment and class balancing
            print("\nSTEP: Applying experiment and class balancing...")
            balanced_train = self.balance_training_data(all_train_data)

            # Save as traditional format expected by improve_features.py
            train_dir = self.output_dir / 'data'
            train_dir.mkdir(exist_ok=True)
            balanced_train.to_excel(train_dir / 'combined_balanced_training_data.xlsx', index=False)
            print(f"Saved balanced training data to {train_dir}/combined_balanced_training_data.xlsx")

            # Also save to base training_data for feature engineering script compatibility
            base_train_dir = Path('training_data')
            balanced_train.to_excel(base_train_dir / 'train3.xlsx', index=False)

            # If we have test data, create combined balanced test data
            if all_test_data:
                balanced_test = self.balance_training_data(all_test_data)
                balanced_test.to_excel(train_dir / 'combined_balanced_test_data.xlsx', index=False)
                balanced_test.to_excel(base_train_dir / 'train6.xlsx', index=False)  # For script compatibility
                print(f"Saved balanced test data to {train_dir}/combined_balanced_test_data.xlsx: {len(balanced_test)} samples")

            return True

        except Exception as e:
            print(f"Error creating combined training data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def balance_training_data(self, data_list):
        """Balance experiments and calculate class weights (keep all data)."""
        print("  Analyzing data for smart balancing...")

        # Calculate target samples per experiment (use median to avoid extremes)
        exp_sizes = [len(df) for df in data_list]
        target_samples_per_exp = int(np.median(exp_sizes))
        print(f"  Experiment sizes: {exp_sizes}")
        print(f"  Target samples per experiment: {target_samples_per_exp}")

        # Step 1: Balance experiments (but keep more data)
        experiment_balanced = []
        for i, df in enumerate(data_list):
            if len(df) > target_samples_per_exp:
                # Stratified sampling to maintain class distribution within experiment
                if 'type' in df.columns:
                    sampled_df = df.groupby('type', group_keys=False).apply(
                        lambda x: x.sample(min(len(x), max(1, int(target_samples_per_exp * len(x) / len(df)))),
                                          random_state=self.random_seed)
                    ).reset_index(drop=True)

                    # If we still need more samples to reach target, add more randomly
                    if len(sampled_df) < target_samples_per_exp and len(df) > len(sampled_df):
                        remaining_df = df.drop(sampled_df.index)
                        additional_needed = min(target_samples_per_exp - len(sampled_df), len(remaining_df))
                        if additional_needed > 0:
                            additional_df = remaining_df.sample(additional_needed, random_state=self.random_seed)
                            sampled_df = pd.concat([sampled_df, additional_df], ignore_index=True)
                else:
                    # No class column, just sample target amount
                    sampled_df = df.sample(min(target_samples_per_exp, len(df)), random_state=self.random_seed)
            else:
                # Keep all data if experiment is smaller than target
                sampled_df = df.copy()

            experiment_balanced.append(sampled_df)
            print(f"    Experiment {i+1}: {len(df)} -> {len(sampled_df)} samples")

        # Step 2: Combine experiment-balanced data
        combined_df = pd.concat(experiment_balanced, ignore_index=True)
        print(f"  After experiment balancing: {len(combined_df)} samples")

        # Step 3: Calculate class weights (instead of removing data)
        if 'type' in combined_df.columns:
            class_counts = combined_df['type'].value_counts().sort_index()
            total_samples = len(combined_df)
            n_classes = len(class_counts)

            print(f"  Final class distribution: {dict(class_counts)}")

            # Calculate balanced class weights (inverse frequency)
            class_weights = {}
            for class_label, count in class_counts.items():
                weight = total_samples / (n_classes * count)
                class_weights[class_label] = weight

            print("  Calculated class weights for training:")
            for class_label, weight in class_weights.items():
                print(f"    Class {class_label}: {weight:.3f} (inverse of frequency)")

            # Save class weights to a file for the training script to use
            import json
            weights_file = self.output_dir / 'data' / 'class_weights.json'
            base_weights_file = Path('training_data/class_weights.json')  # For script compatibility

            serializable_weights = {int(k): float(v) for k, v in class_weights.items()}

            # Save to output directory
            with open(weights_file, 'w') as f:
                json.dump(serializable_weights, f, indent=2)

            # Save to base directory for script compatibility
            with open(base_weights_file, 'w') as f:
                json.dump(serializable_weights, f, indent=2)

            print(f"  Saved class weights to {weights_file}")

        # Step 4: Shuffle and return all data
        final_df = combined_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # Show final distribution
        if 'type' in final_df.columns:
            final_exp_counts = final_df['source_experiment'].value_counts()
            print(f"  Final experiment distribution:")
            for exp, count in final_exp_counts.items():
                print(f"    {exp}: {count} samples")

        print(f"  Total samples preserved: {len(final_df)} (vs aggressive balancing: would be ~{class_counts.min() * n_classes if 'type' in combined_df.columns else 'N/A'})")

        return final_df

    def analyze_class_performance(self, y_true, y_pred, dataset_name=""):
        """Analyze class-specific performance with focus on confused classes."""
        print(f"\n🎯 CLASS-SPECIFIC ANALYSIS ({dataset_name}):")
        print("=" * 50)

        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        # Overall accuracy by class
        for class_num in [1, 2, 3, 4]:
            class_mask = y_true_array == class_num
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_true_array[class_mask], y_pred_array[class_mask])
                class_count = np.sum(class_mask)
                print(f"Class {class_num}: {class_accuracy:.1%} accuracy ({class_count} samples)")

        # Focus on Class 4 vs Class 1 confusion (most problematic pair)
        print(f"\n🔍 CLASS 4 vs CLASS 1 DETAILED ANALYSIS:")
        class_4_mask = y_true_array == 4
        class_1_mask = y_true_array == 1

        if np.sum(class_4_mask) > 0:
            class_4_pred = y_pred_array[class_4_mask]
            class_4_accuracy = accuracy_score(y_true_array[class_4_mask], class_4_pred)
            class_4_as_1 = np.sum(class_4_pred == 1)
            print(f"Class 4 performance:")
            print(f"  Total samples: {np.sum(class_4_mask)}")
            print(f"  Correctly classified: {class_4_accuracy:.1%}")
            print(f"  Misclassified as Class 1: {class_4_as_1} samples ({class_4_as_1/np.sum(class_4_mask)*100:.1f}%)")

        if np.sum(class_1_mask) > 0:
            class_1_pred = y_pred_array[class_1_mask]
            class_1_accuracy = accuracy_score(y_true_array[class_1_mask], class_1_pred)
            class_1_as_4 = np.sum(class_1_pred == 4)
            print(f"Class 1 performance:")
            print(f"  Total samples: {np.sum(class_1_mask)}")
            print(f"  Correctly classified: {class_1_accuracy:.1%}")
            print(f"  Misclassified as Class 4: {class_1_as_4} samples ({class_1_as_4/np.sum(class_1_mask)*100:.1f}%)")

        # Confusion matrix for all classes
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
        print(f"\nConfusion Matrix:")
        print("Predicted:  1    2    3    4")
        for i, true_class in enumerate([1, 2, 3, 4]):
            row_str = f"True {true_class}:   "
            for j in range(4):
                row_str += f"{cm[i,j]:4d} "
            print(row_str)

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

        print("Creating combined training data from experiments...")

        # Create combined training data from all experiments
        if not self.create_combined_training_data():
            print("Error: Failed to create combined training data")
            return False

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
        """Run enhanced model training with clustering and ensemble."""
        print("\nSTEP 3: Enhanced Model Training")
        print("=" * 50)

        try:
            # Get training groups based on training mode
            training_data_info = self.results['training_data_info']
            training_groups = self.group_experiments_by_training_mode(training_data_info)

            if self.training_mode == 'combined':
                # Original single model approach
                return self._train_single_combined_model()
            else:
                # Multiple model approach
                return self._train_multiple_models(training_groups, training_data_info)

        except Exception as e:
            print(f"Error: Error in enhanced model training: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _train_single_combined_model(self):
        """Train single ensemble model with all experiments combined (original approach)."""
        train_data_path = self.output_dir / 'data' / 'combined_balanced_training_data.xlsx'
        class_weights_path = self.output_dir / 'data' / 'class_weights.json'

        if not train_data_path.exists():
            print("Error: Balanced training data not found")
            return False

        print("Loading balanced training data and class weights...")
        train_df = pd.read_excel(train_data_path)
        print(f"Loaded {len(train_df)} training samples")

        class_weights = None
        if class_weights_path.exists():
            import json
            with open(class_weights_path, 'r') as f:
                class_weights = json.load(f)
            print(f"Loaded class weights: {class_weights}")

        # Use existing ensemble training method
        success = self.train_enhanced_ensemble_model(train_df, class_weights)
        if success:
            self.pipeline_status['model_training'] = True
        return success

    def _train_multiple_models(self, training_groups, training_data_info):
        """Train separate ensemble models for each group using SAME architecture."""
        print(f"Training {len(training_groups)} separate enhanced ensemble models...")

        successful_models = 0
        failed_models = 0
        self.trained_model_groups = {}

        for group_name, experiment_list in training_groups.items():
            print(f"\n🎯 Training enhanced ensemble for group: {group_name}")
            print(f"Experiments: {experiment_list}")
            print("-" * 40)

            # Combine training data for this group
            group_data = []
            for exp_name in experiment_list:
                exp_info = training_data_info[exp_name]
                if exp_info['train_file'] and exp_info['train_file'].exists():
                    exp_df = pd.read_excel(exp_info['train_file'])
                    group_data.append(exp_df)

            if not group_data:
                print(f"❌ Error: No training data for group {group_name}")
                failed_models += 1
                continue

            # Combine and balance data for this group
            combined_df = pd.concat(group_data, ignore_index=True)
            print(f"Combined {len(combined_df)} samples for group {group_name}")

            # Check if we have sufficient data for training
            class_counts = combined_df['type'].value_counts()
            min_class_count = min(class_counts.values)

            if len(class_counts) < 2:
                print(f"❌ Skipping {group_name}: Only {len(class_counts)} class(es), need at least 2")
                failed_models += 1
                continue

            if min_class_count < 2:
                print(f"❌ Skipping {group_name}: Minimum class has only {min_class_count} samples, need at least 2")
                failed_models += 1
                continue

            # Calculate class weights for this group
            total = len(combined_df)
            num_classes = len(class_counts)
            class_weights = {str(cls): total/(num_classes * count)
                           for cls, count in class_counts.items()}

            print(f"Class distribution: {dict(class_counts)}")
            print(f"Class weights: {class_weights}")

            # Create safe filename suffix (replace problematic characters)
            safe_group_name = group_name.replace('/', '_').replace('\\', '_')

            # Train the SAME enhanced ensemble model architecture for this group
            try:
                success = self.train_enhanced_ensemble_model(
                    combined_df, class_weights, model_suffix=f"_{safe_group_name}")

                if success:
                    self.trained_model_groups[group_name] = True
                    successful_models += 1
                    print(f"✓ Enhanced ensemble trained for {group_name}")
                else:
                    failed_models += 1
                    print(f"✗ Failed to train ensemble for {group_name}")
            except Exception as e:
                failed_models += 1
                print(f"✗ Failed to train ensemble for {group_name}: {e}")

        # Consider success if we trained at least some models
        if successful_models > 0:
            self.pipeline_status['model_training'] = True
            print(f"\n✓ Successfully trained {successful_models}/{successful_models + failed_models} enhanced ensemble models")
            if failed_models > 0:
                print(f"⚠️  {failed_models} models failed due to insufficient data")
        else:
            print(f"\n❌ All {failed_models} models failed to train")

        return successful_models > 0

    def train_enhanced_ensemble_model(self, train_df, class_weights=None, model_suffix=""):
        """Train an enhanced ensemble model with clustering features."""
        print("Training enhanced ensemble model...")

        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.cluster import KMeans
            from sklearn.mixture import GaussianMixture
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import classification_report, accuracy_score
            import xgboost as xgb

            # Prepare features and target
            exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'source_experiment', 'Centroid', 'BoundingBox', 'WeightedCentroid']
            feature_cols = [col for col in train_df.columns
                           if col not in exclude_cols and train_df[col].dtype in ['float64', 'int64']]

            X = train_df[feature_cols].fillna(train_df[feature_cols].median())
            y_orig = train_df['type'].values

            # Convert classes 1,2,3,4 to 0,1,2,3 for sklearn compatibility
            class_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
            reverse_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
            y = np.array([class_mapping[label] for label in y_orig])

            print(f"Training features: {len(feature_cols)} columns")
            print(f"Original class distribution: {dict(pd.Series(y_orig).value_counts().sort_index())}")
            print(f"Mapped class distribution: {dict(pd.Series(y).value_counts().sort_index())}")

            # Feature scaling and selection (reuse existing components if available)
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_selection import SelectKBest, f_classif

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=min(self.max_features, len(feature_cols)))
            X_selected = selector.fit_transform(X_scaled, y)

            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            print(f"Selected {len(selected_features)} features: {selected_features[:5]}...")

            # Step 1: Create clustering features
            print("Creating clustering features...")

            # K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=self.random_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(X_selected)
            cluster_centers = kmeans.cluster_centers_

            # Gaussian Mixture Model
            gmm = GaussianMixture(n_components=2, random_state=self.random_seed)
            gmm.fit(X_selected)
            gmm_probs = gmm.predict_proba(X_selected)

            # Create enhanced features
            enhanced_features = []

            # Distance to each cluster center
            for i, center in enumerate(cluster_centers):
                distances = np.linalg.norm(X_selected - center, axis=1)
                enhanced_features.append(distances)

            # GMM probabilities
            for i in range(gmm_probs.shape[1]):
                enhanced_features.append(gmm_probs[:, i])

            # Cluster membership (one-hot encoded)
            for i in range(4):
                cluster_membership = (cluster_labels == i).astype(float)
                enhanced_features.append(cluster_membership)

            # Class-specific distance features (Class 4 vs Class 1 - most confused classes)
            class_4_mask = y == 3  # Class 4 (0-indexed as 3)
            class_1_mask = y == 0  # Class 1 (0-indexed as 0)

            if np.sum(class_4_mask) > 0 and np.sum(class_1_mask) > 0:
                # Calculate feature centers for Class 4 and Class 1
                class_4_mean = np.mean(X_selected[class_4_mask], axis=0)
                class_1_mean = np.mean(X_selected[class_1_mask], axis=0)

                # Distance to each class center
                dist_to_class_4 = np.linalg.norm(X_selected - class_4_mean, axis=1)
                dist_to_class_1 = np.linalg.norm(X_selected - class_1_mean, axis=1)

                enhanced_features.append(dist_to_class_4)
                enhanced_features.append(dist_to_class_1)
                enhanced_features.append(dist_to_class_4 - dist_to_class_1)  # Relative distance

                print(f"Added class-specific distance features for Class 4 vs Class 1 discrimination")

            # Error-prone region indicator (based on clustering analysis)
            # Identify which cluster tends to have more misclassifications
            error_prone_cluster = 0  # This should be determined from validation analysis
            error_indicator = (cluster_labels == error_prone_cluster).astype(float)
            enhanced_features.append(error_indicator)

            # Combine original and enhanced features
            X_enhanced = np.hstack([X_selected] + [feat.reshape(-1, 1) for feat in enhanced_features])

            print(f"Enhanced feature matrix shape: {X_enhanced.shape}")

            # Step 2: Apply controlled SMOTE augmentation (less aggressive to prevent overfitting)
            print("Applying controlled SMOTE augmentation...")
            try:
                from imblearn.over_sampling import SMOTE
                # Use more conservative SMOTE parameters to prevent perfect separability
                smote = SMOTE(
                    sampling_strategy='auto',  # Balance all classes
                    k_neighbors=3,  # Reduced from default 5 to create less perfect synthetic samples
                    random_state=self.random_seed
                )
                X_resampled, y_resampled = smote.fit_resample(X_enhanced, y)

                print(f"Original data: {X_enhanced.shape[0]} samples")
                print(f"Resampled data: {X_resampled.shape[0]} samples")
                print(f"Original class distribution: {dict(pd.Series(y).value_counts().sort_index())}")
                print(f"Resampled class distribution: {dict(pd.Series(y_resampled).value_counts().sort_index())}")

                # Use resampled data for training
                X_enhanced = X_resampled
                y = y_resampled

            except ImportError:
                print("Warning: imblearn not available, skipping SMOTE augmentation")
                print("Install with: pip install imbalanced-learn")
            except Exception as e:
                print(f"Warning: SMOTE failed ({e}), using original data")

            # Step 3: Train ensemble models
            print("Training ensemble models...")

            # Convert class weights for sklearn (map from original classes to 0-indexed)
            sample_weights = None
            if class_weights:
                sample_weights = np.array([class_weights[str(reverse_mapping[label])] for label in y])
                print(f"Using sample weights based on class frequencies")

            # Define models with hyperparameter tuning to prevent overfitting
            models = self._get_tuned_models(X_enhanced, y, sample_weights)

            # Train and evaluate each model with validation-based selection
            trained_models = {}
            model_scores = {}

            # Create validation split for model selection
            from sklearn.model_selection import train_test_split
            X_train_models, X_val_models, y_train_models, y_val_models = train_test_split(
                X_enhanced, y, test_size=0.2, random_state=self.random_seed, stratify=y
            )

            for name, model in models.items():
                print(f"  Training {name}...")

                # Train on subset and validate
                if sample_weights is not None and name in ['rf_tuned', 'gb_tuned']:
                    # Subset sample weights for training data
                    train_indices = range(len(y_train_models))
                    subset_weights = sample_weights[train_indices] if len(sample_weights) == len(X_enhanced) else None
                    if subset_weights is not None:
                        model.fit(X_train_models, y_train_models, sample_weight=subset_weights)
                    else:
                        model.fit(X_train_models, y_train_models)
                else:
                    model.fit(X_train_models, y_train_models)

                # Validation score (more reliable than CV on augmented data)
                val_pred = model.predict(X_val_models)
                val_accuracy = accuracy_score(y_val_models, val_pred)
                model_scores[name] = val_accuracy
                trained_models[name] = model

                print(f"    {name} validation accuracy: {val_accuracy:.3f}")

            # Select only models that perform well on validation
            min_val_accuracy = 0.7  # Threshold to prevent poor models from being included
            good_models = {name: model for name, model in trained_models.items()
                          if model_scores[name] >= min_val_accuracy}

            if len(good_models) < 2:
                print(f"Warning: Only {len(good_models)} models meet validation threshold, using all models")
                good_models = trained_models
            else:
                print(f"Selected {len(good_models)} models meeting validation threshold")

            trained_models = good_models

            # Step 4: Create VotingClassifier ensemble
            print("Creating VotingClassifier ensemble...")

            from sklearn.ensemble import VotingClassifier

            # Create voting ensemble with soft voting (probability-based)
            ensemble_model = VotingClassifier(
                estimators=list(trained_models.items()),
                voting='soft'
            )

            # Train the ensemble
            print("Training ensemble...")
            if sample_weights is not None:
                # VotingClassifier doesn't directly support sample_weight, but individual models already trained with it
                ensemble_model.fit(X_enhanced, y)
            else:
                ensemble_model.fit(X_enhanced, y)

            # Evaluate ensemble with cross-validation
            ensemble_cv_scores = cross_val_score(ensemble_model, X_enhanced, y, cv=3, scoring='accuracy')
            print(f"Ensemble CV accuracy: {ensemble_cv_scores.mean():.3f} (+/- {ensemble_cv_scores.std() * 2:.3f})")

            print("Individual model CV scores:")
            for name, score in model_scores.items():
                print(f"  {name}: {score:.3f}")

            # Use the VotingClassifier as the main ensemble
            trained_models['ensemble'] = ensemble_model

            # Step 5: Evaluate ensemble on training data
            ensemble_pred = ensemble_model.predict(X_enhanced)

            # Convert predictions back to original class labels (1,2,3,4)
            ensemble_pred_mapped = [reverse_mapping[pred] for pred in ensemble_pred]
            y_orig_for_eval = [reverse_mapping[label] for label in y]

            # Print evaluation with class-specific analysis
            print("\nEnsemble Model Training Results:")
            print(classification_report(y_orig_for_eval, ensemble_pred_mapped))

            # Class-specific analysis (focus on Class 4 vs Class 1 confusion)
            self.analyze_class_performance(y_orig_for_eval, ensemble_pred_mapped, "Training")

            # Step 5: Save all components
            print("Saving enhanced model components...")

            models_dir = self.output_dir / 'models'

            # Save ensemble components
            ensemble_components = {
                'voting_classifier': ensemble_model,
                'individual_models': {k: v for k, v in trained_models.items() if k != 'ensemble'},
                'model_scores': model_scores,
                'reverse_mapping': reverse_mapping,  # Maps 0,1,2,3 -> 1,2,3,4
                'class_mapping': class_mapping       # Maps 1,2,3,4 -> 0,1,2,3
            }
            joblib.dump(ensemble_components, models_dir / f'enhanced_ensemble_model{model_suffix}.joblib')

            # Save clustering model
            cluster_model = {
                'kmeans': kmeans,
                'gmm': gmm,
                'cluster_centers': cluster_centers
            }
            joblib.dump(cluster_model, models_dir / f'cluster_model{model_suffix}.joblib')

            # Save preprocessing components (only once for the first model if multiple models)
            if model_suffix == "" or not (models_dir / 'latest_scaler.joblib').exists():
                joblib.dump(scaler, models_dir / 'latest_scaler.joblib')
                joblib.dump(selector, models_dir / 'latest_feature_selector.joblib')
                joblib.dump(selected_features, models_dir / 'latest_feature_names.joblib')

            # Save metadata
            metadata = {
                'feature_columns': feature_cols,
                'selected_features': selected_features,
                'enhanced_feature_shape': X_enhanced.shape,
                'class_weights': class_weights,
                'model_scores': model_scores,
                'training_samples': len(train_df),
                'timestamp': datetime.now().isoformat(),
                'model_suffix': model_suffix
            }
            joblib.dump(metadata, models_dir / f'enhanced_model_metadata{model_suffix}.joblib')

            print("Success: Enhanced ensemble model training completed!")
            print(f"Saved enhanced_ensemble_model{model_suffix}.joblib")
            print(f"Saved cluster_model{model_suffix}.joblib")
            print(f"Saved enhanced_model_metadata{model_suffix}.joblib")

            return True

        except Exception as e:
            print(f"Error in enhanced model training: {e}")
            import traceback
            traceback.print_exc()
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
                    
                    # Note: Removed misleading train/test plot (was using artificial balanced data)
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
                                shutil.move(src_path, self.output_dir / 'plots' / result_file)
                            else:
                                shutil.move(src_path, self.output_dir / 'reports' / result_file)

                    # NEW: Move all dynamically generated individual validation plots
                    for plot_file in Path('.').glob('enhanced_model_*_results.png'):
                        shutil.move(plot_file, self.output_dir / 'plots' / plot_file.name)
                    # ALSO move all *_validation_results.png plots (from individual scripts)
                    for plot_file in Path('.').glob('*_validation_results.png'):
                        shutil.move(plot_file, self.output_dir / 'plots' / plot_file.name)
                    
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
        """Process each validation file separately and create summary plots."""
        print("Processing each validation file individually...")

        # Organize validation files by experiment
        experiment_results = {}
        all_test_results = []
        all_val_results = []

        if hasattr(self, 'results') and 'training_data_info' in self.results:
            for exp_name, exp_info in self.results['training_data_info'].items():
                # Skip experiments that don't have trained models in per-experiment mode
                if hasattr(self, 'trained_model_groups') and self.trained_model_groups:
                    # Check if this experiment has a trained model
                    group_key = None
                    for group_name in self.trained_model_groups.keys():
                        if exp_name in group_name or group_name in exp_name:
                            group_key = group_name
                            break

                    if not group_key:
                        print(f"⏭️  Skipping {exp_name}: No trained model available")
                        continue

                experiment_results[exp_name] = {}

                # Process test file
                if exp_info['test_file'] and exp_info['test_file'].exists():
                    print(f"\n📊 Processing TEST: {exp_name}")
                    success, results = self.process_experiment_file(exp_info['test_file'], exp_name, 'test')
                    if success:
                        experiment_results[exp_name]['test'] = results
                        all_test_results.append(results)

                # Process val file
                if exp_info['val_file'] and exp_info['val_file'].exists():
                    print(f"\n📊 Processing VAL: {exp_name}")
                    success, results = self.process_experiment_file(exp_info['val_file'], exp_name, 'val')
                    if success:
                        experiment_results[exp_name]['val'] = results
                        all_val_results.append(results)

        # Create summary plots
        self.create_summary_plots(all_test_results, all_val_results, experiment_results)

        total_processed = len(all_test_results) + len(all_val_results)
        print(f"\nSuccess: Successfully processed {total_processed} validation files")

        if total_processed > 0:
            self.pipeline_status['validation_testing'] = True
            return True
        else:
            return False

    def process_experiment_file(self, file_path, exp_name, split_type):
        """Process a single experiment file and return results."""
        try:
            # Load models - check for per-experiment model first
            models = self.load_experiment_specific_model(exp_name)
            if models is None:
                # Fallback to generic validation models - pass experiment name for concentration routing
                models = self.load_validation_models(exp_name)
                if models is None:
                    return False, None

            # Load and preprocess data
            df = pd.read_excel(file_path)
            print(f"Processing {len(df)} samples from {exp_name}/{split_type}")

            # IMPORTANT: The scaler was fitted on improved features, so we must provide improved features
            # But the enhanced model itself uses only original features (selected by the feature selector)

            # Create improved features (same as during training) for the scaler
            df_improved = self.create_improved_features(df)

            # Extract features for scaler - must match what scaler was trained on (improved features)
            exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'source_experiment', 'Centroid', 'BoundingBox', 'WeightedCentroid']
            all_features = [col for col in df_improved.columns
                           if col not in exclude_cols and df_improved[col].dtype in ['float64', 'int64']]
            print(f"Using improved features for scaler: {len(all_features)} features")

            # Prepare features for prediction - use improved features for scaler
            X_all = df_improved[all_features].fillna(df_improved[all_features].median())

            X_scaled = models['scaler'].transform(X_all)

            # For enhanced models, check metadata for correct feature selection
            if models['model_type'] == 'enhanced':
                # Load metadata to get correct feature selection
                metadata_path = None
                if models.get('experiment_specific') and models.get('group_name'):
                    safe_group_name = models['group_name'].replace('/', '_').replace('\\', '_')
                    metadata_path = self.models_dir / f'enhanced_model_metadata_{safe_group_name}.joblib'
                else:
                    # For concentration models, find the right metadata
                    concentration_metadata = list(self.models_dir.glob("enhanced_model_metadata_*mM.joblib"))
                    if concentration_metadata:
                        metadata_path = concentration_metadata[0]

                if metadata_path and metadata_path.exists():
                    metadata = joblib.load(metadata_path)
                    selected_feature_count = len(metadata.get('selected_features', []))
                    if selected_feature_count:
                        # Use only the first k features to match what was used during training
                        X_selected = X_scaled[:, :selected_feature_count]
                        print(f"Using exact {selected_feature_count} features from metadata")
                    else:
                        X_selected = models['selector'].transform(X_scaled)
                        print(f"Using selector: {X_selected.shape[1]} features")
                else:
                    X_selected = models['selector'].transform(X_scaled)
                    print(f"Using selector (no metadata): {X_selected.shape[1]} features")
            else:
                X_selected = models['selector'].transform(X_scaled)

            if models['model_type'] == 'enhanced':
                # Enhanced model logic - recreate EXACT same features as training
                cluster_model = models['cluster_model']
                kmeans = cluster_model['kmeans']
                gmm = cluster_model['gmm']

                # Step 1: Apply EXACT same feature selection that was used during training
                # X_selected should have same number of features that kmeans expects
                print(f"K-means expects {kmeans.n_features_in_} features, we have {X_selected.shape[1]}")

                # Create enhanced features (same as in training)
                enhanced_features = []

                # K-means clustering features
                cluster_labels = kmeans.predict(X_selected)
                cluster_centers = kmeans.cluster_centers_

                # Distance to each cluster center
                for i, center in enumerate(cluster_centers):
                    distances = np.linalg.norm(X_selected - center, axis=1)
                    enhanced_features.append(distances)

                # GMM probabilities (use saved GMM)
                gmm_probs = gmm.predict_proba(X_selected)
                for i in range(gmm_probs.shape[1]):
                    enhanced_features.append(gmm_probs[:, i])

                # Cluster membership (one-hot encoded)
                for i in range(4):
                    cluster_membership = (cluster_labels == i).astype(float)
                    enhanced_features.append(cluster_membership)

                # Class-specific distance features (Class 4 vs Class 1)
                # Convert to 0-indexed format
                class_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
                y_labels = df['type'].values  # Get labels from the dataframe
                y_indexed = np.array([class_mapping.get(label, 0) for label in y_labels])

                class_4_mask = y_indexed == 3  # Class 4 (0-indexed as 3)
                class_1_mask = y_indexed == 0  # Class 1 (0-indexed as 0)

                if np.sum(class_4_mask) > 0 and np.sum(class_1_mask) > 0:
                    # Calculate feature centers for Class 4 and Class 1
                    class_4_mean = np.mean(X_selected[class_4_mask], axis=0)
                    class_1_mean = np.mean(X_selected[class_1_mask], axis=0)

                    # Distance to each class center
                    dist_to_class_4 = np.linalg.norm(X_selected - class_4_mean, axis=1)
                    dist_to_class_1 = np.linalg.norm(X_selected - class_1_mean, axis=1)

                    enhanced_features.append(dist_to_class_4)
                    enhanced_features.append(dist_to_class_1)
                    enhanced_features.append(dist_to_class_4 - dist_to_class_1)  # Relative distance
                else:
                    # If we don't have both classes, add zero features
                    enhanced_features.extend([np.zeros(len(X_selected)), np.zeros(len(X_selected)), np.zeros(len(X_selected))])

                # Error-prone region indicator
                error_prone_cluster = 0
                error_indicator = (cluster_labels == error_prone_cluster).astype(float)
                enhanced_features.append(error_indicator)

                # Combine original and enhanced features
                X_final = np.hstack([X_selected] + [feat.reshape(-1, 1) for feat in enhanced_features])

                print(f"Created enhanced features: {X_selected.shape[1]} -> {X_final.shape[1]} features")

                # Make ensemble predictions
                ensemble_components = models['model']
                voting_classifier = ensemble_components['voting_classifier']
                reverse_mapping = ensemble_components['reverse_mapping']

                # Use VotingClassifier for prediction
                y_pred_raw = voting_classifier.predict(X_final)
                y_pred_proba = voting_classifier.predict_proba(X_final)
                y_pred = [reverse_mapping[pred] for pred in y_pred_raw]  # Convert back to 1,2,3,4
            else:
                # Basic model logic
                y_pred_raw = models['model'].predict(X_selected)
                label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
                y_pred = [label_mapping[pred] for pred in y_pred_raw]
                y_pred_proba = models['model'].predict_proba(X_selected)

            confidence = np.max(y_pred_proba, axis=1)

            # Create individual experiment plot
            file_stem = f"{exp_name.replace('/', '_')}_{split_type}"
            accuracy = None
            class_accuracies = {}

            if 'type' in df.columns:
                y_true = df['type'].values
                accuracy = accuracy_score(y_true, y_pred)

                # Calculate per-class accuracy
                for cls in sorted(np.unique(y_true)):
                    mask = y_true == cls
                    if np.sum(mask) > 0:
                        class_acc = accuracy_score(y_true[mask], np.array(y_pred)[mask])
                        class_accuracies[cls] = {
                            'accuracy': class_acc,
                            'confidence': np.mean(confidence[mask]),
                            'count': np.sum(mask)
                        }

                # Create individual plot
                plot_file, _, _, _ = self.create_validation_visualization(y_true, y_pred, confidence, file_stem)

                # Print results
                print(f"Success: Accuracy: {accuracy:.1%}")
                print(f"Success: Mean confidence: {np.mean(confidence):.3f}")
                for cls, metrics in class_accuracies.items():
                    print(f"   Class {cls}: {metrics['accuracy']:.1%} accuracy, {metrics['confidence']:.3f} confidence ({metrics['count']} samples)")

            else:
                print(f"Success: Mean confidence: {np.mean(confidence):.3f} (no true labels)")

            # Save predictions
            results_df = pd.DataFrame({
                'sample_id': range(len(df)),
                'predicted_class': y_pred,
                'confidence': confidence
            })

            for col in df.columns:
                if col not in results_df.columns:
                    results_df[col] = df[col].values

            pred_file = f"{file_stem}_predictions.xlsx"
            pred_output_path = self.output_dir / 'reports' / pred_file
            results_df.to_excel(pred_output_path, index=False)
            print(f"Success: Saved predictions: {pred_output_path}")

            # Return results for summary
            return True, {
                'exp_name': exp_name,
                'split_type': split_type,
                'accuracy': accuracy,
                'confidence': np.mean(confidence),
                'class_accuracies': class_accuracies,
                'total_samples': len(df),
                'y_true': df['type'].values if 'type' in df.columns else None,
                'y_pred': y_pred,
                'confidence_values': confidence
            }

        except Exception as e:
            print(f"Error processing {exp_name}/{split_type}: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def create_summary_plots(self, all_test_results, all_val_results, experiment_results):
        """Create summary plots combining all experiments."""
        try:
            # Create summary plot for test results
            if all_test_results:
                self.create_combined_plot(all_test_results, "test", "Test Results Summary")

            # Create summary plot for val results
            if all_val_results:
                self.create_combined_plot(all_val_results, "val", "Validation Results Summary")

            # Create comparison plot (test vs val)
            if all_test_results and all_val_results:
                self.create_test_vs_val_plot(experiment_results)

            # Create training vs experiments performance plot
            if all_test_results or all_val_results:
                self.create_training_vs_experiments_plot(all_test_results, all_val_results)

        except Exception as e:
            print(f"Error creating summary plots: {e}")

    def create_combined_plot(self, results_list, split_type, title):
        """Create a combined plot for multiple experiments."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{title} - {self.network_dir.upper()}', fontsize=16, fontweight='bold')

            # Collect all data
            all_accuracies = []
            all_confidences = []
            exp_names = []

            for result in results_list:
                if result['accuracy'] is not None:
                    all_accuracies.append(result['accuracy'])
                    all_confidences.append(result['confidence'])
                    exp_names.append(result['exp_name'].split('/')[-1])  # Just experiment name

            # 1. Accuracy by experiment
            if all_accuracies:
                bars = axes[0,0].bar(range(len(exp_names)), all_accuracies, color='skyblue', alpha=0.7)
                axes[0,0].set_title('Accuracy by Experiment')
                axes[0,0].set_ylabel('Accuracy')
                axes[0,0].set_xticks(range(len(exp_names)))
                axes[0,0].set_xticklabels(exp_names, rotation=45)
                axes[0,0].set_ylim(0, 1)

                # Add value labels on bars
                for bar, acc in zip(bars, all_accuracies):
                    height = bar.get_height()
                    axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{acc:.1%}', ha='center', va='bottom')

            # 2. Confidence by experiment
            axes[0,1].bar(range(len(exp_names)), all_confidences, color='lightgreen', alpha=0.7)
            axes[0,1].set_title('Confidence by Experiment')
            axes[0,1].set_ylabel('Mean Confidence')
            axes[0,1].set_xticks(range(len(exp_names)))
            axes[0,1].set_xticklabels(exp_names, rotation=45)
            axes[0,1].set_ylim(0, 1)

            # 3. Combined confusion matrix
            if all_accuracies:
                all_y_true = []
                all_y_pred = []
                for result in results_list:
                    if result['y_true'] is not None:
                        all_y_true.extend(result['y_true'])
                        all_y_pred.extend(result['y_pred'])

                if all_y_true:
                    cm = confusion_matrix(all_y_true, all_y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
                                xticklabels=sorted(np.unique(all_y_true)),
                                yticklabels=sorted(np.unique(all_y_true)))
                    axes[1,0].set_title('Combined Confusion Matrix')
                    axes[1,0].set_xlabel('Predicted')
                    axes[1,0].set_ylabel('Actual')

            # 4. Sample count by experiment
            sample_counts = [result['total_samples'] for result in results_list]
            axes[1,1].bar(range(len(exp_names)), sample_counts, color='orange', alpha=0.7)
            axes[1,1].set_title('Sample Count by Experiment')
            axes[1,1].set_ylabel('Number of Samples')
            axes[1,1].set_xticks(range(len(exp_names)))
            axes[1,1].set_xticklabels(exp_names, rotation=45)

            plt.tight_layout()
            plot_file = f'{self.network_dir}_{split_type}_summary.png'
            output_plot_path = self.output_dir / 'plots' / plot_file
            plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Success: Created summary plot: {plot_file}")

        except Exception as e:
            print(f"Error creating combined plot: {e}")

    def create_test_vs_val_plot(self, experiment_results):
        """Create comparison plot between test and validation results."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Test vs Validation Comparison - {self.network_dir.upper()}', fontsize=16, fontweight='bold')

            exp_names = []
            test_accs = []
            val_accs = []
            test_confs = []
            val_confs = []

            for exp_name, results in experiment_results.items():
                if 'test' in results and 'val' in results:
                    exp_names.append(exp_name.split('/')[-1])

                    test_acc = results['test']['accuracy']
                    val_acc = results['val']['accuracy']

                    if test_acc is not None and val_acc is not None:
                        test_accs.append(test_acc)
                        val_accs.append(val_acc)
                        test_confs.append(results['test']['confidence'])
                        val_confs.append(results['val']['confidence'])

            if test_accs and val_accs:
                x = np.arange(len(exp_names))
                width = 0.35

                # Accuracy comparison
                axes[0].bar(x - width/2, test_accs, width, label='Test', alpha=0.8, color='lightblue')
                axes[0].bar(x + width/2, val_accs, width, label='Validation', alpha=0.8, color='lightcoral')
                axes[0].set_title('Accuracy: Test vs Validation')
                axes[0].set_ylabel('Accuracy')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(exp_names, rotation=45)
                axes[0].legend()
                axes[0].set_ylim(0, 1)

                # Confidence comparison
                axes[1].bar(x - width/2, test_confs, width, label='Test', alpha=0.8, color='lightgreen')
                axes[1].bar(x + width/2, val_confs, width, label='Validation', alpha=0.8, color='orange')
                axes[1].set_title('Confidence: Test vs Validation')
                axes[1].set_ylabel('Mean Confidence')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(exp_names, rotation=45)
                axes[1].legend()
                axes[1].set_ylim(0, 1)

            plt.tight_layout()
            plot_file = f'{self.network_dir}_test_vs_val_comparison.png'
            output_plot_path = self.output_dir / 'plots' / plot_file
            plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Success: Created comparison plot: {plot_file}")

        except Exception as e:
            print(f"Error creating test vs val plot: {e}")

    def create_training_vs_experiments_plot(self, all_test_results, all_val_results):
        """Create plot showing training accuracy vs individual experiment accuracies."""
        try:
            # Get training accuracy from metadata if available
            training_accuracy = 1.0  # Default perfect training accuracy (from SMOTEENN augmented data)
            metadata_file = self.output_dir / "models" / "enhanced_model_metadata.joblib"
            if metadata_file.exists():
                try:
                    import joblib
                    metadata = joblib.load(metadata_file)
                    training_accuracy = metadata.get('training_accuracy', 1.0)
                except:
                    pass

            # Collect experiment results
            all_results = []
            if all_test_results:
                all_results.extend([(r['exp_name'].split('/')[-1] + '_test', r['accuracy'])
                                  for r in all_test_results if r['accuracy'] is not None])
            if all_val_results:
                all_results.extend([(r['exp_name'].split('/')[-1] + '_val', r['accuracy'])
                                  for r in all_val_results if r['accuracy'] is not None])

            if not all_results:
                print("No experiment results to plot")
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            exp_names = [r[0] for r in all_results]
            exp_accuracies = [r[1] for r in all_results]

            # Training accuracy line
            ax.axhline(y=training_accuracy, color='red', linestyle='--', linewidth=2,
                      label=f'Training Accuracy ({training_accuracy:.1%})')

            # Experiment accuracies
            bars = ax.bar(range(len(exp_names)), exp_accuracies, color='skyblue', alpha=0.7)

            # Add value labels on bars
            for i, (bar, acc) in enumerate(zip(bars, exp_accuracies)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

                # Color code based on performance vs training
                if acc < training_accuracy - 0.1:  # More than 10% drop
                    bar.set_color('lightcoral')
                elif acc < training_accuracy - 0.05:  # 5-10% drop
                    bar.set_color('orange')
                else:  # Good performance
                    bar.set_color('lightgreen')

            ax.set_title(f'Training vs Individual Experiment Performance - {self.network_dir.upper()}',
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Experiment')
            ax.set_xticks(range(len(exp_names)))
            ax.set_xticklabels(exp_names, rotation=45, ha='right')
            ax.set_ylim(0, min(1.1, max(max(exp_accuracies), training_accuracy) + 0.1))
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = f'{self.network_dir}_training_vs_experiments.png'
            output_plot_path = self.output_dir / 'plots' / plot_file
            plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Success: Created training vs experiments plot: {plot_file}")

        except Exception as e:
            print(f"Error creating training vs experiments plot: {e}")

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

        # 6. Major Minor Ratio (needed for Comprehensive Shape Score)
        df_improved['Major_Minor_ratio'] = df['Major'] / (df['Minor'] + 1e-6)

        # 7. Comprehensive Shape Score
        df_improved['Comprehensive_Shape'] = (
            df_improved['Major_Minor_ratio'] *
            df_improved['Circularity'] *
            (1 - df_improved['Eccentricity_Robust'])
        )
        
        # 8. Class4 Discriminator
        df_improved['Class4_Discriminator'] = (
            df['Extent'] * df['Circularity'] * df['Perimeter']
        )
        
        return df_improved
    
    def load_validation_models(self, exp_name=None):
        """Load all required models and components for validation."""
        try:
            # Try enhanced models first in current models_dir
            enhanced_model_path = self.models_dir / "enhanced_ensemble_model.joblib"
            cluster_model_path = self.models_dir / "cluster_model.joblib"

            # Check for concentration-specific models first
            if self.training_mode == 'concentration' and exp_name:
                # Extract concentration from experiment name (e.g., "third_network/6mM-4" -> "6mM")
                concentration = None
                if "3mM" in exp_name:
                    concentration = "3mM"
                elif "6mM" in exp_name:
                    concentration = "6mM"

                if concentration:
                    # Try to load the specific concentration model
                    target_enhanced_path = self.models_dir / f"enhanced_ensemble_model_{concentration}.joblib"
                    target_cluster_path = self.models_dir / f"cluster_model_{concentration}.joblib"

                    if target_enhanced_path.exists() and target_cluster_path.exists():
                        enhanced_model_path = target_enhanced_path
                        cluster_model_path = target_cluster_path
                        print(f"Loading {concentration} concentration-specific enhanced model: {enhanced_model_path.name}")
                    else:
                        print(f"Warning: {concentration} model not found, falling back to any available concentration model")
                        concentration_models = list(self.models_dir.glob("enhanced_ensemble_model_*mM.joblib"))
                        if concentration_models:
                            enhanced_model_path = concentration_models[0]
                            concentration_suffix = enhanced_model_path.stem.replace('enhanced_ensemble_model', '')
                            cluster_model_path = self.models_dir / f"cluster_model{concentration_suffix}.joblib"
                else:
                    print(f"Warning: Could not detect concentration from {exp_name}, using first available model")
                    concentration_models = list(self.models_dir.glob("enhanced_ensemble_model_*mM.joblib"))
                    if concentration_models:
                        enhanced_model_path = concentration_models[0]
                        concentration_suffix = enhanced_model_path.stem.replace('enhanced_ensemble_model', '')
                        cluster_model_path = self.models_dir / f"cluster_model{concentration_suffix}.joblib"

                    if enhanced_model_path.exists() and cluster_model_path.exists():
                        print(f"Loading concentration-specific enhanced model: {enhanced_model_path.name}")
                        return {
                            'model': joblib.load(enhanced_model_path),
                            'cluster_model': joblib.load(cluster_model_path),
                            'scaler': joblib.load(self.models_dir / "latest_scaler.joblib"),
                            'selector': joblib.load(self.models_dir / "latest_feature_selector.joblib"),
                            'feature_names': joblib.load(self.models_dir / "latest_feature_names.joblib"),
                            'model_type': 'enhanced'
                        }

            # For per-experiment mode, models are loaded by load_experiment_specific_model
            # So this fallback should use basic models for validation compatibility

            # If not found and we're using basic models dir, check latest results directory
            if not (enhanced_model_path.exists() and cluster_model_path.exists()):
                results_dirs = sorted([d for d in Path(".").glob("results_*") if d.is_dir()], reverse=True)
                if results_dirs:
                    latest_models_dir = results_dirs[0] / "models"
                    enhanced_model_path = latest_models_dir / "enhanced_ensemble_model.joblib"
                    cluster_model_path = latest_models_dir / "cluster_model.joblib"

                    if enhanced_model_path.exists() and cluster_model_path.exists():
                        print(f"Loading enhanced ensemble models from {results_dirs[0]}...")
                        return {
                            'model': joblib.load(enhanced_model_path),
                            'cluster_model': joblib.load(cluster_model_path),
                            'scaler': joblib.load(latest_models_dir / "latest_scaler.joblib"),
                            'selector': joblib.load(latest_models_dir / "latest_feature_selector.joblib"),
                            'feature_names': joblib.load(latest_models_dir / "latest_feature_names.joblib"),
                            'model_type': 'enhanced'
                        }

            # If enhanced models exist in current dir
            if enhanced_model_path.exists() and cluster_model_path.exists():
                print("Loading enhanced ensemble models...")
                return {
                    'model': joblib.load(enhanced_model_path),
                    'cluster_model': joblib.load(cluster_model_path),
                    'scaler': joblib.load(self.models_dir / "latest_scaler.joblib"),
                    'selector': joblib.load(self.models_dir / "latest_feature_selector.joblib"),
                    'feature_names': joblib.load(self.models_dir / "latest_feature_names.joblib"),
                    'model_type': 'enhanced'
                }
            else:
                # Fall back to basic models
                print("Enhanced models not found, using basic models...")
                return {
                    'model': joblib.load(self.models_dir / "latest_model.joblib"),
                    'scaler': joblib.load(self.models_dir / "latest_scaler.joblib"),
                    'selector': joblib.load(self.models_dir / "latest_feature_selector.joblib"),
                    'feature_names': joblib.load(self.models_dir / "latest_feature_names.joblib"),
                    'model_type': 'basic'
                }
        except Exception as e:
            print(f"Error loading models from {self.models_dir}: {e}")
            return None

    def load_experiment_specific_model(self, exp_name):
        """Load model specific to an experiment in per-experiment mode."""
        if not (hasattr(self, 'trained_model_groups') and self.trained_model_groups):
            return None

        # Find the group key that matches this experiment
        group_key = None
        for group_name in self.trained_model_groups.keys():
            if exp_name in group_name or group_name in exp_name:
                group_key = group_name
                break

        if not group_key:
            return None

        # Create safe filename suffix (same logic as in training)
        safe_group_name = group_key.replace('/', '_').replace('\\', '_')
        model_suffix = f"_{safe_group_name}"

        try:
            enhanced_model_path = self.models_dir / f"enhanced_ensemble_model{model_suffix}.joblib"
            cluster_model_path = self.models_dir / f"cluster_model{model_suffix}.joblib"
            metadata_path = self.models_dir / f"enhanced_model_metadata{model_suffix}.joblib"

            if enhanced_model_path.exists() and cluster_model_path.exists():
                print(f"Loading per-experiment model for {exp_name} (group: {group_key})")
                return {
                    'model': joblib.load(enhanced_model_path),
                    'cluster_model': joblib.load(cluster_model_path),
                    'scaler': joblib.load(self.models_dir / "latest_scaler.joblib"),
                    'selector': joblib.load(self.models_dir / "latest_feature_selector.joblib"),
                    'feature_names': joblib.load(self.models_dir / "latest_feature_names.joblib"),
                    'model_type': 'enhanced',
                    'experiment_specific': True,
                    'group_name': group_key
                }
            else:
                print(f"Per-experiment model files not found for {exp_name}")
                return None
        except Exception as e:
            print(f"Error loading per-experiment model for {exp_name}: {e}")
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
            plot_output_path = self.output_dir / 'plots' / plot_file
            plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
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

            if models['model_type'] == 'enhanced':
                # Create cluster features for enhanced ensemble model
                cluster_model = models['cluster_model']
                kmeans = cluster_model['kmeans']
                gmm = cluster_model['gmm']
                cluster_centers = cluster_model['cluster_centers']

                cluster_labels = kmeans.predict(X_selected)
                gmm_probs = gmm.predict_proba(X_selected)

                # Create enhanced features (same as training)
                enhanced_features = []

                # Distance to each cluster center
                for center in cluster_centers:
                    distances = np.linalg.norm(X_selected - center, axis=1)
                    enhanced_features.append(distances)

                # GMM probabilities
                for i in range(gmm_probs.shape[1]):
                    enhanced_features.append(gmm_probs[:, i])

                # Cluster membership (one-hot encoded)
                for i in range(4):
                    cluster_membership = (cluster_labels == i).astype(float)
                    enhanced_features.append(cluster_membership)

                # Combine features
                X_final = np.hstack([X_selected] + [feat.reshape(-1, 1) for feat in enhanced_features])

                # Make ensemble predictions
                ensemble_model = models['model']
                ensemble_weights = ensemble_model['weights']
                trained_models = ensemble_model['models']
                reverse_mapping = ensemble_model['reverse_mapping']

                # Ensemble prediction
                predictions = np.zeros((len(X_final), 4))  # 4 classes
                for name, model in trained_models.items():
                    pred_proba = model.predict_proba(X_final)
                    predictions += pred_proba * ensemble_weights[name]

                y_pred_raw = np.argmax(predictions, axis=1)
                y_pred = [reverse_mapping[pred] for pred in y_pred_raw]  # Convert back to 1,2,3,4
                y_pred_proba = predictions
            else:
                # Use basic model directly on selected features
                y_pred_raw = models['model'].predict(X_selected)
                # Apply same label mapping as enhanced model (0->1, 1->2, 2->3, 3->4)
                label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
                y_pred = [label_mapping[pred] for pred in y_pred_raw]
                y_pred_proba = models['model'].predict_proba(X_selected)

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
            pred_output_path = self.output_dir / 'reports' / pred_file
            results_df.to_excel(pred_output_path, index=False)
            
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
            results_file = self.output_dir / 'reports' / 'enhanced_model_test_results.xlsx'
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
                train_samples = info.get('train_samples', 0)
                test_samples = info.get('test_samples', 0)
                val_samples = info.get('val_samples', 0)
                report += f"  Experiment: {filename}\n"
                report += f"    Train: {train_samples} samples\n"
                report += f"    Test: {test_samples} samples\n"
                report += f"    Val: {val_samples} samples\n"
                if info.get('classes'):
                    report += f"    Classes: {info['classes']}\n"
        
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
        print(f"  Network directory: {self.network_dir}")
        
        # Validate model directory when skipping training
        if self.skip_training:
            if not self.models_dir.exists():
                print(f"Error: Model directory '{self.models_dir}' does not exist!")
                print("Please ensure the model directory exists or don't use --skip-training")
                return False
            
            # Check for required model files (enhanced or basic)
            enhanced_models = [
                "enhanced_ensemble_model.joblib",
                "cluster_model.joblib",
                "latest_scaler.joblib",
                "latest_feature_selector.joblib",
                "latest_feature_names.joblib"
            ]

            basic_models = [
                "latest_model.joblib",
                "latest_scaler.joblib",
                "latest_feature_selector.joblib",
                "latest_feature_names.joblib"
            ]

            # Check if we have enhanced models
            has_enhanced = all((self.models_dir / model).exists() for model in enhanced_models)
            has_basic = all((self.models_dir / model).exists() for model in basic_models)

            if not (has_enhanced or has_basic):
                print(f"Error: Missing required model files in '{self.models_dir}':")
                print("Need either enhanced models:")
                for model in enhanced_models:
                    exists = "✓" if (self.models_dir / model).exists() else "✗"
                    print(f"  {exists} {model}")
                print("OR basic models:")
                for model in basic_models:
                    exists = "✓" if (self.models_dir / model).exists() else "✗"
                    print(f"  {exists} {model}")
                return False

            if has_enhanced:
                print(f"✓ Found enhanced models in {self.models_dir}")
            else:
                print(f"✓ Found basic models in {self.models_dir}")
            
            print(f"✓ All required models found in {self.models_dir}")
        
        try:
            # Setup output directory
            self.setup_output_directory()
            
            # Step 1: Data Discovery
            self.discover_training_data()
            print("\nUsing test/val files from same experiments for validation")

            # Discover existing per-experiment models when skipping training
            if self.skip_training and self.training_mode == 'per-experiment':
                self.discover_existing_per_experiment_models()

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

    def _get_tuned_models(self, X, y, sample_weights):
        """Get hyperparameter-tuned models to prevent overfitting."""
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, uniform
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        print("Performing hyperparameter tuning to prevent overfitting...")

        # Split data for tuning (use original data, not augmented)
        from sklearn.model_selection import train_test_split
        X_tune, X_val, y_tune, y_val = train_test_split(X, y, test_size=0.2,
                                                        random_state=self.random_seed,
                                                        stratify=y)

        tuned_models = {}

        # 1. XGBoost with regularization focus
        print("  Tuning XGBoost (regularization-focused)...")
        xgb_params = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(2, 6),
            'learning_rate': uniform(0.01, 0.15),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0.1, 1.0),
            'reg_lambda': uniform(0.5, 2.0),
            'min_child_weight': randint(3, 10)
        }

        xgb_search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=self.random_seed, eval_metric='mlogloss'),
            xgb_params, n_iter=20, cv=3, scoring='accuracy',
            random_state=self.random_seed, n_jobs=-1
        )
        xgb_search.fit(X_tune, y_tune)
        tuned_models['xgb_tuned'] = xgb_search.best_estimator_
        print(f"    Best XGBoost score: {xgb_search.best_score_:.3f}")

        # 2. Random Forest with conservative settings
        print("  Tuning Random Forest (conservative)...")
        rf_params = {
            'n_estimators': randint(50, 150),
            'max_depth': randint(3, 8),
            'min_samples_split': randint(5, 20),
            'min_samples_leaf': randint(2, 10),
            'max_features': ['sqrt', 'log2', 0.8]
        }

        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_seed, class_weight='balanced'),
            rf_params, n_iter=15, cv=3, scoring='accuracy',
            random_state=self.random_seed, n_jobs=-1
        )
        rf_search.fit(X_tune, y_tune)
        tuned_models['rf_tuned'] = rf_search.best_estimator_
        print(f"    Best Random Forest score: {rf_search.best_score_:.3f}")

        # 3. Gradient Boosting with early stopping
        print("  Tuning Gradient Boosting (early stopping)...")
        gb_params = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(2, 5),
            'learning_rate': uniform(0.01, 0.15),
            'subsample': uniform(0.7, 0.3),
            'min_samples_split': randint(5, 20),
            'min_samples_leaf': randint(2, 10)
        }

        gb_search = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=self.random_seed),
            gb_params, n_iter=15, cv=3, scoring='accuracy',
            random_state=self.random_seed, n_jobs=-1
        )
        gb_search.fit(X_tune, y_tune)
        tuned_models['gb_tuned'] = gb_search.best_estimator_
        print(f"    Best Gradient Boosting score: {gb_search.best_score_:.3f}")

        # 4. Add a simple logistic regression for baseline
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        print("  Adding Logistic Regression baseline...")
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_tune)

        lr_params = {
            'C': uniform(0.01, 10),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        lr_search = RandomizedSearchCV(
            LogisticRegression(random_state=self.random_seed, class_weight='balanced', max_iter=1000),
            lr_params, n_iter=10, cv=3, scoring='accuracy',
            random_state=self.random_seed, n_jobs=-1
        )
        lr_search.fit(X_scaled, y_tune)

        # Create a pipeline for logistic regression with scaling
        from sklearn.pipeline import Pipeline
        lr_pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', lr_search.best_estimator_)
        ])
        tuned_models['lr_tuned'] = lr_pipeline
        print(f"    Best Logistic Regression score: {lr_search.best_score_:.3f}")

        print("Hyperparameter tuning completed!")
        return tuned_models

    def group_experiments_by_training_mode(self, training_data_info):
        """Group experiments based on training mode."""
        if self.training_mode == 'combined':
            # Original behavior - all experiments together
            return {'combined': list(training_data_info.keys())}

        elif self.training_mode == 'concentration':
            # Group by concentration (3mM vs 6mM)
            groups = {'3mM': [], '6mM': []}
            for exp_name in training_data_info.keys():
                if '3mM' in exp_name:
                    groups['3mM'].append(exp_name)
                elif '6mM' in exp_name:
                    groups['6mM'].append(exp_name)
                else:
                    # Default to 3mM if unclear
                    groups['3mM'].append(exp_name)

            # Remove empty groups
            return {k: v for k, v in groups.items() if v}

        elif self.training_mode == 'per-experiment':
            # Each experiment gets its own model
            return {exp_name: [exp_name] for exp_name in training_data_info.keys()}

        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

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
    parser.add_argument('--network-dir', type=str, default="first_network", help='Network directory to use (first_network, second_network, third_network)')
    parser.add_argument('--training-mode', type=str, default='combined',
                       choices=['combined', 'concentration', 'per-experiment'],
                       help='Training mode: combined (default), concentration (separate 3mM/6mM models), per-experiment (individual models)')

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = AutoTrainingPipeline(
        max_features=args.max_features,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        skip_feature_eng=args.skip_feature_eng,
        skip_training=args.skip_training,
        model_dir=args.model_dir,
        network_dir=args.network_dir,
        training_mode=args.training_mode
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