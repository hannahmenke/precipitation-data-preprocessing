#!/usr/bin/env python3
"""
Excel Data XGBoost Classifier for Precipitation Data

This script imports tabular data from Excel files and uses XGBoost to predict 
the last column classification (particle type) from the other features.
It handles data preprocessing, model training, evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

def load_excel_data(file_paths):
    """
    Load and combine data from multiple Excel files.
    
    Args:
        file_paths: List of paths to Excel files
        
    Returns:
        Combined pandas DataFrame
    """
    dataframes = []
    
    for file_path in file_paths:
        print(f"Loading: {file_path}")
        df = pd.read_excel(file_path)
        
        # Add source file information
        df['source_file'] = Path(file_path).stem
        dataframes.append(df)
        
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    
    return combined_df

def preprocess_data(df, target_column='type'):
    """
    Preprocess the data for machine learning.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (X, y, feature_names, scaler)
    """
    print(f"\n{'='*60}")
    print("DATA PREPROCESSING")
    print(f"{'='*60}")
    
    # Identify feature columns (numeric columns excluding problematic features)
    exclude_cols = [
        # ID and target columns
        'NO.', target_column, 'source_file',
        # Spatial coordinates (don't generalize across images)
        'Centroid', 'BoundingBox', 'WeightedCentroid',
        # Redundant area measurements
        'FilledArea', 'FilledArea.1',
        # Pixel-level data (too detailed, not generalizable)
        'PixelldxList', 'PixList', 'PixelValues', 'Image',
        # Duplicate measurements
        'Gray_var.1'  # Duplicate of Gray_var
    ]
    
    # Handle the "Aera" typo by creating a standardized "Area" column if needed
    if 'Aera' in df.columns and 'Area' not in df.columns:
        df = df.copy()
        df['Area'] = df['Aera']
        exclude_cols.append('Aera')  # Exclude the typo version after copying
        print("ℹ️  Found 'Aera' column - standardized to 'Area' and excluded original")
    
    feature_columns = [col for col in df.columns 
                      if col not in exclude_cols 
                      and df[col].dtype in ['int64', 'float64']
                      and not col.startswith('Unnamed')]
    
    # Print categorized features for clarity
    shape_features = [col for col in feature_columns if any(x in col.lower() for x in ['major', 'minor', 'eccentric', 'circular', 'convex', 'extent', 'perimeter', 'area', 'ratio'])]
    intensity_features = [col for col in feature_columns if any(x in col.lower() for x in ['gray', 'mean', 'intensity'])]
    other_features = [col for col in feature_columns if col not in shape_features and col not in intensity_features]
    
    print(f"📐 Shape features ({len(shape_features)}): {shape_features}")
    print(f"🎨 Intensity features ({len(intensity_features)}): {intensity_features}")
    print(f"📊 Other features ({len(other_features)}): {other_features}")
    
    print(f"Feature columns identified: {feature_columns}")
    print(f"Target column: {target_column}")
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=[target_column]).copy()
    print(f"Rows after removing missing targets: {len(df_clean)}")
    
    # Extract features and target
    X = df_clean[feature_columns].copy()
    y = df_clean[target_column].copy()
    
    # Handle missing values in features
    print(f"Missing values per feature:")
    missing_counts = X.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count}")
            # Fill missing values with median
            X[col].fillna(X[col].median(), inplace=True)
    
    # Remove constant or near-constant features
    feature_variance = X.var()
    low_variance_features = feature_variance[feature_variance < 1e-10].index
    if len(low_variance_features) > 0:
        print(f"Removing low variance features: {list(low_variance_features)}")
        X = X.drop(columns=low_variance_features)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Encode target to ensure classes start from 0
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Encoded target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Print class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    print(f"\nClass distribution:")
    for class_val, count in zip(unique, counts):
        print(f"  Class {class_val}: {count} samples ({count/len(y_encoded)*100:.1f}%)")
    
    print(f"\nFinal feature matrix shape: {X_scaled.shape}")
    print(f"Final target vector shape: {y_encoded.shape}")
    
    return X_scaled, y_encoded, X.columns.tolist(), scaler, le

def balance_dataset(X, y, method='smote', random_state=42):
    """
    Balance the dataset using various resampling techniques.
    
    Args:
        X: Feature matrix
        y: Target vector
        method: Balancing method ('smote', 'undersampling', 'smoteenn', 'class_weights', 'none')
        random_state: Random seed
        
    Returns:
        Tuple of (X_balanced, y_balanced, class_weights_dict)
    """
    print(f"\n{'='*60}")
    print("HANDLING CLASS IMBALANCE")
    print(f"{'='*60}")
    
    # Print original class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Original class distribution:")
    for class_val, count in zip(unique, counts):
        print(f"  Class {class_val}: {count} samples ({count/len(y)*100:.1f}%)")
    
    if method == 'none':
        print("No rebalancing applied.")
        return X, y, None
    
    elif method == 'class_weights':
        print("Using class weights (no resampling)...")
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Calculated class weights: {class_weight_dict}")
        return X, y, class_weight_dict
    
    elif method == 'smote':
        print("Applying SMOTE oversampling...")
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
    elif method == 'undersampling':
        print("Applying random undersampling...")
        undersampler = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        
    elif method == 'smoteenn':
        print("Applying SMOTE + Edited Nearest Neighbours...")
        smoteenn = SMOTEENN(random_state=random_state)
        X_balanced, y_balanced = smoteenn.fit_resample(X, y)
        
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Print new class distribution
    unique_new, counts_new = np.unique(y_balanced, return_counts=True)
    print(f"\nBalanced class distribution:")
    for class_val, count in zip(unique_new, counts_new):
        print(f"  Class {class_val}: {count} samples ({count/len(y_balanced)*100:.1f}%)")
    
    print(f"Dataset size changed from {len(y)} to {len(y_balanced)} samples")
    
    return X_balanced, y_balanced, None

def select_features(X, y, feature_names, max_features=12, random_state=42):
    """
    Select the most important features to prevent overfitting.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        max_features: Maximum number of features to keep
        random_state: Random seed
        
    Returns:
        Tuple of (X_selected, selected_feature_names, selector)
    """
    print(f"\n{'='*60}")
    print("FEATURE SELECTION (Preventing Overfitting)")
    print(f"{'='*60}")
    
    print(f"Original features: {len(feature_names)}")
    print(f"Target features: {max_features}")
    
    if len(feature_names) <= max_features:
        print(f"No feature selection needed (≤ {max_features} features)")
        return X, feature_names, None
    
    # Use SelectKBest with f_classif (ANOVA F-test)
    selector = SelectKBest(score_func=f_classif, k=max_features)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Get feature scores
    feature_scores = selector.scores_
    feature_ranking = sorted(zip(feature_names, feature_scores), key=lambda x: x[1], reverse=True)
    
    print(f"✓ Selected {len(selected_features)} most important features:")
    for i, (feat, score) in enumerate(feature_ranking[:max_features]):
        print(f"  {i+1:2d}. {feat:<20} (score: {score:.2f})")
    
    print(f"\nFeatures excluded (lower importance):")
    for i, (feat, score) in enumerate(feature_ranking[max_features:], max_features+1):
        print(f"  {i:2d}. {feat:<20} (score: {score:.2f})")
    
    return X_selected, selected_features, selector

def train_xgboost_model(X, y, feature_names, test_size=0.2, random_state=42, balance_method='smote', 
                       feature_selection=True, max_features=12):
    """
    Train XGBoost classifier with hyperparameter tuning, class balancing, and feature selection.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        balance_method: Method for handling class imbalance
        feature_selection: Whether to perform feature selection
        max_features: Maximum number of features to keep
        
    Returns:
        Tuple of (best_model, X_train, X_test, y_train, y_test, selected_features, feature_selector)
    """
    print(f"\n{'='*60}")
    print("MODEL TRAINING (Anti-Overfitting)")
    print(f"{'='*60}")
    
    # Perform feature selection first (on full dataset for stability)
    if feature_selection:
        X_selected, selected_features, feature_selector = select_features(
            X, y, feature_names, max_features=max_features, random_state=random_state
        )
    else:
        X_selected, selected_features, feature_selector = X, feature_names, None
    
    # Split the data (before balancing to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nOriginal training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Balance the training data only (never balance test data!)
    X_train_balanced, y_train_balanced, class_weights = balance_dataset(
        X_train, y_train, method=balance_method, random_state=random_state
    )
    
    print(f"Balanced training set size: {X_train_balanced.shape[0]} samples")
    
    # Anti-overfitting hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],        # Fewer trees
        'max_depth': [2, 3, 4],                # Shallower trees
        'learning_rate': [0.01, 0.05, 0.1],    # Lower learning rates
        'subsample': [0.6, 0.8, 0.9],          # Subsampling for regularization
        'colsample_bytree': [0.6, 0.8, 1.0],   # Feature subsampling
        'reg_alpha': [0, 0.1, 1],              # L1 regularization
        'reg_lambda': [1, 2, 5],               # L2 regularization
        'min_child_weight': [1, 3, 5],         # Minimum samples per leaf
        'random_state': [random_state]
    }
    
    # Create XGBoost classifier without early stopping for GridSearchCV
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=random_state,
        n_jobs=-1,
        verbosity=0  # Reduce verbose output
    )
    
    # Add class weights if using class weighting method
    if balance_method == 'class_weight':
        # Calculate class weights
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(y_train_balanced), 
                                           y=y_train_balanced)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        # XGBoost doesn't directly support class_weight parameter in sklearn interface,
        # but we can calculate sample weights
        sample_weights = np.array([class_weights[class_] for class_ in y_train_balanced])
        print(f"Using sample weights for class balancing")
        
        # Perform grid search with sample weights
        print("Performing hyperparameter tuning on balanced data...")
        fit_params = {'sample_weight': sample_weights}
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=5, 
            scoring='balanced_accuracy',  # Better for imbalanced classes
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_balanced, y_train_balanced, **fit_params)
    else:
        # Perform grid search without sample weights
        print("Performing hyperparameter tuning on balanced data...")
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=5, 
            scoring='balanced_accuracy',  # Better for imbalanced classes
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_balanced, y_train_balanced)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    return best_model, X_train_balanced, X_test, y_train_balanced, y_test, selected_features, feature_selector

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate the trained model and generate performance metrics.
    
    Args:
        model: Trained XGBoost model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        feature_names: List of feature names
    """
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'feature_importance': feature_importance_df,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }

def create_visualizations(results, feature_importance_df, save_plots=False):
    """
    Create visualization plots for model results.
    
    Args:
        results: Dictionary containing evaluation results
        feature_importance_df: DataFrame with feature importance
        save_plots: Whether to save plots to files
    """
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('XGBoost Classification Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Class')
    axes[0, 0].set_ylabel('True Class')
    
    # 2. Feature Importance
    top_features = feature_importance_df.head(10)
    axes[0, 1].barh(range(len(top_features)), top_features['importance'])
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'])
    axes[0, 1].set_title('Top 10 Feature Importance')
    axes[0, 1].set_xlabel('Importance Score')
    
    # 3. Class Distribution
    unique, counts = np.unique(results['y_test'], return_counts=True)
    axes[1, 0].bar(unique, counts, alpha=0.7, color='skyblue')
    axes[1, 0].set_title('Test Set Class Distribution')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Count')
    
    # 4. Accuracy Comparison
    accuracies = [results['train_accuracy'], results['test_accuracy']]
    labels = ['Training', 'Test']
    bars = axes[1, 1].bar(labels, accuracies, alpha=0.7, color=['green', 'orange'])
    axes[1, 1].set_title('Training vs Test Accuracy')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('xgboost_classification_results.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization to 'xgboost_classification_results.png'")
    
    plt.show()

def save_model_components(model, scaler, label_encoder, feature_names, feature_selector=None, model_dir="models"):
    """
    Save the trained model and all necessary components for future predictions.
    
    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        feature_names: List of feature names used in training
        model_dir: Directory to save model components
    """
    print(f"\n{'='*60}")
    print("SAVING MODEL COMPONENTS")
    print(f"{'='*60}")
    
    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    # Generate timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model components
    model_file = model_path / f"xgboost_model_{timestamp}.joblib"
    scaler_file = model_path / f"scaler_{timestamp}.joblib"
    encoder_file = model_path / f"label_encoder_{timestamp}.joblib"
    features_file = model_path / f"feature_names_{timestamp}.joblib"
    
    # Save each component
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(label_encoder, encoder_file)
    joblib.dump(feature_names, features_file)
    
    # Also save latest versions (for easy loading)
    joblib.dump(model, model_path / "latest_model.joblib")
    joblib.dump(scaler, model_path / "latest_scaler.joblib")
    joblib.dump(label_encoder, model_path / "latest_label_encoder.joblib")
    joblib.dump(feature_names, model_path / "latest_feature_names.joblib")
    
    # Save feature selector if it exists
    if feature_selector is not None:
        selector_file = model_path / f"feature_selector_{timestamp}.joblib"
        joblib.dump(feature_selector, selector_file)
        joblib.dump(feature_selector, model_path / "latest_feature_selector.joblib")
        print(f"✓ Feature selector saved: {selector_file}")
    else:
        print("ℹ️  No feature selector to save (feature selection was disabled)")
    
    print(f"✓ Model saved: {model_file}")
    print(f"✓ Scaler saved: {scaler_file}")
    print(f"✓ Label encoder saved: {encoder_file}")
    print(f"✓ Feature names saved: {features_file}")
    print(f"✓ Latest versions saved in {model_path}/")
    
    # Create model metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'XGBoost',
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'classes': label_encoder.classes_.tolist(),
        'model_params': model.get_params()
    }
    
    metadata_file = model_path / f"model_metadata_{timestamp}.joblib"
    joblib.dump(metadata, metadata_file)
    joblib.dump(metadata, model_path / "latest_metadata.joblib")
    
    print(f"✓ Metadata saved: {metadata_file}")
    
    return {
        'model_file': model_file,
        'scaler_file': scaler_file,
        'encoder_file': encoder_file,
        'features_file': features_file,
        'metadata_file': metadata_file
    }

def main():
    """Main function to handle command-line arguments and coordinate the classification process."""
    parser = argparse.ArgumentParser(
        description="XGBoost classifier for Excel precipitation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python excel_xgboost_classifier.py                                    # Default: SMOTE + feature selection
  python excel_xgboost_classifier.py --files file1.xlsx file2.xlsx     # Specify custom files
  python excel_xgboost_classifier.py --target-column "class"           # Custom target column
  python excel_xgboost_classifier.py --test-size 0.3 --save-plots      # Custom test size and save plots
  python excel_xgboost_classifier.py --balance-method smote            # SMOTE oversampling (default)
  python excel_xgboost_classifier.py --balance-method class_weights    # Use class weights
  python excel_xgboost_classifier.py --max-features 8                  # Keep only 8 best features
  python excel_xgboost_classifier.py --no-feature-selection            # Use all features (may overfit)
        """
    )
    
    parser.add_argument(
        "--files", "-f",
        nargs="*",
        default=["training_data/6mM-10-label-train.xlsx", "training_data/3mM-4-label-train.xlsx"],
        help="Excel files to process (default: new training_data/*-train.xlsx files)"
    )
    
    parser.add_argument(
        "--target-column", "-t",
        type=str,
        default="type",
        help="Name of the target column for classification (default: 'type')"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save visualization plots to PNG files"
    )
    
    parser.add_argument(
        "--balance-method",
        type=str,
        default="smote",
        choices=["smote", "undersampling", "smoteenn", "class_weights", "none"],
        help="Method for handling class imbalance (default: smote)"
    )
    
    parser.add_argument(
        "--max-features",
        type=int,
        default=12,
        help="Maximum number of features to keep (prevents overfitting, default: 12)"
    )
    
    parser.add_argument(
        "--no-feature-selection",
        action="store_true",
        help="Disable automatic feature selection (may cause overfitting)"
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    for file_path in args.files:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            return
    
    print(f"XGBoost Classification for Excel Data")
    print(f"{'='*60}")
    print(f"Files: {args.files}")
    print(f"Target column: {args.target_column}")
    print(f"Test size: {args.test_size}")
    print(f"Random seed: {args.random_seed}")
    print(f"Balance method: {args.balance_method}")
    
    try:
        # Load data
        df = load_excel_data(args.files)
        
        # Preprocess data
        X, y, feature_names, scaler, label_encoder = preprocess_data(df, args.target_column)
        
        # Train model with class balancing and feature selection
        model, X_train, X_test, y_train, y_test, selected_features, feature_selector = train_xgboost_model(
            X, y, feature_names, test_size=args.test_size, random_state=args.random_seed, 
            balance_method=args.balance_method, feature_selection=not args.no_feature_selection, 
            max_features=args.max_features
        )
        
        # Evaluate model
        results = evaluate_model(model, X_train, X_test, y_train, y_test, selected_features)
        
        # Create visualizations
        create_visualizations(results, results['feature_importance'], args.save_plots)
        
        # Save model components
        saved_files = save_model_components(model, scaler, label_encoder, selected_features, feature_selector)
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION COMPLETE!")
        print(f"{'='*60}")
        print(f"✓ Final Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"✓ Model successfully trained on {len(X)} samples")
        print(f"✓ Used {len(selected_features)} selected features: {selected_features}")
        if feature_selector is not None:
            print(f"✓ Feature selection: Reduced from {len(feature_names)} to {len(selected_features)} features")
        print(f"✓ Class mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        print(f"✓ Model components saved to: models/")
        
        print(f"\n📋 To use this model for predictions:")
        print(f"  python excel_predictor.py --data new_data.xlsx")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 