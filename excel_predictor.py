#!/usr/bin/env python3
"""
Excel Data Predictor using Trained XGBoost Model

This script loads a previously trained XGBoost model and makes predictions 
on new, unclassified Excel data. It handles all the preprocessing steps
automatically and outputs predictions with confidence scores.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_model_components(model_dir="models"):
    """
    Load all trained model components.
    
    Args:
        model_dir: Directory containing saved model components
        
    Returns:
        Tuple of (model, scaler, label_encoder, feature_names, metadata, feature_selector)
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please train a model first.")
    
    # Load latest model components
    try:
        model = joblib.load(model_path / "latest_model.joblib")
        scaler = joblib.load(model_path / "latest_scaler.joblib")
        label_encoder = joblib.load(model_path / "latest_label_encoder.joblib")
        feature_names = joblib.load(model_path / "latest_feature_names.joblib")
        metadata = joblib.load(model_path / "latest_metadata.joblib")
        feature_selector = joblib.load(model_path / "latest_feature_selector.joblib")
        
        print(f"✓ Loaded model components from: {model_path}")
        print(f"✓ Model trained on: {metadata['timestamp']}")
        print(f"✓ Features used: {len(feature_names)}")
        print(f"✓ Classes: {metadata['classes']}")
        
        return model, scaler, label_encoder, feature_names, metadata, feature_selector
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model components not found in '{model_dir}'. Please train a model first using excel_xgboost_classifier.py")

def preprocess_prediction_data(df, feature_names, scaler, feature_selector):
    """
    Preprocess new data for prediction using the same steps as training.
    
    Args:
        df: Input DataFrame with new data
        feature_names: List of selected feature names (12 features)
        scaler: Fitted StandardScaler from training (expects 20 features)
        feature_selector: Fitted feature selector to reduce 20→12 features
        
    Returns:
        Preprocessed feature matrix
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING NEW DATA")
    print(f"{'='*60}")
    
    print(f"Input data shape: {df.shape}")
    
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Apply same preprocessing as training script
    # Handle the "Aera" typo by creating a standardized "Area" column if needed
    if 'Aera' in df.columns and 'Area' not in df.columns:
        df['Area'] = df['Aera']
        print("ℹ️  Found 'Aera' column - standardized to 'Area' and excluded original")
    
    # Handle validation data column differences
    if 'NO.' in df.columns and 'ID' not in df.columns:
        df['ID'] = df['NO.']
        print("ℹ️  Mapped 'NO.' to 'ID' column")
    
    # Create missing columns if needed (using reasonable defaults/mappings)
    if 'Gray_ave' not in df.columns:
        if 'MeanIntensity' in df.columns:
            df['Gray_ave'] = df['MeanIntensity']
            print("ℹ️  Created 'Gray_ave' from 'MeanIntensity'")
        else:
            df['Gray_ave'] = 0  # fallback
            print("⚠️  Created 'Gray_ave' with default value 0")
    
    if 'overall_mean_gray' not in df.columns:
        if 'MeanIntensity' in df.columns:
            df['overall_mean_gray'] = df['MeanIntensity']
            print("ℹ️  Created 'overall_mean_gray' from 'MeanIntensity'")
        else:
            df['overall_mean_gray'] = 0  # fallback
            print("⚠️  Created 'overall_mean_gray' with default value 0")
    
    # Get all features that the scaler expects
    all_feature_names = scaler.feature_names_in_
    print(f"Scaler expects {len(all_feature_names)} features: {list(all_feature_names)}")
    print(f"After feature selection, model uses {len(feature_names)} features: {feature_names}")
    
    # Check if all required features for scaling are present
    missing_features = [feat for feat in all_feature_names if feat not in df.columns]
    if missing_features:
        print(f"Available columns: {sorted(df.columns.tolist())}")
        raise ValueError(f"Missing required features for scaling: {missing_features}")
    
    # Extract all features needed for scaling
    X_all = df[all_feature_names].copy()
    
    # Handle missing values for all features (same as training)
    print(f"Missing values per feature:")
    missing_counts = X_all.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} (filling with median)")
            X_all[col].fillna(X_all[col].median(), inplace=True)
        else:
            print(f"  {col}: 0")
    
    # Scale all features using the trained scaler
    X_scaled_all = scaler.transform(X_all)
    X_scaled_all = pd.DataFrame(X_scaled_all, columns=all_feature_names, index=X_all.index)
    
    # Apply feature selection to get the same features as training
    X_selected = feature_selector.transform(X_scaled_all)
    X_final = pd.DataFrame(X_selected, columns=feature_names, index=X_all.index)
    
    print(f"✓ Scaled {len(all_feature_names)} features → Selected {len(feature_names)} features → Final shape: {X_final.shape}")
    
    return X_final

def make_predictions(model, X, label_encoder):
    """
    Make predictions and return both class labels and probabilities.
    
    Args:
        model: Trained XGBoost model
        X: Preprocessed feature matrix
        label_encoder: Fitted LabelEncoder from training
        
    Returns:
        Tuple of (predictions, probabilities, confidence_scores)
    """
    print(f"\n{'='*60}")
    print("MAKING PREDICTIONS")
    print(f"{'='*60}")
    
    # Get predictions and probabilities
    y_pred_encoded = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Convert encoded predictions back to original labels
    predictions = label_encoder.inverse_transform(y_pred_encoded)
    
    # Calculate confidence scores (maximum probability)
    confidence_scores = np.max(y_pred_proba, axis=1)
    
    print(f"✓ Generated predictions for {len(predictions)} samples")
    
    return predictions, y_pred_proba, confidence_scores

def create_prediction_output(df, predictions, probabilities, confidence_scores, 
                           label_encoder, feature_names, output_file=None):
    """
    Create a comprehensive output DataFrame with predictions and metadata.
    
    Args:
        df: Original input DataFrame
        predictions: Predicted class labels
        probabilities: Prediction probabilities for each class
        confidence_scores: Maximum probability for each prediction
        label_encoder: LabelEncoder for class names
        feature_names: List of feature names used
        output_file: Optional file path to save results
        
    Returns:
        DataFrame with predictions and metadata
    """
    print(f"\n{'='*60}")
    print("CREATING PREDICTION OUTPUT")
    print(f"{'='*60}")
    
    # Create output DataFrame starting with original data
    output_df = df.copy()
    
    # Add predictions
    output_df['predicted_class'] = predictions
    output_df['confidence'] = confidence_scores
    
    # Add probability columns for each class
    class_names = label_encoder.classes_
    for i, class_name in enumerate(class_names):
        output_df[f'prob_class_{class_name}'] = probabilities[:, i]
    
    # Add prediction metadata
    output_df['prediction_rank'] = range(1, len(output_df) + 1)
    
    # Sort by confidence (most confident first)
    output_df = output_df.sort_values('confidence', ascending=False).reset_index(drop=True)
    output_df['confidence_rank'] = range(1, len(output_df) + 1)
    
    # Print summary
    print(f"✓ Created output with {len(output_df)} predictions")
    print(f"\nPrediction Summary:")
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    for class_label, count in prediction_counts.items():
        percentage = (count / len(predictions)) * 100
        print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {confidence_scores.mean():.3f}")
    print(f"  Min confidence: {confidence_scores.min():.3f}")
    print(f"  Max confidence: {confidence_scores.max():.3f}")
    
    # Count high-confidence predictions (>0.8)
    high_conf_count = np.sum(confidence_scores > 0.8)
    print(f"  High confidence (>0.8): {high_conf_count} samples ({(high_conf_count/len(confidence_scores)*100):.1f}%)")
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_df.to_excel(output_path, index=False)
        print(f"✓ Saved predictions to: {output_path}")
    
    return output_df

def main():
    """Main function to handle command-line arguments and coordinate prediction process."""
    parser = argparse.ArgumentParser(
        description="Make predictions on new Excel data using trained XGBoost model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python excel_predictor.py --data new_data.xlsx                    # Basic prediction
  python excel_predictor.py --data file.xlsx --output results.xlsx # Save results
  python excel_predictor.py --data file.xlsx --model-dir models2   # Custom model directory
  python excel_predictor.py --data file.xlsx --show-top 20         # Show top 20 predictions
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Excel file containing data to predict"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output Excel file path (default: predictions_TIMESTAMP.xlsx)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained model components (default: models)"
    )
    
    parser.add_argument(
        "--show-top",
        type=int,
        default=10,
        help="Number of top predictions to display (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        return
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"predictions_{timestamp}.xlsx"
    
    print(f"XGBoost Precipitation Particle Predictor")
    print(f"{'='*60}")
    print(f"Input data: {args.data}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output file: {args.output}")
    
    try:
        # Load model components
        model, scaler, label_encoder, feature_names, metadata, feature_selector = load_model_components(args.model_dir)
        
        # Load new data
        print(f"\nLoading data from: {data_path}")
        df = pd.read_excel(data_path)
        print(f"✓ Loaded {len(df)} samples")
        
        # Preprocess data
        X = preprocess_prediction_data(df, feature_names, scaler, feature_selector)
        
        # Make predictions
        predictions, probabilities, confidence_scores = make_predictions(model, X, label_encoder)
        
        # Create output
        output_df = create_prediction_output(
            df, predictions, probabilities, confidence_scores,
            label_encoder, feature_names, args.output
        )
        
        # Display top predictions
        print(f"\n{'='*60}")
        print(f"TOP {args.show_top} PREDICTIONS (by confidence)")
        print(f"{'='*60}")
        
        display_cols = ['predicted_class', 'confidence'] + [col for col in output_df.columns if col.startswith('prob_class_')]
        top_predictions = output_df[display_cols].head(args.show_top)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.3f}'.format)
        print(top_predictions.to_string(index=False))
        
        print(f"\n{'='*60}")
        print("PREDICTION COMPLETE!")
        print(f"{'='*60}")
        print(f"✓ Processed {len(df)} samples")
        print(f"✓ Results saved to: {args.output}")
        print(f"✓ Model features used: {feature_names}")
        print(f"✓ Mean prediction confidence: {confidence_scores.mean():.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 