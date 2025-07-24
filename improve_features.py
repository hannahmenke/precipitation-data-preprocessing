#!/usr/bin/env python3
"""
Feature engineering improvements based on error analysis.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

def create_improved_features(df):
    """Create improved features based on error analysis."""
    df_improved = df.copy()
    
    print("🔧 Creating improved features...")
    
    # 1. Shape Complexity Score (address Extent, Circularity, Perimeter issues)
    # Normalize each component before combining
    extent_norm = (df['Extent'] - df['Extent'].min()) / (df['Extent'].max() - df['Extent'].min())
    circularity_norm = (df['Circularity'] - df['Circularity'].min()) / (df['Circularity'].max() - df['Circularity'].min())
    
    df_improved['Shape_Complexity'] = extent_norm * (1 - circularity_norm)  # High extent + low circularity = complex
    
    # 2. Normalized Shape Ratios (better than raw ratios)
    df_improved['Area_Perimeter_Ratio'] = df['Area'] / (df['Perimeter'] + 1e-6)  # Compactness measure
    df_improved['Convex_Efficiency'] = df['Area'] / (df['ConvexArea'] + 1e-6)  # How much of convex hull is filled
    
    # 3. Robust Eccentricity (Class 1 issue)
    # Clip extreme values that might be measurement errors
    df_improved['Eccentricity_Robust'] = np.clip(df['Eccentricity'], 0, 1)
    
    # 4. Distance Feature Combinations (dis, dis_normal issues)
    df_improved['Distance_Ratio'] = df['dis'] / (df['dis_normal'] + 1e-6)
    df_improved['Distance_Interaction'] = df['dis'] * df['dis_normal']
    
    # 5. Intensity Stability Score
    # Combine Gray_var with other intensity features for stability
    gray_mean = df['MeanIntensity'] if 'MeanIntensity' in df.columns else df.get('Gray_ave', 0)
    df_improved['Intensity_Stability'] = gray_mean / (df['Gray_var'] + 1e-6)
    
    # 6. Comprehensive Shape Score
    # Combine problematic features into a single robust measure
    df_improved['Comprehensive_Shape'] = (
        df_improved['Major_Minor_ratio'] * 
        df_improved['Circularity'] * 
        (1 - df_improved['Eccentricity_Robust'])
    )
    
    # 7. Feature Interactions for Class 4 (most problematic)
    # Based on confusion analysis: Class 4 ↔ Class 1 issues
    df_improved['Class4_Discriminator'] = (
        df['Extent'] * df['Circularity'] * df['Perimeter']
    )
    
    print(f"✓ Added {len(df_improved.columns) - len(df.columns)} new features")
    
    return df_improved

def load_and_improve_training_data():
    """Load training data and apply feature improvements."""
    print("[LOAD] Loading training data...")
    
    # Load training files
    train_files = ["training_data/train3.xlsx", "training_data/train6.xlsx"]
    dataframes = []
    
    for file_path in train_files:
        df = pd.read_excel(file_path)
        dataframes.append(df)
    
    train_df = pd.concat(dataframes, ignore_index=True)
    print(f"✓ Loaded training data: {train_df.shape}")
    
    # Apply improvements
    train_improved = create_improved_features(train_df)
    
    return train_improved

def load_and_improve_test_data():
    """Load test data and apply feature improvements."""
    print("[LOAD] Loading test data...")
    
    # Load test files from data_for_classification folder
    val3 = pd.read_excel('data_for_classification/val3.xlsx')
    val6 = pd.read_excel('data_for_classification/val6.xlsx')
    test_df = pd.concat([val3, val6], ignore_index=True)
    
    print(f"✓ Loaded test data: {test_df.shape}")
    
    # Apply improvements
    test_improved = create_improved_features(test_df)
    
    return test_improved

def analyze_feature_improvements(train_df, test_df):
    """Analyze how the new features perform."""
    print("\n🔍 ANALYZING FEATURE IMPROVEMENTS")
    print("="*50)
    
    # Define all potential features (original + new)
    exclude_cols = [
        'NO.', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid',
        'FilledArea', 'FilledArea.1', 'PixelldxList', 'PixList', 'PixelValues', 
        'Image', 'Gray_var.1'
    ]
    
    feature_columns = [col for col in train_df.columns 
                      if col not in exclude_cols 
                      and train_df[col].dtype in ['int64', 'float64']
                      and not col.startswith('Unnamed')]
    
    print(f"✓ Total features available: {len(feature_columns)}")
    
    # Prepare data
    X_train = train_df[feature_columns].fillna(train_df[feature_columns].median())
    y_train = train_df['type']
    
    X_test = test_df[feature_columns].fillna(test_df[feature_columns].median())
    y_test = test_df['type']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    
    # Feature selection with more features
    print(f"\n🎯 Feature Selection Analysis")
    
    for n_features in [15, 20, 25]:
        if n_features <= len(feature_columns):
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(X_train_scaled, y_train)
            
            # Get selected features
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            feature_scores = selector.scores_
            
            print(f"\nTop {n_features} features:")
            feature_ranking = sorted(zip(feature_columns, feature_scores), key=lambda x: x[1], reverse=True)
            
            new_features = [name for name, score in feature_ranking[:n_features] 
                           if name not in ['Area', 'Major', 'Minor', 'Eccentricity', 'ConvexArea', 
                                         'Circularity', 'Extent', 'Perimeter', 'MeanIntensity', 
                                         'Gray_var', 'Gray_skew', 'Gray_skew_abs', 'Gray_kur', 
                                         'dis', 'dis_normal', 'Major_Minor_ratio', 'normalized_local_mean_gray']]
            
            print(f"  New features in top {n_features}: {new_features}")
            
            for i, (name, score) in enumerate(feature_ranking[:n_features]):
                status = "NEW" if name in new_features else "orig"
                print(f"  {i+1:2d}. {name:30} (score: {score:8.2f}) [{status}]")

def create_feature_importance_comparison():
    """Compare original vs improved feature importance."""
    print(f"\n[ANALYSIS] Creating feature importance comparison...")
    
    # Load original model for comparison
    models_dir = Path("models")
    if models_dir.exists():
        try:
            original_feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
            original_model = joblib.load(models_dir / "latest_model.joblib")
            
            print(f"✓ Original model uses {len(original_feature_names)} features")
            
            # Show original feature importance
            feature_importance = original_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': original_feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nOriginal Top 10 Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:30} {row['importance']:.4f}")
                
        except Exception as e:
            print(f"[WARNING] Could not load original model: {e}")

def suggest_model_improvements():
    """Suggest specific model training improvements."""
    print(f"\n[RECOMMENDATIONS] SUGGESTED IMPROVEMENTS")
    print("="*50)
    
    print(f"""
1. [TARGET] TARGETED FEATURE ENGINEERING:
   • Use new Shape_Complexity instead of raw Extent
   • Replace Circularity with Comprehensive_Shape  
   • Add Class4_Discriminator for Class 4 vs Class 1 separation
   
2. [RETRAIN] RETRAIN WITH MORE FEATURES:
   • Increase --max-features from 15 to 20-25
   • Include the new engineered features
   • Use feature selection to pick the best combination

3. [MODEL] MODEL ARCHITECTURE ADJUSTMENTS:
   • Class 4 specific: Use class weights to boost Class 4 performance
   • Try ensemble methods (multiple models for different class pairs)
   • Consider tree-based features interactions

4. 📈 TRAINING STRATEGY:
   • Stratified sampling to ensure Class 4 representation
   • Cross-validation focused on Class 4 performance
   • Early stopping based on Class 4 recall

5. 🎛️ HYPERPARAMETER FOCUS:
   • Lower learning rate for better Class 4 learning
   • Higher min_child_weight for Class 4 stability
   • Increase reg_alpha for Class 4 regularization
   
6. 📋 VALIDATION IMPROVEMENTS:
   • Check for domain shift between train/val data
   • Normalize intensity features across datasets
   • Consider separate models for 3mM vs 6mM data
""")

def main():
    """Main function to demonstrate feature improvements."""
    print("🔧 Feature Engineering for Improved Classification")
    print("=" * 60)
    
    try:
        # Load and improve data
        train_improved = load_and_improve_training_data()
        test_improved = load_and_improve_test_data()
        
        # Analyze improvements
        analyze_feature_improvements(train_improved, test_improved)
        
        # Compare with original
        create_feature_importance_comparison()
        
        # Provide recommendations
        suggest_model_improvements()
        
        # Save improved datasets
        train_improved.to_excel('training_data/train_improved.xlsx', index=False)
        test_improved.to_excel('test_data_improved.xlsx', index=False)
        
        print(f"\n[SUCCESS] Feature engineering complete!")
        print(f"✓ Saved improved training data: training_data/train_improved.xlsx")
        print(f"✓ Saved improved test data: test_data_improved.xlsx")
        
        print(f"\n[NEXT] NEXT STEPS:")
        print(f"1. Run: python excel_xgboost_classifier.py --files training_data/train_improved.xlsx --max-features 25")
        print(f"2. Test on validation data using the new model")
        print(f"3. Compare results with original model")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 