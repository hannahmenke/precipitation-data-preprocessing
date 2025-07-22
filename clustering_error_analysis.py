#!/usr/bin/env python3
"""
Clustering-based Error Analysis for XGBoost Predictions.

This script uses clustering techniques to identify specific combinations of features
that lead to prediction errors, helping to understand failure modes and patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ClusteringErrorAnalyzer:
    """
    Class to perform clustering-based error analysis.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the clustering error analyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.clustering_results = {}
        self.dimensionality_reduction_results = {}
        
    def load_prediction_data(self):
        """
        Load model components and make predictions on validation data.
        
        Returns:
            Dictionary with features, predictions, and error information
        """
        print("📂 Loading model and validation data...")
        
        # Load model components
        models_dir = Path("models")
        model = joblib.load(models_dir / "latest_model.joblib")
        scaler = joblib.load(models_dir / "latest_scaler.joblib")
        feature_selector = joblib.load(models_dir / "latest_feature_selector.joblib")
        feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
        
        # Load validation data
        val_improved = pd.read_excel('validation_data_improved.xlsx')
        
        # Preprocess validation data
        exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
        original_features = [col for col in val_improved.columns 
                           if col not in exclude_cols and val_improved[col].dtype in ['int64', 'float64']]
        
        X_val_raw = val_improved[original_features].fillna(val_improved[original_features].median())
        X_val_scaled = scaler.transform(X_val_raw)
        X_val_selected = feature_selector.transform(X_val_scaled)
        y_val_true = val_improved['type']
        
        # Make predictions
        y_val_pred = model.predict(X_val_selected)
        y_val_proba = model.predict_proba(X_val_selected)
        confidence = np.max(y_val_proba, axis=1)
        
        # Convert predictions back to original labels
        label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
        y_val_pred_labels = [label_mapping[pred] for pred in y_val_pred]
        
        # Identify errors
        is_correct = (y_val_true == y_val_pred_labels)
        
        print(f"✓ Loaded data: {len(y_val_true)} samples")
        print(f"✓ Accuracy: {np.mean(is_correct):.1%}")
        print(f"✓ Errors: {np.sum(~is_correct)} samples")
        
        return {
            'X_features': X_val_selected,
            'X_raw': X_val_raw,
            'feature_names': feature_names,
            'y_true': y_val_true,
            'y_pred': y_val_pred_labels,
            'confidence': confidence,
            'is_correct': is_correct,
            'model': model
        }
    
    def perform_dimensionality_reduction(self, X, methods=['pca', 'tsne', 'umap']):
        """
        Perform dimensionality reduction for visualization.
        
        Args:
            X: Feature matrix
            methods: List of dimensionality reduction methods to use
            
        Returns:
            Dictionary with reduced dimensional representations
        """
        print("🔄 Performing dimensionality reduction...")
        
        results = {}
        
        if 'pca' in methods:
            print("  - PCA...")
            pca = PCA(n_components=2, random_state=self.random_state)
            results['pca'] = pca.fit_transform(X)
            results['pca_explained'] = pca.explained_variance_ratio_
        
        if 'tsne' in methods:
            print("  - t-SNE...")
            tsne = TSNE(n_components=2, random_state=self.random_state, 
                       perplexity=min(30, len(X)//4))
            results['tsne'] = tsne.fit_transform(X)
        
        if 'umap' in methods:
            print("  - UMAP...")
            try:
                import umap
                umap_reducer = umap.UMAP(n_components=2, random_state=self.random_state)
                results['umap'] = umap_reducer.fit_transform(X)
            except ImportError:
                print("    ⚠️  UMAP not available, skipping...")
        
        self.dimensionality_reduction_results = results
        return results
    
    def perform_clustering(self, X, methods=['kmeans', 'dbscan', 'gaussian_mixture', 'hierarchical']):
        """
        Perform multiple clustering algorithms.
        
        Args:
            X: Feature matrix
            methods: List of clustering methods to use
            
        Returns:
            Dictionary with clustering results
        """
        print("🎯 Performing clustering analysis...")
        
        results = {}
        n_samples = len(X)
        
        if 'kmeans' in methods:
            print("  - K-Means clustering...")
            # Try different numbers of clusters
            silhouette_scores = []
            k_range = range(2, min(11, n_samples//2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(score)
            
            # Choose optimal k
            optimal_k = k_range[np.argmax(silhouette_scores)]
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            kmeans_labels = kmeans_final.fit_predict(X)
            
            results['kmeans'] = {
                'labels': kmeans_labels,
                'n_clusters': optimal_k,
                'silhouette_score': max(silhouette_scores),
                'cluster_centers': kmeans_final.cluster_centers_,
                'model': kmeans_final
            }
        
        if 'dbscan' in methods:
            print("  - DBSCAN clustering...")
            # Try different epsilon values
            eps_values = np.linspace(0.1, 2.0, 20)
            best_eps = 0.5
            best_score = -1
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=max(2, n_samples//20))
                labels = dbscan.fit_predict(X)
                
                if len(set(labels)) > 1 and -1 not in labels:  # Valid clustering
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
            
            # Final DBSCAN with best parameters
            dbscan_final = DBSCAN(eps=best_eps, min_samples=max(2, n_samples//20))
            dbscan_labels = dbscan_final.fit_predict(X)
            
            results['dbscan'] = {
                'labels': dbscan_labels,
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'eps': best_eps,
                'n_noise': list(dbscan_labels).count(-1),
                'model': dbscan_final
            }
        
        if 'gaussian_mixture' in methods:
            print("  - Gaussian Mixture clustering...")
            # Try different numbers of components
            aic_scores = []
            bic_scores = []
            k_range = range(2, min(11, n_samples//2))
            
            for k in k_range:
                gmm = GaussianMixture(n_components=k, random_state=self.random_state)
                gmm.fit(X)
                aic_scores.append(gmm.aic(X))
                bic_scores.append(gmm.bic(X))
            
            # Choose optimal k based on BIC
            optimal_k = k_range[np.argmin(bic_scores)]
            gmm_final = GaussianMixture(n_components=optimal_k, random_state=self.random_state)
            gmm_final.fit(X)
            gmm_labels = gmm_final.predict(X)
            
            results['gaussian_mixture'] = {
                'labels': gmm_labels,
                'n_clusters': optimal_k,
                'bic_score': min(bic_scores),
                'aic_score': aic_scores[np.argmin(bic_scores)],
                'model': gmm_final
            }
        
        if 'hierarchical' in methods:
            print("  - Hierarchical clustering...")
            # Use Ward linkage
            linkage_matrix = linkage(X, method='ward')
            
            # Try different numbers of clusters
            silhouette_scores = []
            k_range = range(2, min(11, n_samples//2))
            
            for k in k_range:
                hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
                cluster_labels = hierarchical.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(score)
            
            # Choose optimal k
            optimal_k = k_range[np.argmax(silhouette_scores)]
            hierarchical_final = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
            hierarchical_labels = hierarchical_final.fit_predict(X)
            
            results['hierarchical'] = {
                'labels': hierarchical_labels,
                'n_clusters': optimal_k,
                'silhouette_score': max(silhouette_scores),
                'linkage_matrix': linkage_matrix,
                'model': hierarchical_final
            }
        
        self.clustering_results = results
        return results
    
    def analyze_error_clusters(self, data_dict, clustering_results):
        """
        Analyze how errors are distributed across clusters.
        
        Args:
            data_dict: Dictionary with prediction data
            clustering_results: Results from clustering analysis
            
        Returns:
            Dictionary with error analysis for each clustering method
        """
        print("📊 Analyzing error patterns in clusters...")
        
        is_correct = data_dict['is_correct']
        y_true = data_dict['y_true']
        y_pred = data_dict['y_pred']
        confidence = data_dict['confidence']
        
        analysis_results = {}
        
        for method_name, clustering_result in clustering_results.items():
            print(f"  - Analyzing {method_name}...")
            
            cluster_labels = clustering_result['labels']
            unique_clusters = set(cluster_labels)
            if -1 in unique_clusters:  # Remove noise cluster for DBSCAN
                unique_clusters.remove(-1)
            
            cluster_analysis = {}
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_correct = is_correct[cluster_mask]
                cluster_true = y_true[cluster_mask]
                cluster_pred = [y_pred[i] for i, mask in enumerate(cluster_mask) if mask]
                cluster_conf = confidence[cluster_mask]
                
                # Calculate cluster statistics
                cluster_size = np.sum(cluster_mask)
                accuracy = np.mean(cluster_correct)
                error_rate = 1 - accuracy
                avg_confidence = np.mean(cluster_conf)
                
                # Class distribution
                true_class_dist = Counter(cluster_true)
                pred_class_dist = Counter(cluster_pred)
                
                # Most common errors
                errors_mask = ~cluster_correct
                if np.sum(errors_mask) > 0:
                    # Convert to numpy arrays for proper indexing
                    cluster_true_arr = np.array(cluster_true)
                    cluster_pred_arr = np.array(cluster_pred)
                    error_true = cluster_true_arr[errors_mask]
                    error_pred = cluster_pred_arr[errors_mask]
                    common_errors = Counter(zip(error_true, error_pred))
                else:
                    common_errors = Counter()
                
                cluster_analysis[cluster_id] = {
                    'size': cluster_size,
                    'accuracy': accuracy,
                    'error_rate': error_rate,
                    'avg_confidence': avg_confidence,
                    'true_class_distribution': dict(true_class_dist),
                    'pred_class_distribution': dict(pred_class_dist),
                    'common_errors': dict(common_errors),
                    'n_errors': np.sum(errors_mask)
                }
            
            # Overall method statistics
            method_stats = {
                'n_clusters': len(unique_clusters),
                'overall_silhouette': clustering_result.get('silhouette_score', None),
                'cluster_error_rates': [cluster_analysis[cid]['error_rate'] 
                                      for cid in unique_clusters],
                'high_error_clusters': [cid for cid in unique_clusters 
                                      if cluster_analysis[cid]['error_rate'] > 0.5],
                'clusters': cluster_analysis
            }
            
            analysis_results[method_name] = method_stats
        
        return analysis_results
    
    def identify_feature_patterns_in_error_clusters(self, data_dict, clustering_results, error_analysis):
        """
        Identify specific feature patterns in high-error clusters.
        
        Args:
            data_dict: Dictionary with prediction data
            clustering_results: Results from clustering
            error_analysis: Results from error analysis
            
        Returns:
            Dictionary with feature pattern analysis
        """
        print("🔍 Identifying feature patterns in error clusters...")
        
        X_features = data_dict['X_features']
        X_raw = data_dict['X_raw']
        feature_names = data_dict['feature_names']
        
        pattern_results = {}
        
        for method_name, method_analysis in error_analysis.items():
            print(f"  - Analyzing feature patterns for {method_name}...")
            
            cluster_labels = clustering_results[method_name]['labels']
            high_error_clusters = method_analysis['high_error_clusters']
            
            method_patterns = {}
            
            for cluster_id in high_error_clusters:
                cluster_mask = cluster_labels == cluster_id
                
                # Get cluster data
                cluster_features = X_features[cluster_mask]
                cluster_raw_features = X_raw.iloc[cluster_mask]
                
                # Compare with overall statistics
                overall_mean = np.mean(X_features, axis=0)
                overall_std = np.std(X_features, axis=0)
                
                cluster_mean = np.mean(cluster_features, axis=0)
                cluster_std = np.std(cluster_features, axis=0)
                
                # Calculate z-scores for cluster means
                z_scores = (cluster_mean - overall_mean) / (overall_std + 1e-8)
                
                # Identify distinguishing features (high absolute z-score)
                distinguishing_features = []
                for i, (feature_name, z_score) in enumerate(zip(feature_names, z_scores)):
                    if abs(z_score) > 1.5:  # Significant deviation
                        distinguishing_features.append({
                            'feature': feature_name,
                            'z_score': z_score,
                            'cluster_mean': cluster_mean[i],
                            'overall_mean': overall_mean[i],
                            'cluster_std': cluster_std[i],
                            'overall_std': overall_std[i]
                        })
                
                # Sort by absolute z-score
                distinguishing_features.sort(key=lambda x: abs(x['z_score']), reverse=True)
                
                method_patterns[cluster_id] = {
                    'cluster_size': np.sum(cluster_mask),
                    'error_rate': method_analysis['clusters'][cluster_id]['error_rate'],
                    'distinguishing_features': distinguishing_features[:10],  # Top 10
                    'cluster_center': cluster_mean,
                    'feature_ranges': {
                        feature_names[i]: {
                            'min': np.min(cluster_features[:, i]),
                            'max': np.max(cluster_features[:, i]),
                            'mean': cluster_mean[i],
                            'std': cluster_std[i]
                        } for i in range(len(feature_names))
                    }
                }
            
            pattern_results[method_name] = method_patterns
        
        return pattern_results
    
    def create_clustering_visualizations(self, data_dict, clustering_results, error_analysis, 
                                       dim_reduction_results):
        """
        Create comprehensive visualizations of clustering results.
        
        Args:
            data_dict: Dictionary with prediction data
            clustering_results: Results from clustering
            error_analysis: Results from error analysis
            dim_reduction_results: Results from dimensionality reduction
        """
        print("📈 Creating clustering visualizations...")
        
        is_correct = data_dict['is_correct']
        y_true = data_dict['y_true']
        confidence = data_dict['confidence']
        
        # Create a large figure with subplots
        n_methods = len(clustering_results)
        n_dim_reduction = len(dim_reduction_results)
        
        fig, axes = plt.subplots(n_methods, n_dim_reduction + 1, 
                                figsize=(6*(n_dim_reduction + 1), 6*n_methods))
        
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        if n_dim_reduction + 1 == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Clustering Analysis: Error Patterns in Feature Space', 
                    fontsize=16, fontweight='bold')
        
        for method_idx, (method_name, clustering_result) in enumerate(clustering_results.items()):
            cluster_labels = clustering_result['labels']
            
            # Plot for each dimensionality reduction method
            for dim_idx, (dim_method, dim_data) in enumerate(dim_reduction_results.items()):
                ax = axes[method_idx, dim_idx]
                
                # Create scatter plot colored by clusters
                unique_clusters = set(cluster_labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
                
                for cluster_id, color in zip(unique_clusters, colors):
                    cluster_mask = cluster_labels == cluster_id
                    
                    # Plot correct predictions
                    correct_mask = cluster_mask & is_correct
                    if np.sum(correct_mask) > 0:
                        correct_indices = np.where(correct_mask)[0]
                        ax.scatter(dim_data[correct_indices, 0], dim_data[correct_indices, 1], 
                                 c=[color], alpha=0.6, s=50, 
                                 label=f'Cluster {cluster_id} (Correct)', marker='o')
                    
                    # Plot errors with different marker
                    error_mask = cluster_mask & ~is_correct
                    if np.sum(error_mask) > 0:
                        error_indices = np.where(error_mask)[0]
                        ax.scatter(dim_data[error_indices, 0], dim_data[error_indices, 1], 
                                 c=[color], alpha=0.8, s=100, 
                                 label=f'Cluster {cluster_id} (Error)', marker='X')
                
                ax.set_title(f'{method_name.title()} Clusters\n({dim_method.upper()} projection)')
                ax.set_xlabel(f'{dim_method.upper()} 1')
                ax.set_ylabel(f'{dim_method.upper()} 2')
                ax.grid(alpha=0.3)
                
                # Add error rate annotations for high-error clusters
                high_error_clusters = error_analysis[method_name]['high_error_clusters']
                for cluster_id in high_error_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    if np.sum(cluster_mask) > 0:
                        cluster_indices = np.where(cluster_mask)[0]
                        cluster_center_x = np.mean(dim_data[cluster_indices, 0])
                        cluster_center_y = np.mean(dim_data[cluster_indices, 1])
                        error_rate = error_analysis[method_name]['clusters'][cluster_id]['error_rate']
                        ax.annotate(f'ER: {error_rate:.1%}', 
                                  (cluster_center_x, cluster_center_y),
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                                  fontsize=8, ha='center')
            
            # Summary plot (rightmost column)
            ax_summary = axes[method_idx, -1]
            
            # Bar plot of error rates by cluster
            cluster_ids = list(error_analysis[method_name]['clusters'].keys())
            error_rates = [error_analysis[method_name]['clusters'][cid]['error_rate'] 
                          for cid in cluster_ids]
            cluster_sizes = [error_analysis[method_name]['clusters'][cid]['size'] 
                           for cid in cluster_ids]
            
            # Color bars by error rate
            colors = ['red' if er > 0.5 else 'orange' if er > 0.3 else 'green' 
                     for er in error_rates]
            
            bars = ax_summary.bar(range(len(cluster_ids)), error_rates, color=colors, alpha=0.7)
            
            # Add size annotations
            for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
                height = bar.get_height()
                ax_summary.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'n={size}', ha='center', va='bottom', fontsize=8)
            
            ax_summary.set_title(f'{method_name.title()}\nError Rates by Cluster')
            ax_summary.set_xlabel('Cluster ID')
            ax_summary.set_ylabel('Error Rate')
            ax_summary.set_xticks(range(len(cluster_ids)))
            ax_summary.set_xticklabels(cluster_ids)
            ax_summary.grid(axis='y', alpha=0.3)
            ax_summary.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('clustering_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Created clustering visualization")
    
    def print_detailed_cluster_analysis(self, error_analysis, pattern_results):
        """
        Print detailed analysis of clustering results.
        
        Args:
            error_analysis: Results from error analysis
            pattern_results: Results from feature pattern analysis
        """
        print("\n" + "="*80)
        print("🎯 DETAILED CLUSTERING ERROR ANALYSIS")
        print("="*80)
        
        for method_name, method_analysis in error_analysis.items():
            print(f"\n🔍 {method_name.upper()} CLUSTERING ANALYSIS:")
            print("-" * 50)
            
            print(f"Number of clusters: {method_analysis['n_clusters']}")
            if method_analysis['overall_silhouette']:
                print(f"Silhouette score: {method_analysis['overall_silhouette']:.3f}")
            
            high_error_clusters = method_analysis['high_error_clusters']
            print(f"High-error clusters (>50% error): {len(high_error_clusters)}")
            
            if high_error_clusters:
                print(f"\n⚠️  HIGH-ERROR CLUSTERS:")
                for cluster_id in high_error_clusters:
                    cluster_info = method_analysis['clusters'][cluster_id]
                    print(f"\n   Cluster {cluster_id}:")
                    print(f"     Size: {cluster_info['size']} samples")
                    print(f"     Error rate: {cluster_info['error_rate']:.1%}")
                    print(f"     Avg confidence: {cluster_info['avg_confidence']:.3f}")
                    print(f"     Errors: {cluster_info['n_errors']}")
                    
                    # Show common errors
                    if cluster_info['common_errors']:
                        print(f"     Common errors:")
                        for (true_class, pred_class), count in cluster_info['common_errors'].items():
                            print(f"       {true_class} → {pred_class}: {count} times")
                    
                    # Show distinguishing features
                    if method_name in pattern_results and cluster_id in pattern_results[method_name]:
                        patterns = pattern_results[method_name][cluster_id]
                        if patterns['distinguishing_features']:
                            print(f"     Distinguishing features:")
                            for feat_info in patterns['distinguishing_features'][:5]:
                                direction = "higher" if feat_info['z_score'] > 0 else "lower"
                                print(f"       {feat_info['feature']:20}: {direction} ({feat_info['z_score']:+.2f}σ)")
            
            # Show all clusters summary
            print(f"\n📊 ALL CLUSTERS SUMMARY:")
            all_clusters = sorted(method_analysis['clusters'].keys())
            for cluster_id in all_clusters:
                cluster_info = method_analysis['clusters'][cluster_id]
                status = "⚠️ " if cluster_info['error_rate'] > 0.5 else "✓ "
                print(f"   {status}Cluster {cluster_id}: {cluster_info['size']:3d} samples, "
                      f"{cluster_info['error_rate']:5.1%} error rate")

def main():
    """
    Main function to perform clustering-based error analysis.
    """
    print("🎯 Clustering-based Error Analysis")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = ClusteringErrorAnalyzer()
        
        # Load prediction data
        data_dict = analyzer.load_prediction_data()
        
        # Perform dimensionality reduction
        dim_reduction_results = analyzer.perform_dimensionality_reduction(data_dict['X_features'])
        
        # Perform clustering
        clustering_results = analyzer.perform_clustering(data_dict['X_features'])
        
        # Analyze errors in clusters
        error_analysis = analyzer.analyze_error_clusters(data_dict, clustering_results)
        
        # Identify feature patterns in error clusters
        pattern_results = analyzer.identify_feature_patterns_in_error_clusters(
            data_dict, clustering_results, error_analysis)
        
        # Create visualizations
        analyzer.create_clustering_visualizations(
            data_dict, clustering_results, error_analysis, dim_reduction_results)
        
        # Print detailed analysis
        analyzer.print_detailed_cluster_analysis(error_analysis, pattern_results)
        
        # Save results
        results_summary = {
            'clustering_results': clustering_results,
            'error_analysis': error_analysis,
            'pattern_results': pattern_results
        }
        
        # Convert to DataFrame for saving (simplified version)
        cluster_summary_data = []
        for method_name, method_analysis in error_analysis.items():
            for cluster_id, cluster_info in method_analysis['clusters'].items():
                cluster_summary_data.append({
                    'method': method_name,
                    'cluster_id': cluster_id,
                    'size': cluster_info['size'],
                    'error_rate': cluster_info['error_rate'],
                    'avg_confidence': cluster_info['avg_confidence'],
                    'n_errors': cluster_info['n_errors'],
                    'is_high_error': cluster_info['error_rate'] > 0.5
                })
        
        cluster_summary_df = pd.DataFrame(cluster_summary_data)
        cluster_summary_df.to_excel('clustering_error_analysis_results.xlsx', index=False)
        
        print(f"\n✅ Clustering analysis complete!")
        print(f"📁 Saved results to: clustering_error_analysis_results.xlsx")
        print(f"📊 Generated visualization: clustering_error_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 