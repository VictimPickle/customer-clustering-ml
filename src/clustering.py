#!/usr/bin/env python3
"""
Customer Clustering Analysis

This script implements and compares three clustering algorithms:
- K-Means
- Hierarchical Clustering
- DBSCAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(filepath):
    """Load customer data from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {len(df)} customers, {len(df.columns)} features")
    print(f"Columns: {list(df.columns)}")
    return df


def explore_data(df):
    """Perform exploratory data analysis."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")


def prepare_features(df):
    """Extract and scale features for clustering."""
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler


def kmeans_clustering(X_scaled, df, scaler):
    """Apply K-Means clustering with k=5."""
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING")
    print("="*60)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['KMeans'] = kmeans.fit_predict(X_scaled)
    
    # Get centroids in original scale
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    print(f"\nCluster sizes:")
    print(df['KMeans'].value_counts().sort_index())
    print(f"\nInertia (WCSS): {kmeans.inertia_:.2f}")
    print(f"\nCentroids (Original Scale):")
    print(f"{'Income (k$)':<15} {'Spending Score':<15}")
    for i, centroid in enumerate(centroids):
        print(f"{centroid[0]:<15.2f} {centroid[1]:<15.2f}")
    
    return kmeans, centroids


def hierarchical_clustering(X_scaled, df):
    """Apply Hierarchical clustering with Ward linkage."""
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING")
    print("="*60)
    
    linkage_matrix = linkage(X_scaled, method='ward')
    df['Hierarchical'] = fcluster(linkage_matrix, t=5, criterion='distance')
    
    print(f"\nCluster sizes:")
    print(df['Hierarchical'].value_counts().sort_index())
    print(f"\nLinkage matrix shape: {linkage_matrix.shape}")
    
    return linkage_matrix


def dbscan_clustering(X_scaled, df):
    """Apply DBSCAN clustering with eps=0.20."""
    print("\n" + "="*60)
    print("DBSCAN CLUSTERING")
    print("="*60)
    
    dbscan = DBSCAN(eps=0.20, min_samples=4)
    df['DBSCAN'] = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(df['DBSCAN'])) - (1 if -1 in df['DBSCAN'] else 0)
    n_noise = list(df['DBSCAN']).count(-1)
    
    print(f"\nNumber of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(df)*100:.1f}%)")
    print(f"\nCluster sizes (including noise):")
    print(df['DBSCAN'].value_counts().sort_index())
    
    return dbscan


def find_optimal_eps(X_scaled):
    """Analyze K-distance graph to find optimal eps."""
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances[:, -1], axis=0)
    
    print(f"\nEps value analysis:")
    print(f"Min distance: {distances.min():.3f}")
    print(f"Max distance: {distances.max():.3f}")
    print(f"Mean distance: {distances.mean():.3f}")
    print(f"\nRecommended eps range: 0.15 - 0.30")


def create_visualizations(df, X, kmeans, centroids, linkage_matrix):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # K-Means
    scatter1 = axes[0].scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
                               c=df['KMeans'], cmap='viridis', s=50, alpha=0.7)
    axes[0].scatter(centroids[:, 0], centroids[:, 1],
                    c='red', marker='X', s=300, edgecolors='black', linewidth=2)
    axes[0].set_title('K-Means (k=5)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Annual Income (k$)')
    axes[0].set_ylabel('Spending Score (1-100)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Hierarchical
    scatter2 = axes[1].scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
                               c=df['Hierarchical'], cmap='viridis', s=50, alpha=0.7)
    axes[1].set_title('Hierarchical (k=5)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Annual Income (k$)')
    axes[1].set_ylabel('Spending Score (1-100)')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    # DBSCAN
    scatter3 = axes[2].scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
                               c=df['DBSCAN'], cmap='viridis', s=50, alpha=0.7)
    noise_mask = df['DBSCAN'] == -1
    axes[2].scatter(X[noise_mask]['Annual Income (k$)'],
                    X[noise_mask]['Spending Score (1-100)'],
                    c='red', marker='X', s=100, edgecolors='black', linewidth=1, label='Outliers')
    axes[2].set_title('DBSCAN (eps=0.20)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Annual Income (k$)')
    axes[2].set_ylabel('Spending Score (1-100)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    plt.colorbar(scatter3, ax=axes[2], label='Cluster')
    
    plt.tight_layout()
    plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot: results/algorithm_comparison.png")
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("CUSTOMER CLUSTERING ANALYSIS")
    print("="*60)
    
    # Load and explore data
    df = load_data('data/customer_data.csv')
    explore_data(df)
    
    # Prepare features
    X, X_scaled, scaler = prepare_features(df)
    
    # Apply clustering algorithms
    kmeans, centroids = kmeans_clustering(X_scaled, df, scaler)
    hierarchical_matrix = hierarchical_clustering(X_scaled, df)
    dbscan_model = dbscan_clustering(X_scaled, df)
    
    # Find optimal eps
    find_optimal_eps(X_scaled)
    
    # Create visualizations
    create_visualizations(df, X, kmeans, centroids, hierarchical_matrix)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: results/")
    print(f"\nSummary:")
    print(f"  K-Means:       5 clusters, 0 outliers")
    print(f"  Hierarchical:  5 clusters, 0 outliers")
    print(f"  DBSCAN:        {len(set(df['DBSCAN'])) - 1} clusters, {(df['DBSCAN'] == -1).sum()} outliers")


if __name__ == '__main__':
    main()
