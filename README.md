# Customer Clustering - Machine Learning Project

A comprehensive customer segmentation project using multiple unsupervised clustering algorithms to divide customers into actionable business segments based on purchasing behavior.

## 📋 Project Overview

This project implements and compares three major clustering algorithms on a dataset of 200 mall customers. The goal is to identify natural customer segments based on **Annual Income** and **Spending Score** to enable targeted marketing strategies.

### Dataset Features
- **CustomerID**: Unique customer identifier (1-200)
- **Gender**: Male or Female
- **Age**: Customer age (18-70 years)
- **Annual Income (k$)**: Yearly income in thousands (15-137k)
- **Spending Score (1-100)**: Behavioral metric assigned by mall

## 🎯 Algorithms Implemented

### 1. K-Means Clustering
- **Optimal k**: 5 (determined via Elbow Method)
- **Approach**: Partitioning-based, distance to centroids
- **Strengths**: Fast, scalable, easy to interpret
- **Limitations**: Assumes spherical clusters, forces all points into clusters

**Result**: 5 distinct customer segments with clear centroid locations

### 2. Hierarchical Clustering (Agglomerative)
- **Linkage Method**: Ward (minimizes within-cluster variance)
- **Optimal k**: 5 (determined via dendrogram analysis)
- **Approach**: Builds a tree of nested clusters
- **Strengths**: Natural boundaries, dendrogram shows relationships, interpretable
- **Limitations**: More computationally expensive than K-Means

**Result**: Natural hierarchical groupings with dendrogram visualization

### 3. DBSCAN (Density-Based Spatial Clustering)
- **Optimal eps**: 0.20 (from K-distance graph analysis)
- **min_samples**: 4
- **Approach**: Density-based, finds arbitrary shapes
- **Strengths**: Identifies outliers, no need to specify k, finds realistic clusters
- **Limitations**: Sensitive to parameter tuning

**Result**: 4 main clusters + 73 outliers (36.5% of data marked as noise points)

## 📊 Key Findings

### Customer Segments Identified

| Segment | Characteristics | Business Insight |
|---------|-----------------|------------------|
| **Segment 1** | Low Income, High Spending | Loyal budget customers, credit-heavy |
| **Segment 2** | High Income, High Spending | Premium VIP customers, target for luxury |
| **Segment 3** | Moderate Income, Moderate Spending | Average mainstream customers |
| **Segment 4** | Low Income, Low Spending | Price-sensitive segment, discount focused |
| **Segment 5** | High Income, Low Spending | Conversion opportunity, investigate competitors |

### Algorithm Comparison

```
K-Means:           5 clusters, 0 outliers
Hierarchical:      5 clusters, 0 outliers
DBSCAN:            4 clusters, 73 outliers (36.5%)
```

**Observation**: K-Means and Hierarchical produce very similar segments. DBSCAN reveals significant outliers worth investigating separately.

## 🛠️ Technologies & Libraries

- **Python 3.8+**
- **pandas**: Data manipulation
- **numpy**: Numerical computation
- **scikit-learn**: Machine learning algorithms
  - KMeans
  - AgglomerativeClustering
  - DBSCAN
- **matplotlib & seaborn**: Visualization
- **scipy**: Hierarchical clustering dendrograms

## 📁 Project Structure

```
customer-clustering/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── customer_data.csv             # Raw dataset (200 records)
├── notebooks/
│   └── clustering_analysis.ipynb      # Full analysis notebook
├── src/
│   ├── clustering.py                  # Clustering utility functions
│   └── visualization.py               # Plotting functions
└── results/
    └── algorithm_comparison.png       # Side-by-side comparison plot
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/VictimPickle/customer-clustering.git
cd customer-clustering
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
jupyter notebook notebooks/clustering_analysis.ipynb
```

Or run the Python script:
```bash
python src/clustering.py
```

## 📈 Analysis Steps

1. **Exploratory Data Analysis (EDA)**
   - Load and inspect dataset
   - Check for missing values
   - Analyze distributions (Age, Income, Spending Score)
   - Identify relationships between features

2. **Feature Scaling**
   - Standardize features using StandardScaler
   - Essential for distance-based algorithms

3. **K-Means Clustering**
   - Implement K-Means with k=5
   - Plot elbow curve to validate k
   - Visualize customer segments

4. **Hierarchical Clustering**
   - Build dendrogram with Ward linkage
   - Cut at distance threshold for k=5
   - Compare with K-Means results

5. **DBSCAN**
   - Analyze K-distance graph
   - Select optimal eps parameter
   - Identify outliers and noise points

6. **Algorithm Comparison**
   - Side-by-side visualization
   - Quantitative metrics
   - Business interpretation

## 🔍 Key Insights

### Elbow Method (K-Means)
- WCSS drops significantly from k=1 to k=5
- Diminishing returns after k=5
- Clear elbow at k=5 confirms optimal cluster count

### Dendrogram Analysis (Hierarchical)
- Sharp vertical jumps indicate natural cluster boundaries
- Horizontal cut at distance ~5 creates 5 balanced clusters
- More natural groupings than rigid K-Means spheres

### Outlier Detection (DBSCAN)
- 73 customers (36.5%) identified as anomalies
- Clustered at extremes: high income/low spend OR low income/high spend
- Require separate business strategy

## 💡 Business Applications

1. **Targeted Marketing**: Tailor campaigns to each segment
2. **Risk Assessment**: Monitor high-income/low-spending customers
3. **Pricing Strategy**: Adjust prices by segment sensitivity
4. **Customer Retention**: Focus on high-value segments
5. **Anomaly Detection**: Flag outliers for manual review

## 📚 Learning Resources

- [Scikit-Learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [K-Means vs DBSCAN](https://www.kdnuggets.com/2020/09/dbscan-clustering-algorithm-machine-learning.html)
- [Elbow Method Explained](https://www.geeksforgeeks.org/elbow-method-for-optimal-k-in-kmeans-clustering/)

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Mobin Ghorbani** - [@VictimPickle](https://github.com/VictimPickle)

Computer Science Student at University of Tehran

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Add more clustering algorithms
- Enhance visualizations

---

**Last Updated**: December 2025
**Status**: Complete ✅
