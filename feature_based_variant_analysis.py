"""
Pattern-Based Feature Generation and Variant Clustering Module

Usage:
    from feature_based_variant_clustering import perform_feature_based_variant_clustering
    
    medoid_case_ids, results = perform_feature_based_variant_clustering(
        log=event_log,           # pandas DataFrame or pm4py EventLog
        case_id_col='stay_id',   # column name for case identifier
        activity_col='activity', # column name for activity
        timestamp_col='time'     # column name for timestamp
    )
    
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# CLINICAL THRESHOLDS (Configurable)

DEFAULT_VITAL_THRESHOLDS = {
    'fever': {'column': 'temperature', 'operator': '>=', 'value': 100.0},
    'tachycardia': {'column': 'heartrate', 'operator': '>', 'value': 100},
    'tachypnea': {'column': 'resprate', 'operator': '>', 'value': 20},
    'hypoxemia': {'column': 'o2sat', 'operator': '<', 'value': 90},
    'high_sbp': {'column': 'sbp', 'operator': '>', 'value': 120},
    'high_dbp': {'column': 'dbp', 'operator': '>', 'value': 80},
    'has_pain': {'column': 'pain', 'operator': '>', 'value': 0},
}

DEFAULT_CATEGORICAL_FEATURES = {
    'arrival_by_ambulance': {'column': 'arrival_transport', 'value': 'AMBULANCE'},
    'arrival_by_walk_in': {'column': 'arrival_transport', 'value': 'WALK IN'},
    'arrival_by_helicopter': {'column': 'arrival_transport', 'value': 'HELICOPTER'},
}

DEFAULT_PRESENCE_FEATURES = {
    'has_received_medicine': {'column': 'drug_name'},
}

DEFAULT_ACUITY_FEATURE = {
    'high_acuity': {'column': 'acuity', 'values': [1, 2, 3]},
}



# K-MEDOIDS IMPLEMENTATION

class KMedoids:
    """
    K-Medoids clustering using PAM (Partitioning Around Medoids) algorithm.
    
    Unlike K-Means, K-Medoids selects actual data points as cluster centers,
    making them directly interpretable and usable for downstream analysis.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    metric : str
        Distance metric to use. Default is 'hamming' for binary data.
    random_state : int
        Random seed for reproducibility.
    max_iter : int
        Maximum number of iterations.
    """
    
    def __init__(self, n_clusters: int = 3, metric: str = 'hamming', 
                 random_state: int = 42, max_iter: int = 300):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.medoid_indices_ = None
        self.labels_ = None
        self.inertia_ = None
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Compute pairwise distance matrix
        dist_matrix = cdist(X, X, metric=self.metric)
        
        # Initialize medoids randomly
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for iteration in range(self.max_iter):
            # Assign points to nearest medoid
            distances_to_medoids = dist_matrix[:, medoid_indices]
            labels = np.argmin(distances_to_medoids, axis=1)
            
            # Update medoids
            new_medoid_indices = np.zeros(self.n_clusters, dtype=int)
            for k in range(self.n_clusters):
                cluster_mask = labels == k
                if cluster_mask.sum() == 0:
                    new_medoid_indices[k] = medoid_indices[k]
                    continue
                cluster_indices = np.where(cluster_mask)[0]
                
                # Find the point that minimizes total distance within cluster
                cluster_distances = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
                total_distances = cluster_distances.sum(axis=1)
                best_idx = cluster_indices[np.argmin(total_distances)]
                new_medoid_indices[k] = best_idx
            
            # Check convergence
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
            medoid_indices = new_medoid_indices
        
        # Final assignment
        distances_to_medoids = dist_matrix[:, medoid_indices]
        labels = np.argmin(distances_to_medoids, axis=1)
        
        # Calculate inertia
        inertia = sum(dist_matrix[i, medoid_indices[labels[i]]] for i in range(n_samples))
        
        self.medoid_indices_ = medoid_indices
        self.labels_ = labels
        self.inertia_ = inertia
        
        return labels


# FEATURE GENERATION FUNCTIONS

def _apply_threshold(series: pd.Series, operator: str, value: float) -> pd.Series:
    """Apply a comparison operator to a pandas Series."""
    if operator == '>=':
        return series >= value
    elif operator == '>':
        return series > value
    elif operator == '<=':
        return series <= value
    elif operator == '<':
        return series < value
    elif operator == '==':
        return series == value
    else:
        raise ValueError(f"Unknown operator: {operator}")


def generate_binary_features(
    df: pd.DataFrame,
    case_id_col: str = 'stay_id',
    vital_thresholds: Dict = None,
    categorical_features: Dict = None,
    presence_features: Dict = None,
    acuity_feature: Dict = None,
    aggregation_method: str = 'any'
) -> pd.DataFrame:
    """
    Generate binary feature vectors for each case using the φ mapping function.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log as a pandas DataFrame.
    case_id_col : str
        Column name for case identifier.
    vital_thresholds : dict
        Dictionary defining vital sign thresholds.
    categorical_features : dict
        Dictionary defining categorical features.
    presence_features : dict
        Dictionary defining presence-based features.
    acuity_feature : dict
        Dictionary defining acuity feature.
    aggregation_method : str
        Method to aggregate multiple readings per case: 'any', 'all', 'first', 'last', 'majority'.
        Default is 'any' (worst-case: 1 if any reading was abnormal).
        
    Returns
    -------
    feature_matrix : pd.DataFrame
        Binary feature matrix with case IDs as index.
    """
    # Use defaults if not provided
    vital_thresholds = vital_thresholds or DEFAULT_VITAL_THRESHOLDS
    categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES
    presence_features = presence_features or DEFAULT_PRESENCE_FEATURES
    acuity_feature = acuity_feature or DEFAULT_ACUITY_FEATURE
    
    case_ids = df[case_id_col].unique()
    features = pd.DataFrame(index=case_ids)
    
    # ----- VITAL SIGN FEATURES -----
    for feature_name, config in vital_thresholds.items():
        col = config['column']
        if col not in df.columns:
            features[feature_name] = 0
            continue
            
        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Apply threshold to identify abnormal readings
        abnormal_mask = _apply_threshold(df[col], config['operator'], config['value'])
        
        if aggregation_method == 'any':
            # ANY: 1 if any reading was abnormal
            abnormal_cases = df[abnormal_mask].groupby(case_id_col).size()
            features[feature_name] = features.index.map(
                lambda x: 1 if x in abnormal_cases.index else 0
            )
        elif aggregation_method == 'all':
            # ALL: 1 only if all readings were abnormal
            case_counts = df[df[col].notna()].groupby(case_id_col).size()
            abnormal_counts = df[abnormal_mask].groupby(case_id_col).size()
            features[feature_name] = features.index.map(
                lambda x: 1 if (x in abnormal_counts.index and 
                               abnormal_counts.get(x, 0) == case_counts.get(x, 0)) else 0
            )
        elif aggregation_method == 'majority':
            # MAJORITY: 1 if >50% of readings were abnormal
            case_counts = df[df[col].notna()].groupby(case_id_col).size()
            abnormal_counts = df[abnormal_mask].groupby(case_id_col).size()
            features[feature_name] = features.index.map(
                lambda x: 1 if abnormal_counts.get(x, 0) > case_counts.get(x, 1) / 2 else 0
            )
    
    # ----- CATEGORICAL FEATURES -----
    cases_grouped = df.groupby(case_id_col)
    for feature_name, config in categorical_features.items():
        col = config['column']
        if col not in df.columns:
            features[feature_name] = 0
            continue
        case_values = cases_grouped[col].first()
        features[feature_name] = (case_values == config['value']).astype(int)
    
    # ----- PRESENCE FEATURES -----
    for feature_name, config in presence_features.items():
        col = config['column']
        if col not in df.columns:
            features[feature_name] = 0
            continue
        has_value = df[df[col].notna()].groupby(case_id_col).size()
        features[feature_name] = features.index.map(
            lambda x: 1 if x in has_value.index else 0
        )
    
    # ----- ACUITY FEATURE -----
    for feature_name, config in acuity_feature.items():
        col = config['column']
        if col not in df.columns:
            features[feature_name] = 0
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        case_acuity = cases_grouped[col].first()
        features[feature_name] = case_acuity.isin(config['values']).astype(int)
        features[feature_name] = features[feature_name].fillna(0).astype(int)
    
    return features


def compute_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute clustering quality metrics.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    labels : array-like
        Cluster labels.
        
    Returns
    -------
    metrics : dict
        Dictionary containing silhouette, calinski_harabasz, and davies_bouldin scores.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    return {
        'silhouette': silhouette_score(X, labels, metric='hamming'),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    }


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11), 
                   random_state: int = 42) -> Tuple[int, pd.DataFrame]:
    """
    Find optimal number of clusters using silhouette score.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    k_range : range
        Range of k values to evaluate.
    random_state : int
        Random seed.
        
    Returns
    -------
    optimal_k : int
        Optimal number of clusters.
    metrics_df : pd.DataFrame
        DataFrame with metrics for each k.
    """
    metrics_list = []
    
    for k in k_range:
        kmedoids = KMedoids(n_clusters=k, metric='hamming', random_state=random_state)
        labels = kmedoids.fit_predict(X)
        metrics = compute_cluster_metrics(X, labels)
        metrics['k'] = k
        metrics['inertia'] = kmedoids.inertia_
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    optimal_k = int(metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k'])
    
    return optimal_k, metrics_df



# MAIN FUNCTION

def perform_feature_based_variant_clustering(
    log: Union[pd.DataFrame, 'pm4py.objects.log.obj.EventLog'],
    case_id_col: str = 'stay_id',
    activity_col: str = 'activity',
    timestamp_col: str = 'time',
    n_clusters: Optional[int] = None,
    vital_thresholds: Dict = None,
    categorical_features: Dict = None,
    presence_features: Dict = None,
    acuity_feature: Dict = None,
    aggregation_method: str = 'any',
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[List, Dict]:
    """
    Perform pattern-based feature generation and K-Medoids clustering
    to identify representative case variants.
    
    This function implements a binary mapping function φ that encodes each case
    with a vector of 0s and 1s based on clinically meaningful properties,
    then applies K-Medoids clustering to identify representative cases (medoids).
    
    Parameters
    ----------
    log : pd.DataFrame or pm4py EventLog
        Event log containing case data.
    case_id_col : str
        Column name for case identifier. Default: 'stay_id'.
    activity_col : str
        Column name for activity. Default: 'activity'.
    timestamp_col : str
        Column name for timestamp. Default: 'time'.
    n_clusters : int, optional
        Number of clusters. If None, optimal k is determined automatically.
    vital_thresholds : dict, optional
        Custom vital sign thresholds. Uses clinical defaults if not provided.
    categorical_features : dict, optional
        Custom categorical features. Uses defaults if not provided.
    presence_features : dict, optional
        Custom presence-based features. Uses defaults if not provided.
    acuity_feature : dict, optional
        Custom acuity feature. Uses defaults if not provided.
    aggregation_method : str
        Method to aggregate multiple readings: 'any' (default), 'all', 'majority'.
    random_state : int
        Random seed for reproducibility. Default: 42.
    verbose : bool
        Whether to print progress information. Default: True.
        
    Returns
    -------
    medoid_case_ids : list
        List of case IDs representing cluster centers (medoids).
    results : dict
        Dictionary containing:
        - 'feature_matrix': Binary feature matrix (pd.DataFrame)
        - 'cluster_labels': Cluster assignment for each case (np.ndarray)
        - 'n_clusters': Number of clusters used (int)
        - 'cluster_metrics': Clustering quality metrics (dict)
        - 'medoid_details': Detailed information about each medoid (list of dicts)
        - 'feature_names': List of feature names (list)
        - 'cluster_summary': Summary statistics per cluster (pd.DataFrame)
        
    Examples
    --------
    >>> import pandas as pd
    >>> from feature_based_variant_clustering import perform_feature_based_variant_clustering
    >>> 
    >>> # Load event log
    >>> df = pd.read_csv('event_log.csv')
    >>> 
    >>> # Perform clustering with automatic k selection
    >>> medoid_ids, results = perform_feature_based_variant_clustering(df)
    >>> 
    >>> # Use medoids for process discovery
    >>> print(f"Medoid Case IDs: {medoid_ids}")
    >>> 
    >>> # Access detailed results
    >>> feature_matrix = results['feature_matrix']
    >>> cluster_labels = results['cluster_labels']
    """
    
    # Convert pm4py EventLog to DataFrame if necessary
    if hasattr(log, '__iter__') and not isinstance(log, pd.DataFrame):
        try:
            import pm4py
            log = pm4py.convert_to_dataframe(log)
        except:
            raise ValueError("Could not convert log to DataFrame. Please provide a pandas DataFrame.")
    
    df = log.copy()
    
    if verbose:
        print("="*70)
        print("PATTERN-BASED FEATURE GENERATION & CLUSTERING")
        print("="*70)
        print(f"\nDataset: {len(df)} events, {df[case_id_col].nunique()} cases")
    
    # Step 1: Generate binary features
    if verbose:
        print(f"\n[Step 1] Generating binary features (aggregation: {aggregation_method})...")
    
    feature_matrix = generate_binary_features(
        df=df,
        case_id_col=case_id_col,
        vital_thresholds=vital_thresholds,
        categorical_features=categorical_features,
        presence_features=presence_features,
        acuity_feature=acuity_feature,
        aggregation_method=aggregation_method
    )
    
    feature_names = feature_matrix.columns.tolist()
    
    if verbose:
        print(f"    Created {len(feature_names)} features for {len(feature_matrix)} cases")
        print(f"    Features: {', '.join(feature_names)}")
    
    # Step 2: Prepare data for clustering
    X = feature_matrix.values
    case_ids = feature_matrix.index.tolist()
    
    # Step 3: Determine optimal k if not provided
    if n_clusters is None:
        if verbose:
            print(f"\n[Step 2] Finding optimal number of clusters...")
        optimal_k, metrics_df = find_optimal_k(X, random_state=random_state)
        n_clusters = min(optimal_k, 6)  # Cap at 6 for interpretability
        if verbose:
            print(f"    Optimal k (by silhouette): {optimal_k}")
            print(f"    Using k = {n_clusters}")
    else:
        metrics_df = None
    
    # Step 4: Perform K-Medoids clustering
    if verbose:
        print(f"\n[Step 3] Performing K-Medoids clustering (k={n_clusters})...")
    
    kmedoids = KMedoids(n_clusters=n_clusters, metric='hamming', random_state=random_state)
    labels = kmedoids.fit_predict(X)
    
    # Get medoid case IDs
    medoid_indices = kmedoids.medoid_indices_
    medoid_case_ids = [case_ids[i] for i in medoid_indices]
    
    # Compute cluster metrics
    cluster_metrics = compute_cluster_metrics(X, labels)
    
    if verbose:
        print(f"    Silhouette Score: {cluster_metrics['silhouette']:.4f}")
        print(f"    Calinski-Harabasz Index: {cluster_metrics['calinski_harabasz']:.1f}")
    
    # Step 5: Compile detailed results
    if verbose:
        print(f"\n[Step 4] Compiling results...")
    
    feature_matrix['cluster'] = labels
    
    # Create medoid details
    medoid_details = []
    cluster_summary_data = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_size = cluster_mask.sum()
        medoid_id = medoid_case_ids[cluster_id]
        
        # Get medoid's feature vector
        medoid_features = feature_matrix.loc[medoid_id, feature_names]
        active_features = [f for f, v in medoid_features.items() if v == 1]
        
        # Get trace for medoid
        medoid_events = df[df[case_id_col] == medoid_id].sort_values(timestamp_col)
        trace = list(medoid_events[activity_col])
        
        # Cluster feature profile
        cluster_data = feature_matrix[feature_matrix['cluster'] == cluster_id]
        feature_means = cluster_data[feature_names].mean()
        dominant_features = [f for f in feature_names if feature_means[f] > 0.5]
        
        medoid_details.append({
            'cluster': cluster_id,
            'medoid_case_id': medoid_id,
            'cluster_size': cluster_size,
            'cluster_coverage': cluster_size / len(case_ids),
            'active_features': active_features,
            'dominant_features': dominant_features,
            'trace': trace,
            'trace_length': len(trace)
        })
        
        cluster_summary_data.append({
            'cluster': cluster_id,
            'medoid_case_id': medoid_id,
            'size': cluster_size,
            'coverage': f"{100*cluster_size/len(case_ids):.1f}%",
            'n_features_active': len(active_features)
        })
    
    cluster_summary = pd.DataFrame(cluster_summary_data)
    
    if verbose:
        print(f"\n{'='*70}")
        print("CLUSTER MEDOIDS (Representative Cases)")
        print("="*70)
        for detail in medoid_details:
            print(f"\nCluster {detail['cluster']}: {detail['cluster_size']} cases "
                  f"({100*detail['cluster_coverage']:.1f}%)")
            print(f"  Medoid Case ID: {detail['medoid_case_id']}")
            print(f"  Active Features: {', '.join(detail['active_features']) if detail['active_features'] else 'None'}")
            print(f"  Trace Length: {detail['trace_length']}")
    
    # Compile results dictionary
    results = {
        'feature_matrix': feature_matrix,
        'cluster_labels': labels,
        'n_clusters': n_clusters,
        'cluster_metrics': cluster_metrics,
        'medoid_indices': medoid_indices,
        'medoid_details': medoid_details,
        'feature_names': feature_names,
        'cluster_summary': cluster_summary,
        'k_evaluation': metrics_df
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULT: {len(medoid_case_ids)} medoid case IDs returned")
        print(f"{'='*70}")
        print(f"\nmedoid_case_ids = {medoid_case_ids}")
    
    return medoid_case_ids, results


def extract_cluster_sublogs(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    case_id_col: str = 'stay_id'
) -> Dict[int, pd.DataFrame]:
    """
    Extract sublogs for each cluster.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original event log.
    feature_matrix : pd.DataFrame
        Feature matrix with cluster assignments.
    case_id_col : str
        Column name for case identifier.
        
    Returns
    -------
    sublogs : dict
        Dictionary mapping cluster ID to sublog DataFrame.
    """
    sublogs = {}
    for cluster_id in feature_matrix['cluster'].unique():
        cluster_cases = feature_matrix[feature_matrix['cluster'] == cluster_id].index
        sublogs[cluster_id] = df[df[case_id_col].isin(cluster_cases)]
    return sublogs


def get_medoid_traces(
    df: pd.DataFrame,
    medoid_case_ids: List,
    case_id_col: str = 'stay_id',
    activity_col: str = 'activity',
    timestamp_col: str = 'time'
) -> Dict:
    """
    Extract traces for medoid cases.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original event log.
    medoid_case_ids : list
        List of medoid case IDs.
    case_id_col : str
        Column name for case identifier.
    activity_col : str
        Column name for activity.
    timestamp_col : str
        Column name for timestamp.
        
    Returns
    -------
    traces : dict
        Dictionary mapping case ID to activity sequence.
    """
    traces = {}
    for case_id in medoid_case_ids:
        case_events = df[df[case_id_col] == case_id].sort_values(timestamp_col)
        traces[case_id] = list(case_events[activity_col])
    return traces


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cluster_profiles(
    feature_matrix: pd.DataFrame,
    feature_names: List[str],
    output_path: str = None,
    figsize: Tuple = (14, 8),
    cmap: str = 'Blues'
) -> None:
    """
    Plot cluster profiles as a heatmap showing feature prevalence.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Feature matrix with 'cluster' column.
    feature_names : list
        List of feature names.
    output_path : str, optional
        Path to save the figure. If None, displays the plot.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for heatmap (single color gradient).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get cluster labels and sizes
    cluster_labels = sorted(feature_matrix['cluster'].unique())
    cluster_sizes = {label: (feature_matrix['cluster'] == label).sum() for label in cluster_labels}
    
    # Calculate mean feature values per cluster
    cluster_profiles = feature_matrix[feature_names].groupby(feature_matrix['cluster']).mean()
    
    # Create heatmap with single color gradient
    sns.heatmap(cluster_profiles, annot=True, fmt='.2f', cmap=cmap, 
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Feature Prevalence'})
    
    # Add cluster sizes to y-axis labels
    total_cases = len(feature_matrix)
    ylabels = [f'Cluster {label}\n(n={cluster_sizes[label]}, {100*cluster_sizes[label]/total_cases:.1f}%)' 
               for label in cluster_labels]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_title('Feature Prevalence by Cluster (Cluster Profiles)', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pattern_variants(
    feature_matrix: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 20,
    output_path: str = None,
    figsize: Tuple = (14, 10),
    cmap: str = 'Blues'
) -> None:
    """
    Plot top N pattern-based variants as horizontal bar chart.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Feature matrix with feature columns.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top patterns to display.
    output_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for bars.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create pattern strings if not present
    def create_pattern_string(row):
        return ''.join(str(int(v)) for v in row[feature_names])
    
    if 'pattern' not in feature_matrix.columns:
        feature_matrix = feature_matrix.copy()
        feature_matrix['pattern'] = feature_matrix.apply(create_pattern_string, axis=1)
    
    # Count patterns
    pattern_counts = feature_matrix['pattern'].value_counts().head(top_n)
    
    # Create labels with decoded features
    pattern_labels = []
    for pattern in pattern_counts.index:
        active = [feature_names[i][:12] for i, bit in enumerate(pattern) if bit == '1']
        if len(active) > 3:
            label = f"{pattern}\n({', '.join(active[:3])}, +{len(active)-3})"
        elif active:
            label = f"{pattern}\n({', '.join(active)})"
        else:
            label = f"{pattern}\n(all normal)"
        pattern_labels.append(label)
    
    # Plot horizontal bar chart
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(pattern_counts)))
    bars = ax.barh(range(len(pattern_counts)), pattern_counts.values, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(pattern_counts)))
    ax.set_yticklabels(pattern_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Cases', fontsize=12)
    ax.set_title(f'Top {top_n} Pattern-Based Variants (Binary Feature Encoding φ)', fontsize=14, fontweight='bold')
    
    # Add value labels
    total_cases = len(feature_matrix)
    for i, (bar, val) in enumerate(zip(bars, pattern_counts.values)):
        pct = 100 * val / total_cases
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val} ({pct:.1f}%)', 
                va='center', fontsize=9)
    
    ax.set_xlim(0, max(pattern_counts.values) * 1.25)
    ax.grid(axis='x', alpha=0.3)
    
    # Add summary text
    total_in_top = pattern_counts.sum()
    n_unique = feature_matrix['pattern'].nunique()
    ax.text(0.98, 0.02, f'Top {top_n} patterns cover {total_in_top} cases ({100*total_in_top/total_cases:.1f}%)\nTotal unique patterns: {n_unique}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cluster_coverage(
    feature_matrix: pd.DataFrame,
    feature_names: List[str],
    medoid_case_ids: List,
    output_path: str = None,
    figsize: Tuple = (18, 14)
) -> None:
    """
    Plot comprehensive cluster coverage visualization with 3 panels:
    A. Pie chart of cluster distribution
    B. Bar chart with medoid case IDs
    C. Stacked bar chart of feature composition
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Feature matrix with 'cluster' column.
    feature_names : list
        List of feature names.
    medoid_case_ids : list
        List of medoid case IDs.
    output_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.25, wspace=0.25)
    
    # Get cluster info
    cluster_labels = sorted(feature_matrix['cluster'].unique())
    n_clusters = len(cluster_labels)
    cluster_sizes = {label: (feature_matrix['cluster'] == label).sum() for label in cluster_labels}
    total_cases = len(feature_matrix)
    
    colors_pie = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    sizes_list = [cluster_sizes[label] for label in cluster_labels]
    
    # --- Panel A: Pie chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(
        sizes_list, 
        labels=[f'Cluster {label}' for label in cluster_labels],
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_cases)})',
        colors=colors_pie,
        explode=[0.03]*n_clusters,
        startangle=90,
        textprops={'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax1.set_title('A. Cluster Distribution (100% Coverage)', fontsize=12, fontweight='bold')
    
    # --- Panel B: Bar chart with medoid IDs ---
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = range(n_clusters)
    bars = ax2.bar(x_pos, sizes_list, color=colors_pie, edgecolor='black', linewidth=1.5)
    
    for idx, (bar, label) in enumerate(zip(bars, cluster_labels)):
        height = bar.get_height()
        medoid_id = medoid_case_ids[label] if isinstance(medoid_case_ids, dict) else medoid_case_ids[idx]
        ax2.text(bar.get_x() + bar.get_width()/2, height + 15, 
                 f'Medoid: {medoid_id}', ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
        ax2.text(bar.get_x() + bar.get_width()/2, height/2,
                 f'{int(height)}\n({100*height/total_cases:.1f}%)', 
                 ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Cluster {label}' for label in cluster_labels], fontsize=11)
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title('B. Cluster Sizes with Medoid Case IDs', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(sizes_list) * 1.4)
    ax2.grid(axis='y', alpha=0.3)
    
    # --- Panel C: Stacked feature composition ---
    ax3 = fig.add_subplot(gs[1, :])
    bar_height = 0.6
    y_positions = np.arange(n_clusters)
    feature_colors = plt.cm.tab20(np.linspace(0, 1, len(feature_names)))
    
    for idx, label in enumerate(cluster_labels):
        cluster_data = feature_matrix[feature_matrix['cluster'] == label]
        left = 0
        for j, feat in enumerate(feature_names):
            feat_count = cluster_data[feat].sum()
            if feat_count > 0:
                ax3.barh(idx, feat_count, left=left, height=bar_height, 
                        color=feature_colors[j], edgecolor='white', linewidth=0.5,
                        label=feat if idx == 0 else '')
            left += feat_count
    
    # Add cluster size annotations
    for idx, label in enumerate(cluster_labels):
        cluster_size = cluster_sizes[label]
        total_features = feature_matrix[feature_matrix['cluster'] == label][feature_names].sum().sum()
        ax3.text(total_features + 30, idx, f'n={cluster_size} ({100*cluster_size/total_cases:.1f}%)', 
                 va='center', fontsize=10, fontweight='bold')
    
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels([f'Cluster {label}' for label in cluster_labels], fontsize=11)
    ax3.set_xlabel('Cumulative Feature Count (total binary features = 1 across all cases in cluster)', fontsize=11)
    ax3.set_title('C. Feature Composition by Cluster (Stacked Bar Chart)', fontsize=12, fontweight='bold')
    
    handles = [Patch(facecolor=feature_colors[i], label=feat) for i, feat in enumerate(feature_names)]
    ax3.legend(handles=handles, loc='upper right', ncol=3, fontsize=9, title='Features', title_fontsize=10)
    
    plt.suptitle('Cluster Coverage Analysis\n(K-Medoids Clustering with Hamming Distance on Binary Features)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cluster_summary_table(
    results: Dict,
    output_path: str = None,
    figsize: Tuple = (16, 7)
) -> None:
    """
    Plot cluster summary as a visual table.
    
    Parameters
    ----------
    results : dict
        Results dictionary from perform_feature_based_variant_clustering.
    output_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Cluster', 'Medoid Case ID', 'Cases', 'Coverage', 'Dominant Features (>50%)', 'Trace']
    
    for detail in results['medoid_details']:
        if detail['cluster_size'] == 0:
            continue
        cluster_id = detail['cluster']
        medoid_id = detail['medoid_case_id']
        size = detail['cluster_size']
        coverage = f"{100*detail['cluster_coverage']:.1f}%"
        dominant = ', '.join(detail['dominant_features'][:5])
        if len(detail['dominant_features']) > 5:
            dominant += f' (+{len(detail["dominant_features"])-5})'
        trace = ' → '.join(detail['trace'][:5])
        if len(detail['trace']) > 5:
            trace += f' → ... ({len(detail["trace"])} total)'
        
        table_data.append([f'Cluster {cluster_id}', str(medoid_id), str(size), coverage, dominant, trace])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='left',
        colColours=['#4472C4']*len(headers),
        colWidths=[0.08, 0.1, 0.06, 0.08, 0.35, 0.33]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_facecolor('#4472C4')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE4')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('Cluster Summary: Medoid Cases for Process Discovery & Conformance Checking', 
                 fontsize=14, fontweight='bold', pad=20)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_all_visualizations(
    feature_matrix: pd.DataFrame,
    feature_names: List[str],
    medoid_case_ids: List,
    results: Dict,
) -> List[str]:

    plot_cluster_profiles(feature_matrix, feature_names, output_path=None)
    plot_pattern_variants(feature_matrix, feature_names, output_path=None)
    plot_cluster_coverage(feature_matrix, feature_names, medoid_case_ids, output_path=None)
    plot_cluster_summary_table(results, output_path=None)


