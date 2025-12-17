"""
PROCESS DISCOVERY AND CONFORMANCE CHECKING
Emergency Department Process Mining Analysis

This script provides:
1. Multiple process discovery algorithms with parameter tuning
2. Conformance checking with F1 Score, Generalization, and Simplicity
3. Pareto-based ranking for multi-objective optimization

Algorithms implemented:
- Inductive Miner (with noise threshold tuning)
- Inductive Miner Infrequent (IMf)
- Inductive Miner Directly-Follows (IMd)
- Heuristic Miner (with dependency/AND threshold tuning)
- Alpha Miner (baseline)

Metrics:
- Fitness (token-based replay)
- Precision (token-based replay)
- F1 Score = 2 * (Fitness * Precision) / (Fitness + Precision)
- Generalization
- Simplicity

"""

import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#28A745',
    'danger': '#DC3545',
    'pareto': '#9B59B6'
}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_event_log(filepath):
    """Load and prepare event log for process mining."""
    print("="*80)
    print("LOADING EVENT LOG")
    print("="*80)
    
    df = pd.read_csv(filepath)
    
    # Rename to pm4py standard format
    log_df = df.rename(columns={
        "stay_id": "case:concept:name",
        "activity": "concept:name",
        "time": "time:timestamp"
    })
    
    log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'])
    log_df['case:concept:name'] = log_df['case:concept:name'].astype(str)
    log_df = log_df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    # Convert to event log format
    log = pm4py.convert_to_event_log(log_df)
    
    # Get statistics
    n_cases = len(log)
    n_events = sum(len(trace) for trace in log)
    n_activities = len(set(event['concept:name'] for trace in log for event in trace))
    n_variants = len(pm4py.get_variants(log))
    
    print(f"\n  Cases: {n_cases:,}")
    print(f"  Events: {n_events:,}")
    print(f"  Activities: {n_activities}")
    print(f"  Variants: {n_variants}")
    
    return log, log_df


# =============================================================================
# CONFORMANCE METRICS CALCULATION
# =============================================================================
def calculate_metrics(log, net, initial_marking, final_marking, model_name="Model"):
    """
    Calculate all conformance metrics for a Petri net model.
    
    Metrics:
    - Fitness: How well the model can replay the log
    - Precision: How much behavior the model allows beyond the log
    - F1 Score: Harmonic mean of fitness and precision
    - Generalization: How well the model generalizes to unseen behavior
    - Simplicity: Structural simplicity of the model
    """
    metrics = {'model_name': model_name}
    
    try:
        # Fitness (token-based replay)
        fitness_result = pm4py.fitness_token_based_replay(log, net, initial_marking, final_marking)
        metrics['fitness'] = fitness_result['average_trace_fitness']
    except Exception as e:
        print(f"    Warning: Fitness calculation failed for {model_name}: {e}")
        metrics['fitness'] = 0.0
    
    try:
        # Precision (token-based replay)
        precision_result = pm4py.precision_token_based_replay(log, net, initial_marking, final_marking)
        metrics['precision'] = precision_result
    except Exception as e:
        print(f"    Warning: Precision calculation failed for {model_name}: {e}")
        metrics['precision'] = 0.0
    
    # F1 Score
    if metrics['fitness'] > 0 and metrics['precision'] > 0:
        metrics['f1_score'] = 2 * (metrics['fitness'] * metrics['precision']) / (metrics['fitness'] + metrics['precision'])
    else:
        metrics['f1_score'] = 0.0
    
    try:
        # Generalization
        gen_result = pm4py.generalization_tbr(log, net, initial_marking, final_marking)
        metrics['generalization'] = gen_result
    except Exception as e:
        print(f"    Warning: Generalization calculation failed for {model_name}: {e}")
        metrics['generalization'] = 0.0
    
    try:
        # Simplicity (using pm4py's simplicity_petri_net function with markings)
        simp_result = pm4py.simplicity_petri_net(net, initial_marking, final_marking)
        metrics['simplicity'] = simp_result
    except Exception as e:
        # Fallback: calculate simplicity based on arc degree
        try:
            # Arc degree simplicity: inverse of average arc degree
            n_places = len(net.places)
            n_transitions = len(net.transitions)
            n_arcs = len(net.arcs)
            n_nodes = n_places + n_transitions
            if n_nodes > 0:
                avg_arc_degree = n_arcs / n_nodes
                # Simplicity decreases as arc degree increases
                # Using formula: 1 / (1 + avg_arc_degree) normalized
                simp_result = 1.0 / (1.0 + avg_arc_degree)
                metrics['simplicity'] = simp_result
            else:
                metrics['simplicity'] = 0.0
        except:
            print(f"    Warning: Simplicity calculation failed for {model_name}: {e}")
            metrics['simplicity'] = 0.0
    
    # Count model elements
    metrics['n_places'] = len(net.places)
    metrics['n_transitions'] = len(net.transitions)
    metrics['n_arcs'] = len(net.arcs)
    
    return metrics


# =============================================================================
# PROCESS DISCOVERY ALGORITHMS WITH PARAMETER TUNING
# =============================================================================
def discover_with_inductive_miner(log, noise_threshold=0.0):
    """
    Discover process model using Inductive Miner.
    
    Parameters:
    -----------
    noise_threshold : float (0.0 to 1.0)
        Higher values filter more infrequent behavior
        - 0.0: No filtering (all behavior included)
        - 0.2: Filter 20% most infrequent paths
        - 0.5: Filter 50% most infrequent paths
    """
    try:
        if noise_threshold > 0:
            # Use Inductive Miner Infrequent (IMf)
            process_tree = pm4py.discover_process_tree_inductive(
                log, 
                noise_threshold=noise_threshold
            )
        else:
            # Use standard Inductive Miner
            process_tree = pm4py.discover_process_tree_inductive(log)
        
        # Convert to Petri net
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)
        
        return net, initial_marking, final_marking, process_tree
    except Exception as e:
        print(f"    Error in Inductive Miner (noise={noise_threshold}): {e}")
        return None, None, None, None


def discover_with_heuristics_miner(log, dependency_threshold=0.5, and_threshold=0.65, 
                                    loop_two_threshold=0.5):
    """
    Discover process model using Heuristics Miner.
    
    Parameters:
    -----------
    dependency_threshold : float (0.0 to 1.0)
        Minimum dependency value to include a relation
        - Lower: More connections (complex model)
        - Higher: Fewer connections (simpler model)
    
    and_threshold : float (0.0 to 1.0)
        Threshold for AND-split/join detection
        - Lower: More parallel gateways detected
        - Higher: Fewer parallel gateways
    
    loop_two_threshold : float (0.0 to 1.0)
        Threshold for detecting length-two loops
    """
    try:
        # Discover Heuristics Net
        heu_net = pm4py.discover_heuristics_net(
            log,
            dependency_threshold=dependency_threshold,
            and_threshold=and_threshold,
            loop_two_threshold=loop_two_threshold
        )
        
        # Convert to Petri net
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(heu_net)
        
        return net, initial_marking, final_marking, heu_net
    except Exception as e:
        print(f"    Error in Heuristics Miner: {e}")
        return None, None, None, None


def discover_with_alpha_miner(log):
    """
    Discover process model using Alpha Miner.
    
    Note: Alpha Miner is a baseline algorithm and may not handle
    complex patterns like short loops or invisible tasks well.
    """
    try:
        net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(log)
        return net, initial_marking, final_marking, None
    except Exception as e:
        print(f"    Error in Alpha Miner: {e}")
        return None, None, None, None


def run_parameter_search(log):
    """
    Run comprehensive parameter search across all algorithms.
    
    This function tests multiple parameter combinations to find
    the best models according to F1, Generalization, and Simplicity.
    """
    print("\n" + "="*80)
    print("PROCESS DISCOVERY - PARAMETER SEARCH")
    print("="*80)
    
    results = []
    models = {}  # Store models for later visualization
    
    # 1. INDUCTIVE MINER with noise threshold tuning
    print("\n" + "-"*60)
    print("1. INDUCTIVE MINER (with noise threshold tuning)")
    print("-"*60)
    
    # Noise thresholds to test
    # For high-variant logs, higher noise thresholds often work better
    noise_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for noise in noise_thresholds:
        model_name = f"IM_noise_{noise}"
        print(f"\n  Testing {model_name}...")
        
        net, im, fm, tree = discover_with_inductive_miner(log, noise_threshold=noise)
        
        if net is not None:
            metrics = calculate_metrics(log, net, im, fm, model_name)
            metrics['algorithm'] = 'Inductive Miner'
            metrics['parameters'] = f'noise={noise}'
            results.append(metrics)
            models[model_name] = (net, im, fm, tree)
            
            print(f"    Fitness: {metrics['fitness']:.4f} | Precision: {metrics['precision']:.4f} | "
                  f"F1: {metrics['f1_score']:.4f} | Gen: {metrics['generalization']:.4f} | "
                  f"Simp: {metrics['simplicity']:.4f}")
    
    # 2. HEURISTICS MINER with parameter tuning
    print("\n" + "-"*60)
    print("2. HEURISTICS MINER (with parameter tuning)")
    print("-"*60)
    
    # Parameter grid for Heuristics Miner
    # For high parallelism, lower and_threshold values help detect parallel structures
    # Optimized grid for 943/1307 variants
    dependency_thresholds = [0.3, 0.5, 0.7, 0.9]
    and_thresholds = [0.3, 0.5, 0.65, 0.8]
    
    for dep_th, and_th in product(dependency_thresholds, and_thresholds):
        model_name = f"HM_dep_{dep_th}_and_{and_th}"
        print(f"\n  Testing {model_name}...")
        
        net, im, fm, heu_net = discover_with_heuristics_miner(
            log, 
            dependency_threshold=dep_th,
            and_threshold=and_th
        )
        
        if net is not None:
            metrics = calculate_metrics(log, net, im, fm, model_name)
            metrics['algorithm'] = 'Heuristics Miner'
            metrics['parameters'] = f'dep={dep_th}, and={and_th}'
            results.append(metrics)
            models[model_name] = (net, im, fm, heu_net)
            
            print(f"    Fitness: {metrics['fitness']:.4f} | Precision: {metrics['precision']:.4f} | "
                  f"F1: {metrics['f1_score']:.4f} | Gen: {metrics['generalization']:.4f} | "
                  f"Simp: {metrics['simplicity']:.4f}")
    

    # 3. ALPHA MINER (baseline)
    print("\n" + "-"*60)
    print("3. ALPHA MINER (baseline)")
    print("-"*60)
    
    model_name = "Alpha_Miner"
    print(f"\n  Testing {model_name}...")
    
    net, im, fm, _ = discover_with_alpha_miner(log)
    
    if net is not None:
        metrics = calculate_metrics(log, net, im, fm, model_name)
        metrics['algorithm'] = 'Alpha Miner'
        metrics['parameters'] = 'default'
        results.append(metrics)
        models[model_name] = (net, im, fm, None)
        
        print(f"    Fitness: {metrics['fitness']:.4f} | Precision: {metrics['precision']:.4f} | "
              f"F1: {metrics['f1_score']:.4f} | Gen: {metrics['generalization']:.4f} | "
              f"Simp: {metrics['simplicity']:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, models



# PARETO RANKING
def pareto_dominates(row1, row2, objectives):
    """
    Check if row1 Pareto-dominates row2.
    
    row1 dominates row2 if:
    - row1 is at least as good as row2 in all objectives
    - row1 is strictly better than row2 in at least one objective
    
    All objectives are maximization (higher is better).
    """
    at_least_as_good = all(row1[obj] >= row2[obj] for obj in objectives)
    strictly_better = any(row1[obj] > row2[obj] for obj in objectives)
    return at_least_as_good and strictly_better


def calculate_pareto_ranking(results_df, objectives=['f1_score', 'generalization', 'simplicity']):
    """
    Calculate Pareto ranking for multi-objective optimization.
    
    Pareto Rank 1 = Non-dominated solutions (Pareto front)
    Pareto Rank 2 = Dominated only by Rank 1 solutions
    ... and so on
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model results
    objectives : list
        List of column names to optimize (all maximized)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'pareto_rank' column added
    """
    print("\n" + "="*80)
    print("PARETO-BASED RANKING")
    print("="*80)
    print(f"\n  Objectives (all maximized): {objectives}")
    
    df = results_df.copy()
    n = len(df)
    
    # Initialize all as unranked
    df['pareto_rank'] = 0
    remaining_indices = set(df.index)
    current_rank = 1
    
    while remaining_indices:
        # Find non-dominated solutions among remaining
        non_dominated = []
        
        for i in remaining_indices:
            is_dominated = False
            for j in remaining_indices:
                if i != j:
                    if pareto_dominates(df.loc[j], df.loc[i], objectives):
                        is_dominated = True
                        break
            
            if not is_dominated:
                non_dominated.append(i)
        
        # Assign current rank to non-dominated solutions
        for idx in non_dominated:
            df.loc[idx, 'pareto_rank'] = current_rank
            remaining_indices.remove(idx)
        
        current_rank += 1
    
    # Sort by Pareto rank, then by F1 score within each rank
    df = df.sort_values(['pareto_rank', 'f1_score'], ascending=[True, False])
    
    # Print Pareto front (Rank 1)
    pareto_front = df[df['pareto_rank'] == 1]
    print(f"\n  PARETO FRONT (Rank 1): {len(pareto_front)} models")
    print("  " + "-"*70)
    
    for _, row in pareto_front.iterrows():
        print(f"    {row['model_name']:<30}")
        print(f"      F1: {row['f1_score']:.4f} | Gen: {row['generalization']:.4f} | "
              f"Simp: {row['simplicity']:.4f}")
        print(f"      Algorithm: {row['algorithm']} | Params: {row['parameters']}")
    
    # Summary by rank
    print(f"\n  RANKING SUMMARY:")
    rank_counts = df['pareto_rank'].value_counts().sort_index()
    for rank, count in rank_counts.items():
        print(f"    Rank {rank}: {count} models")
    
    return df



# VISUALIZATION
def visualize_pareto_analysis(results_df):
    """
    Create comprehensive visualization of Pareto analysis results.
    """
    print("\n" + "="*80)
    print("GENERATING PARETO ANALYSIS VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Get Pareto front
    pareto_front = results_df[results_df['pareto_rank'] == 1]
    non_pareto = results_df[results_df['pareto_rank'] > 1]
    
    # ===== Plot 1: F1 Score vs Generalization =====
    ax1 = axes[0, 0]
    
    # Plot non-Pareto points
    scatter1 = ax1.scatter(non_pareto['f1_score'], non_pareto['generalization'],
                           c=non_pareto['pareto_rank'], cmap='Reds', 
                           s=100, alpha=0.6, edgecolors='white')
    
    # Plot Pareto front
    ax1.scatter(pareto_front['f1_score'], pareto_front['generalization'],
               c=COLORS['pareto'], s=200, marker='*', edgecolors='black',
               linewidths=1, label='Pareto Front', zorder=5)
    
    # Add labels for Pareto front models
    for _, row in pareto_front.iterrows():
        ax1.annotate(row['model_name'].replace('_', '\n'), 
                    (row['f1_score'], row['generalization']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax1.set_xlabel('F1 Score', fontsize=11)
    ax1.set_ylabel('Generalization', fontsize=11)
    ax1.set_title('F1 Score vs Generalization\n(★ = Pareto Front)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Pareto Rank')
    
    # ===== Plot 2: F1 Score vs Simplicity =====
    ax2 = axes[0, 1]
    
    scatter2 = ax2.scatter(non_pareto['f1_score'], non_pareto['simplicity'],
                           c=non_pareto['pareto_rank'], cmap='Reds',
                           s=100, alpha=0.6, edgecolors='white')
    
    ax2.scatter(pareto_front['f1_score'], pareto_front['simplicity'],
               c=COLORS['pareto'], s=200, marker='*', edgecolors='black',
               linewidths=1, label='Pareto Front', zorder=5)
    
    for _, row in pareto_front.iterrows():
        ax2.annotate(row['model_name'].replace('_', '\n'),
                    (row['f1_score'], row['simplicity']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax2.set_xlabel('F1 Score', fontsize=11)
    ax2.set_ylabel('Simplicity', fontsize=11)
    ax2.set_title('F1 Score vs Simplicity\n(★ = Pareto Front)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Pareto Rank')
    
    # ===== Plot 3: 3D Pareto Front =====
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    
    # Plot all points
    scatter3 = ax3.scatter(results_df['f1_score'], 
                           results_df['generalization'],
                           results_df['simplicity'],
                           c=results_df['pareto_rank'], cmap='RdYlGn_r',
                           s=100, alpha=0.7)
    
    # Highlight Pareto front
    ax3.scatter(pareto_front['f1_score'],
               pareto_front['generalization'],
               pareto_front['simplicity'],
               c=COLORS['pareto'], s=200, marker='*', 
               edgecolors='black', linewidths=1)
    
    ax3.set_xlabel('F1 Score')
    ax3.set_ylabel('Generalization')
    ax3.set_zlabel('Simplicity')
    ax3.set_title('3D Pareto Space\n(★ = Pareto Front)', fontsize=12, fontweight='bold')
    
    # ===== Plot 4: Algorithm Comparison =====
    ax4 = axes[1, 1]
    
    # Group by algorithm
    algo_groups = results_df.groupby('algorithm').agg({
        'f1_score': ['mean', 'max', 'std'],
        'generalization': ['mean', 'max'],
        'simplicity': ['mean', 'max'],
        'pareto_rank': 'min'  # Best rank achieved
    }).round(4)
    
    algo_groups.columns = ['_'.join(col) for col in algo_groups.columns]
    algo_groups = algo_groups.reset_index()
    
    # Create grouped bar chart
    x = np.arange(len(algo_groups))
    width = 0.25
    
    bars1 = ax4.bar(x - width, algo_groups['f1_score_max'], width, 
                    label='Best F1', color=COLORS['primary'], alpha=0.8)
    bars2 = ax4.bar(x, algo_groups['generalization_max'], width,
                    label='Best Gen', color=COLORS['secondary'], alpha=0.8)
    bars3 = ax4.bar(x + width, algo_groups['simplicity_max'], width,
                    label='Best Simp', color=COLORS['accent'], alpha=0.8)
    
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Score')
    ax4.set_title('Best Scores by Algorithm', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algo_groups['algorithm'], rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('05_pareto_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 05_pareto_analysis.png")
    
    return fig


def visualize_metrics_heatmap(results_df):
    """
    Create heatmap of all metrics across models.
    """
    fig, ax = plt.subplots(figsize=(14, max(8, len(results_df) * 0.4)))
    
    # Select metrics for heatmap
    metrics = ['fitness', 'precision', 'f1_score', 'generalization', 'simplicity']
    
    # Prepare data
    heatmap_data = results_df.set_index('model_name')[metrics]
    
    # Sort by Pareto rank then F1
    sort_order = results_df.sort_values(['pareto_rank', 'f1_score'], 
                                         ascending=[True, False])['model_name']
    heatmap_data = heatmap_data.loc[sort_order]
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Score'})
    
    # Highlight Pareto front rows
    pareto_models = results_df[results_df['pareto_rank'] == 1]['model_name'].tolist()
    for i, model in enumerate(sort_order):
        if model in pareto_models:
            ax.add_patch(plt.Rectangle((0, i), len(metrics), 1, 
                                       fill=False, edgecolor=COLORS['pareto'], 
                                       linewidth=3))
    
    ax.set_title('Conformance Metrics Heatmap\n(Purple border = Pareto Front)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig('05_metrics_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 05_metrics_heatmap.png")
    
    return fig


def visualize_parameter_sensitivity(results_df):
    """
    Visualize how parameters affect metrics for each algorithm.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Plot 1: Inductive Miner - Noise Threshold Effect =====
    ax1 = axes[0]
    
    im_results = results_df[results_df['algorithm'] == 'Inductive Miner'].copy()
    if len(im_results) > 0:
        # Extract noise value from parameters
        im_results['noise'] = im_results['parameters'].str.extract(r'noise=(\d+\.?\d*)').astype(float)
        im_results = im_results.sort_values('noise')
        
        ax1.plot(im_results['noise'], im_results['f1_score'], 'o-', 
                label='F1 Score', color=COLORS['primary'], linewidth=2, markersize=8)
        ax1.plot(im_results['noise'], im_results['generalization'], 's--',
                label='Generalization', color=COLORS['secondary'], linewidth=2, markersize=8)
        ax1.plot(im_results['noise'], im_results['simplicity'], '^:',
                label='Simplicity', color=COLORS['accent'], linewidth=2, markersize=8)
        
        ax1.set_xlabel('Noise Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Inductive Miner: Effect of Noise Threshold', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 1.05)
    
    # ===== Plot 2: Heuristics Miner - Parameter Grid =====
    ax2 = axes[1]
    
    hm_results = results_df[results_df['algorithm'] == 'Heuristics Miner'].copy()
    if len(hm_results) > 0:
        # Create pivot table for F1 scores
        hm_results['dep'] = hm_results['parameters'].str.extract(r'dep=(\d+\.?\d*)').astype(float)
        hm_results['and'] = hm_results['parameters'].str.extract(r'and=(\d+\.?\d*)').astype(float)
        
        pivot = hm_results.pivot(index='dep', columns='and', values='f1_score')
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax2, vmin=0, vmax=1)
        
        ax2.set_xlabel('AND Threshold')
        ax2.set_ylabel('Dependency Threshold')
        ax2.set_title('Heuristics Miner: F1 Score by Parameters', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('05_parameter_sensitivity.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 05_parameter_sensitivity.png")
    
    return fig


# =============================================================================
# EXPORT BEST MODELS
# =============================================================================
def export_best_models(results_df, models, log, top_n=3):
    """
    Export and visualize the top N Pareto-optimal models.
    """
    print("\n" + "="*80)
    print(f"EXPORTING TOP {top_n} PARETO-OPTIMAL MODELS")
    print("="*80)
    
    pareto_front = results_df[results_df['pareto_rank'] == 1].head(top_n)
    
    for i, (_, row) in enumerate(pareto_front.iterrows(), 1):
        model_name = row['model_name']
        print(f"\n  {i}. {model_name}")
        print(f"     F1: {row['f1_score']:.4f} | Gen: {row['generalization']:.4f} | "
              f"Simp: {row['simplicity']:.4f}")
        
        if model_name in models:
            net, im, fm, _ = models[model_name]
            
            # Save Petri net visualization
            try:
                gviz = pn_visualizer.apply(net, im, fm,
                                          parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
                pn_visualizer.save(gviz, f"05_petri_net_{i}_{model_name}.png")
                print(f"     Saved: 05_petri_net_{i}_{model_name}.png")
            except Exception as e:
                print(f"     Warning: Could not save Petri net visualization: {e}")



# SUMMARY REPORT
def print_summary_report(results_df):
    """Print a comprehensive summary report of the analysis."""
    
    print("\n" + "="*80)
    print("PROCESS DISCOVERY & CONFORMANCE - SUMMARY REPORT")
    print("="*80)
    
    # Overall statistics
    print(f"\n  MODELS EVALUATED: {len(results_df)}")
    print(f"  ALGORITHMS TESTED: {results_df['algorithm'].nunique()}")
    
    # Best models by each metric
    print("\n  BEST MODEL BY METRIC:")
    print("  " + "-"*60)
    
    for metric in ['f1_score', 'generalization', 'simplicity']:
        best_idx = results_df[metric].idxmax()
        best = results_df.loc[best_idx]
        print(f"    Best {metric.upper()}: {best['model_name']}")
        print(f"      Value: {best[metric]:.4f} | Algorithm: {best['algorithm']}")
    
    # Pareto front summary
    pareto_front = results_df[results_df['pareto_rank'] == 1]
    print(f"\n  PARETO FRONT: {len(pareto_front)} models")
    print("  " + "-"*60)
    
    for _, row in pareto_front.iterrows():
        print(f"    {row['model_name']}")
        print(f"      F1={row['f1_score']:.4f}, Gen={row['generalization']:.4f}, "
              f"Simp={row['simplicity']:.4f}")
    
    # Recommended model (highest F1 on Pareto front)
    best_pareto = pareto_front.loc[pareto_front['f1_score'].idxmax()]
    print(f"\n  RECOMMENDED MODEL (Highest F1 on Pareto Front):")
    print("  " + "-"*60)
    print(f"    Model: {best_pareto['model_name']}")
    print(f"    Algorithm: {best_pareto['algorithm']}")
    print(f"    Parameters: {best_pareto['parameters']}")
    print(f"    Fitness: {best_pareto['fitness']:.4f}")
    print(f"    Precision: {best_pareto['precision']:.4f}")
    print(f"    F1 Score: {best_pareto['f1_score']:.4f}")
    print(f"    Generalization: {best_pareto['generalization']:.4f}")
    print(f"    Simplicity: {best_pareto['simplicity']:.4f}")
    
    return best_pareto

def perform_process_discovery(log):
    """Main execution function."""
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "   PROCESS DISCOVERY AND CONFORMANCE CHECKING".center(78) + "█")
    print("█" + "   with Pareto-Based Multi-Objective Optimization".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    # Run parameter search
    results_df, models = run_parameter_search(log)
    
    # Calculate Pareto ranking
    results_df = calculate_pareto_ranking(
        results_df, 
        objectives=['f1_score', 'generalization', 'simplicity']
    )
    
    # Generate visualizations
    visualize_pareto_analysis(results_df)
    visualize_metrics_heatmap(results_df)
    visualize_parameter_sensitivity(results_df)
    
    # Export best models
    export_best_models(results_df, models, log, top_n=3)
    
    # Print summary report
    best_model = print_summary_report(results_df)
    
    # Save results to CSV
    results_df.to_csv('05_discovery_results.csv', index=False)
    print("\n  Saved: 05_discovery_results.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results_df, models