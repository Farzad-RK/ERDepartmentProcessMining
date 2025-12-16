import pandas as pd
import numpy as np
import pm4py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#28A745',
    'danger': '#DC3545',
    'neutral': '#6C757D',
    'light': '#E9ECEF'
}


def perform_variant_analysis(log):
    """
    Perform variant analysis with power-law distribution testing.
    
    A variant is a unique sequence of activities observed in a case.
    Power-law/Pareto distribution is expected: few variants cover most cases.
    """
    print("\n" + "="*80)
    print("VARIANT ANALYSIS")
    print("="*80)
    
    # Get variants using pm4py (returns dict with variant_tuple: count)
    variants = pm4py.get_variants(log)
    
    # Also get case-to-variant mapping for later analysis
    case_variants = {}
    for case_id, case_df in log.groupby('case:concept:name'):
        case_df = case_df.sort_values('time:timestamp')
        variant_tuple = tuple(case_df['concept:name'].tolist())
        if variant_tuple not in case_variants:
            case_variants[variant_tuple] = []
        case_variants[variant_tuple].append(case_id)
    
    # Create variant dataframe
    variant_data = []
    for variant_tuple, count in variants.items():
        variant_str = ' -> '.join(variant_tuple)
        case_ids = case_variants.get(variant_tuple, [])
        variant_data.append({
            'variant': variant_str,
            'variant_tuple': variant_tuple,
            'frequency': count,
            'case_ids': case_ids
        })
    
    variant_df = pd.DataFrame(variant_data)
    variant_df = variant_df.sort_values('frequency', ascending=False).reset_index(drop=True)
    variant_df['rank'] = range(1, len(variant_df) + 1)
    variant_df['cumulative_freq'] = variant_df['frequency'].cumsum()
    variant_df['cumulative_pct'] = 100 * variant_df['cumulative_freq'] / variant_df['frequency'].sum()
    variant_df['variant_length'] = variant_df['variant_tuple'].apply(len)
    
    # Summary statistics
    total_cases = variant_df['frequency'].sum()
    n_variants = len(variant_df)
    
    print(f"\n  Total cases: {total_cases:,}")
    print(f"  Total unique variants: {n_variants}")
    print(f"  Variants per case: {n_variants/total_cases:.3f}")
    
    # Pareto analysis
    top_20_pct_variants = int(np.ceil(0.2 * n_variants))
    coverage_by_top20 = variant_df.iloc[:top_20_pct_variants]['frequency'].sum() / total_cases * 100
    
    print(f"\n  Pareto Analysis:")
    print(f"    Top 20% of variants ({top_20_pct_variants}) cover {coverage_by_top20:.1f}% of cases")
    
    # How many variants needed for 80% coverage?
    variants_for_80 = (variant_df['cumulative_pct'] >= 80).idxmax() + 1
    print(f"    {variants_for_80} variants needed to cover 80% of cases ({100*variants_for_80/n_variants:.1f}% of variants)")
    
    # Top variants
    print("\n  Top 10 most frequent variants:")
    print("  " + "-"*70)
    for i, row in variant_df.head(10).iterrows():
        print(f"    {row['rank']:2d}. Freq: {row['frequency']:4d} ({100*row['frequency']/total_cases:5.1f}%) | "
              f"Len: {row['variant_length']:2d} | Cum: {row['cumulative_pct']:.1f}%")
    
    return variant_df, variants


def test_power_law_distribution(variant_df):
    """
    Test if variant frequencies follow a power-law (Zipf) distribution.
    
    Power law: frequency ~ rank^(-alpha)
    In log-log scale, this appears as a straight line with slope -alpha
    """
    print("\n" + "-"*60)
    print("  POWER-LAW DISTRIBUTION TEST")
    print("-"*60)
    
    # Prepare data for fitting
    ranks = variant_df['rank'].values
    frequencies = variant_df['frequency'].values
    
    # Log-log transformation
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)
    
    # Linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
    
    # Zipf exponent (alpha)
    alpha = -slope
    r_squared = r_value**2
    
    print(f"\n  Power-law fit (log-log linear regression):")
    print(f"    Zipf exponent (alpha):     {alpha:.3f}")
    print(f"    R-squared:             {r_squared:.4f}")
    print(f"    Slope:                 {slope:.3f}")
    print(f"    p-value:               {p_value:.2e}")
    
    if r_squared > 0.9:
        print(f"\n  STRONG power-law behavior (R^2 > 0.9)")
    elif r_squared > 0.7:
        print(f"\n  ~ Moderate power-law behavior (0.7 < R^2 < 0.9)")
    else:
        print(f"\n  X Weak power-law behavior (R^2 < 0.7)")
    
    # Store fit parameters
    fit_params = {
        'alpha': alpha,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value
    }
    
    return fit_params


def visualize_variant_distribution(variant_df, fit_params):
    """
    Create comprehensive visualization of variant distribution.
    """
    print("\n" + "="*80)
    print("GENERATING VARIANT DISTRIBUTION VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Frequency Distribution (Bar Chart - Top 20)
    ax1 = axes[0, 0]
    top_n = min(20, len(variant_df))
    top_variants = variant_df.head(top_n)
    
    bars = ax1.bar(range(top_n), top_variants['frequency'], color=COLORS['primary'], edgecolor='white')
    ax1.set_xlabel('Variant Rank')
    ax1.set_ylabel('Frequency (Number of Cases)')
    ax1.set_title(f'Top {top_n} Most Frequent Variants')
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(range(1, top_n + 1))
    
    # Add percentage labels on bars
    total = variant_df['frequency'].sum()
    for i, (bar, freq) in enumerate(zip(bars, top_variants['frequency'])):
        if i < 10:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                    f'{100*freq/total:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Log-Log Plot with Power-Law Fit
    ax2 = axes[0, 1]
    
    ranks = variant_df['rank'].values
    frequencies = variant_df['frequency'].values
    
    # Scatter plot
    ax2.scatter(ranks, frequencies, alpha=0.6, s=30, color=COLORS['primary'], label='Data')
    
    # Power-law fit line
    x_fit = np.linspace(1, len(ranks), 100)
    y_fit = 10**(fit_params['intercept'] + fit_params['slope'] * np.log10(x_fit))
    ax2.plot(x_fit, y_fit, 'r-', linewidth=2, 
             label=f'Power-law fit (a={fit_params["alpha"]:.2f}, R2={fit_params["r_squared"]:.3f})')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Variant Rank (log scale)')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Power-Law (Zipf) Distribution Test\n(Log-Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Distribution (Pareto Chart)
    ax3 = axes[1, 0]
    
    ax3.bar(range(len(variant_df)), variant_df['frequency'], 
            color=COLORS['primary'], alpha=0.7, label='Frequency')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(len(variant_df)), variant_df['cumulative_pct'], 
                  color=COLORS['danger'], linewidth=2, label='Cumulative %')
    ax3_twin.axhline(80, color=COLORS['accent'], linestyle='--', alpha=0.7, label='80% line')
    
    # Find 80% point
    idx_80 = (variant_df['cumulative_pct'] >= 80).idxmax()
    ax3_twin.axvline(idx_80, color=COLORS['accent'], linestyle=':', alpha=0.7)
    ax3_twin.annotate(f'{idx_80+1} variants\nfor 80%', xy=(idx_80, 80), 
                      xytext=(idx_80 + len(variant_df)*0.1, 70),
                      arrowprops=dict(arrowstyle='->', color=COLORS['accent']))
    
    ax3.set_xlabel('Variant Rank')
    ax3.set_ylabel('Frequency', color=COLORS['primary'])
    ax3_twin.set_ylabel('Cumulative %', color=COLORS['danger'])
    ax3.set_title('Pareto Chart of Variant Distribution')
    ax3_twin.set_ylim(0, 105)
    
    # Plot 4: Variant Length Distribution
    ax4 = axes[1, 1]
    
    length_dist = variant_df.groupby('variant_length')['frequency'].sum()
    ax4.bar(length_dist.index, length_dist.values, color=COLORS['secondary'], edgecolor='white')
    ax4.set_xlabel('Variant Length (Number of Activities)')
    ax4.set_ylabel('Total Cases')
    ax4.set_title('Distribution of Cases by Variant Length')
    
    # Add mean line
    weighted_mean = (variant_df['variant_length'] * variant_df['frequency']).sum() / variant_df['frequency'].sum()
    ax4.axvline(weighted_mean, color=COLORS['danger'], linestyle='--', 
                label=f'Weighted Mean: {weighted_mean:.1f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()



def perform_sequential_variant_analysis(log):
    variant_df, variants = perform_variant_analysis(log)
    fit_params = test_power_law_distribution(variant_df)
    visualize_variant_distribution(variant_df, fit_params)
