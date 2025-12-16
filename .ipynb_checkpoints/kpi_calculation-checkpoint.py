"""
================================================================================
KPI VISUALIZATION FUNCTIONS
================================================================================
Emergency Department Process Mining - Visualization Functions

Contains 4 visualization functions:
1. plot_cumulative_lead_time() - Activity-level cumulative lead time
2. plot_domain_specific_kpis() - Door-to-Triage, Door-to-Treatment, etc.
3. plot_nurse_workload() - Nurse workload distribution by job type
4. plot_los_by_acuity() - Length of Stay by Acuity Level

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Color configuration
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#28A745',
    'danger': '#DC3545',
    'neutral': '#6C757D',
    'light': '#E9ECEF'
}

ACUITY_COLORS = {1: '#DC3545', 2: '#FD7E14', 3: '#FFC107', 4: '#28A745', 5: '#17A2B8'}
ACUITY_LABELS = {1: 'Level 1\n(Most Urgent)', 2: 'Level 2', 3: 'Level 3', 
                 4: 'Level 4', 5: 'Level 5\n(Least Urgent)'}


# =============================================================================
# FUNCTION 1: CUMULATIVE LEAD TIME (Activity-Level KPI)
# =============================================================================
def plot_cumulative_lead_time(log: pd.DataFrame):
    """
    Plot cumulative lead time in hours for each activity.
    
    This shows how time accumulates as a case progresses through activities.
    Each activity's bar represents the average cumulative time from case start
    when that activity occurs.
    
    Parameters:
    -----------
    log : pd.DataFrame
        Event log with columns: case:concept:name, concept:name, time:timestamp
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Ensure timestamp is datetime
    log = log.copy()
    if not pd.api.types.is_datetime64_any_dtype(log['time:timestamp']):
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    
    # Calculate time from case start for each event
    case_starts = log.groupby('case:concept:name')['time:timestamp'].min()
    log['case_start'] = log['case:concept:name'].map(case_starts)
    log['time_from_start_hours'] = (log['time:timestamp'] - log['case_start']).dt.total_seconds() / 3600
    
    # Calculate statistics per activity
    activity_stats = log.groupby('concept:name').agg({
        'time_from_start_hours': ['mean', 'median', 'std', 'count']
    }).round(3)
    activity_stats.columns = ['mean_hours', 'median_hours', 'std_hours', 'count']
    activity_stats = activity_stats.sort_values('mean_hours')
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Plot 1: Bar chart with error bars =====
    ax1 = axes[0]
    activities = activity_stats.index.tolist()
    y_pos = np.arange(len(activities))
    
    bars = ax1.barh(y_pos, activity_stats['mean_hours'], 
                    xerr=activity_stats['std_hours'],
                    color=COLORS['primary'], alpha=0.8, capsize=4,
                    error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # Add median markers
    ax1.scatter(activity_stats['median_hours'], y_pos, 
                color=COLORS['danger'], s=80, zorder=5, 
                marker='|', linewidths=2, label='Median')
    
    # Add value labels
    for i, (mean_val, median_val) in enumerate(zip(activity_stats['mean_hours'], 
                                                    activity_stats['median_hours'])):
        ax1.text(mean_val + activity_stats['std_hours'].iloc[i] + 0.3, i, 
                f'{mean_val:.2f}h', va='center', fontsize=9)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(activities)
    ax1.set_xlabel('Cumulative Lead Time (Hours)')
    ax1.set_title('Cumulative Lead Time by Activity\n(Mean ± Std Dev, | = Median)')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # ===== Plot 2: Box plots for distribution =====
    ax2 = axes[1]
    
    # Prepare data for boxplot
    box_data = [log[log['concept:name'] == act]['time_from_start_hours'].values 
                for act in activities]
    
    bp = ax2.boxplot(box_data, vert=False, patch_artist=True,
                     labels=activities)
    
    # Color the boxes
    colors_gradient = plt.cm.Blues(np.linspace(0.3, 0.9, len(activities)))
    for patch, color in zip(bp['boxes'], colors_gradient):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Cumulative Lead Time (Hours)')
    ax2.set_title('Distribution of Cumulative Lead Time\n(Box Plot)')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add overall statistics text box
    total_los = log.groupby('case:concept:name')['time_from_start_hours'].max()
    stats_text = f"Overall Case Statistics:\n"
    stats_text += f"Mean LOS: {total_los.mean():.2f}h\n"
    stats_text += f"Median LOS: {total_los.median():.2f}h\n"
    stats_text += f"N Cases: {len(total_los)}"
    
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('04_cumulative_lead_time.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print("CUMULATIVE LEAD TIME SUMMARY (Hours)")
    print("="*70)
    print(f"\n{'Activity':<30} {'Mean':>10} {'Median':>10} {'Std':>10} {'N':>8}")
    print("-"*70)
    for act in activities:
        stats = activity_stats.loc[act]
        print(f"{act:<30} {stats['mean_hours']:>10.2f} {stats['median_hours']:>10.2f} "
              f"{stats['std_hours']:>10.2f} {int(stats['count']):>8}")
    
    return fig


# =============================================================================
# FUNCTION 2: DOMAIN-SPECIFIC KPIs (Door-to-Triage, Door-to-Treatment, etc.)
# =============================================================================
def plot_domain_specific_kpis(log: pd.DataFrame):
    """
    Plot comprehensive visualization of domain-specific ED KPIs.
    
    KPIs calculated:
    1. Door-to-Triage (DTT) - time from entry to triage
    2. Door-to-Treatment (DTTreat) - time from entry to first treatment
    3. Triage-to-Treatment (TTT) - time from triage to first treatment
    4. Treatment Duration - time from first treatment to discharge
    
    All times displayed in HOURS.
    
    Parameters:
    -----------
    log : pd.DataFrame
        Event log with columns: case:concept:name, concept:name, time:timestamp
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Ensure timestamp is datetime
    log = log.copy()
    if not pd.api.types.is_datetime64_any_dtype(log['time:timestamp']):
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    
    TREATMENT_ACTIVITIES = ['Medicine reconciliation', 'Medicine dispensations']
    
    # Calculate KPIs for each case
    kpi_data = []
    
    for case_id, case_df in log.groupby('case:concept:name'):
        case_df = case_df.sort_values('time:timestamp')
        first_occurrences = case_df.groupby('concept:name')['time:timestamp'].min()
        
        entry_time = first_occurrences.get('Enter the ED')
        triage_time = first_occurrences.get('Triage in the ED')
        discharge_time = first_occurrences.get('Discharge from the ED')
        
        treatment_times = [first_occurrences.get(act) for act in TREATMENT_ACTIVITIES 
                          if act in first_occurrences.index]
        first_treatment_time = min(treatment_times) if treatment_times else None
        
        kpis = {'case_id': case_id}
        
        # Door-to-Triage (hours)
        if entry_time and triage_time:
            kpis['door_to_triage'] = (triage_time - entry_time).total_seconds() / 3600
        else:
            kpis['door_to_triage'] = None
            
        # Door-to-Treatment (hours)
        if entry_time and first_treatment_time:
            kpis['door_to_treatment'] = (first_treatment_time - entry_time).total_seconds() / 3600
        else:
            kpis['door_to_treatment'] = None
            
        # Triage-to-Treatment (hours)
        if triage_time and first_treatment_time:
            kpis['triage_to_treatment'] = (first_treatment_time - triage_time).total_seconds() / 3600
        else:
            kpis['triage_to_treatment'] = None
            
        # Treatment Duration (hours)
        if first_treatment_time and discharge_time:
            kpis['treatment_duration'] = (discharge_time - first_treatment_time).total_seconds() / 3600
        else:
            kpis['treatment_duration'] = None
            
        # Length of Stay (hours)
        if entry_time and discharge_time:
            kpis['los'] = (discharge_time - entry_time).total_seconds() / 3600
        else:
            kpis['los'] = None
        
        kpi_data.append(kpis)
    
    kpi_df = pd.DataFrame(kpi_data)
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    kpi_names = ['door_to_triage', 'door_to_treatment', 'triage_to_treatment', 'treatment_duration']
    kpi_labels = ['Door-to-Triage\n(Entry → Triage)', 
                  'Door-to-Treatment\n(Entry → First Treatment)',
                  'Triage-to-Treatment\n(Triage → First Treatment)', 
                  'Treatment Duration\n(First Treatment → Discharge)']
    kpi_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    
    # ===== Plots 1-4: Individual KPI Histograms =====
    for idx, (kpi_name, kpi_label, color) in enumerate(zip(kpi_names, kpi_labels, kpi_colors)):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        data = kpi_df[kpi_name].dropna()
        
        if len(data) > 0:
            # Histogram
            n, bins, patches = ax.hist(data, bins=40, color=color, alpha=0.7, edgecolor='white')
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            
            ax.axvline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.2f}h')
            ax.axvline(median_val, color='black', linestyle='-', linewidth=2, 
                      label=f'Median: {median_val:.2f}h')
            
            # Add statistics text
            stats_text = f"N = {len(data)}\n"
            stats_text += f"Std = {data.std():.2f}h\n"
            stats_text += f"Min = {data.min():.2f}h\n"
            stats_text += f"Max = {data.max():.2f}h\n"
            stats_text += f"IQR = [{data.quantile(0.25):.2f}, {data.quantile(0.75):.2f}]h"
            
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Time (Hours)')
            ax.set_ylabel('Number of Cases')
            ax.set_title(kpi_label, fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(kpi_label)
    
    # ===== Plot 5: Box Plot Comparison =====
    ax5 = axes[0, 2]
    
    box_data = [kpi_df[kpi].dropna().values for kpi in kpi_names]
    box_labels = ['DTT', 'DTTreat', 'TTT', 'Treat\nDuration']
    
    bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], kpi_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points (jittered)
    for i, data in enumerate(box_data):
        if len(data) > 0:
            x = np.random.normal(i + 1, 0.04, size=min(len(data), 100))
            sample_data = np.random.choice(data, size=min(len(data), 100), replace=False)
            ax5.scatter(x, sample_data, alpha=0.3, s=10, color=kpi_colors[i])
    
    ax5.set_ylabel('Time (Hours)')
    ax5.set_title('KPI Comparison\n(Box Plots with Data Points)', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # ===== Plot 6: Stacked Timeline View =====
    ax6 = axes[1, 2]
    
    # Calculate mean values for timeline
    mean_dtt = kpi_df['door_to_triage'].dropna().mean()
    mean_ttt = kpi_df['triage_to_treatment'].dropna().mean()
    mean_dur = kpi_df['treatment_duration'].dropna().mean()
    mean_los = kpi_df['los'].dropna().mean()
    
    # Create stacked bar showing typical case progression
    phases = ['Door-to-Triage', 'Triage-to-Treatment', 'Treatment Duration']
    phase_durations = [mean_dtt if not np.isnan(mean_dtt) else 0, 
                       mean_ttt if not np.isnan(mean_ttt) else 0, 
                       mean_dur if not np.isnan(mean_dur) else 0]
    phase_colors = [COLORS['primary'], COLORS['accent'], COLORS['success']]
    
    # Horizontal stacked bar
    left = 0
    for phase, duration, color in zip(phases, phase_durations, phase_colors):
        ax6.barh(0, duration, left=left, height=0.5, color=color, 
                label=f'{phase}: {duration:.2f}h', edgecolor='white', linewidth=2)
        if duration > 0.1:  # Only label if visible
            ax6.text(left + duration/2, 0, f'{duration:.2f}h', 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        left += duration
    
    # Add total LOS marker
    ax6.axvline(mean_los, color=COLORS['danger'], linestyle='--', linewidth=2, 
               label=f'Mean LOS: {mean_los:.2f}h')
    
    ax6.set_xlim(0, max(left, mean_los) * 1.1)
    ax6.set_ylim(-0.5, 0.5)
    ax6.set_yticks([])
    ax6.set_xlabel('Time (Hours)')
    ax6.set_title('Typical Case Timeline\n(Mean Duration per Phase)', fontsize=11, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(axis='x', alpha=0.3)
    
    # Add process flow annotation
    ax6.annotate('Entry', xy=(0, -0.35), fontsize=10, ha='center')
    ax6.annotate('Triage', xy=(mean_dtt, -0.35), fontsize=10, ha='center')
    ax6.annotate('Treatment', xy=(mean_dtt + mean_ttt, -0.35), fontsize=10, ha='center')
    ax6.annotate('Discharge', xy=(left, -0.35), fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('04_domain_specific_kpis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("DOMAIN-SPECIFIC KPIs SUMMARY (Hours)")
    print("="*70)
    print(f"\n{'KPI':<30} {'Mean':>10} {'Median':>10} {'Std':>10} {'N':>8}")
    print("-"*70)
    for kpi_name, kpi_label in zip(kpi_names, ['Door-to-Triage', 'Door-to-Treatment', 
                                                'Triage-to-Treatment', 'Treatment Duration']):
        data = kpi_df[kpi_name].dropna()
        if len(data) > 0:
            print(f"{kpi_label:<30} {data.mean():>10.2f} {data.median():>10.2f} "
                  f"{data.std():>10.2f} {len(data):>8}")
    
    return fig


# =============================================================================
# FUNCTION 3: NURSE WORKLOAD DISTRIBUTION
# =============================================================================
def plot_nurse_workload(log: pd.DataFrame):
    """
    Plot nurse workload distribution by job type.
    
    Shows distinct nurse IDs on y-axis with their respective job type
    (Medicine Reconciliation vs Administration) and workload (events handled).
    
    Parameters:
    -----------
    log : pd.DataFrame
        Event log with columns: reconciliation_nurse_id, administering_nurse_id
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Prepare data for reconciliation nurses
    recon_data = log[log['reconciliation_nurse_id'].notna()].copy()
    recon_stats = recon_data.groupby('reconciliation_nurse_id').agg({
        'case:concept:name': ['count', 'nunique']
    })
    recon_stats.columns = ['events_handled', 'cases_handled']
    recon_stats['job_type'] = 'Medicine Reconciliation'
    recon_stats['nurse_id'] = recon_stats.index.astype(int).astype(str)
    recon_stats['nurse_label'] = 'Recon_' + recon_stats['nurse_id']
    
    # Prepare data for administering nurses
    admin_data = log[log['administering_nurse_id'].notna()].copy()
    admin_stats = admin_data.groupby('administering_nurse_id').agg({
        'case:concept:name': ['count', 'nunique']
    })
    admin_stats.columns = ['events_handled', 'cases_handled']
    admin_stats['job_type'] = 'Medicine Administration'
    admin_stats['nurse_id'] = admin_stats.index.astype(int).astype(str)
    admin_stats['nurse_label'] = 'Admin_' + admin_stats['nurse_id']
    
    # Combine data
    all_nurses = pd.concat([recon_stats.reset_index(drop=True), 
                            admin_stats.reset_index(drop=True)])
    all_nurses = all_nurses.sort_values('events_handled', ascending=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== Plot 1: All Nurses Bar Chart =====
    ax1 = axes[0, 0]
    
    colors = [COLORS['primary'] if jt == 'Medicine Reconciliation' else COLORS['secondary'] 
              for jt in all_nurses['job_type']]
    
    y_pos = np.arange(len(all_nurses))
    bars = ax1.barh(y_pos, all_nurses['events_handled'], color=colors, alpha=0.8, edgecolor='white')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(all_nurses['nurse_label'], fontsize=8)
    ax1.set_xlabel('Events Handled')
    ax1.set_title('Workload by Nurse ID\n(Sorted by Events Handled)', fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['primary'], label='Medicine Reconciliation'),
                       Patch(facecolor=COLORS['secondary'], label='Medicine Administration')]
    ax1.legend(handles=legend_elements, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # ===== Plot 2: Grouped Bar by Job Type =====
    ax2 = axes[0, 1]
    
    # Reconciliation nurses
    recon_sorted = recon_stats.sort_values('events_handled', ascending=False)
    admin_sorted = admin_stats.sort_values('events_handled', ascending=False)
    
    # Top 15 of each type
    n_show = min(15, len(admin_sorted))
    
    x = np.arange(n_show)
    width = 0.35
    
    admin_top = admin_sorted.head(n_show)
    
    bars1 = ax2.bar(x, admin_top['events_handled'], width, 
                    label='Events Handled', color=COLORS['secondary'], alpha=0.8)
    bars2 = ax2.bar(x + width, admin_top['cases_handled'], width,
                    label='Cases Handled', color=COLORS['primary'], alpha=0.8)
    
    ax2.set_xlabel('Nurse ID (Administration)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Top {n_show} Administration Nurses\n(Events vs Cases)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels([f'ID {int(i)}' for i in admin_top['nurse_id']], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # ===== Plot 3: Workload Distribution Comparison =====
    ax3 = axes[1, 0]
    
    # Violin plots for each job type
    data_to_plot = [recon_stats['events_handled'].values, admin_stats['events_handled'].values]
    
    parts = ax3.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['primary'], COLORS['secondary']][i])
        pc.set_alpha(0.7)
    
    # Add individual points
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(i + 1, 0.05, size=len(data))
        ax3.scatter(x, data, alpha=0.6, s=50, 
                   color=[COLORS['primary'], COLORS['secondary']][i],
                   edgecolors='white', linewidths=0.5)
    
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Medicine\nReconciliation\n(N=3 nurses)', 
                         'Medicine\nAdministration\n(N=41 nurses)'])
    ax3.set_ylabel('Events Handled per Nurse')
    ax3.set_title('Workload Distribution by Job Type\n(Violin Plot with Individual Points)', 
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add statistics
    for i, (data, label) in enumerate(zip(data_to_plot, ['Reconciliation', 'Administration'])):
        stats_text = f"Mean: {np.mean(data):.0f}\nMedian: {np.median(data):.0f}\nStd: {np.std(data):.0f}"
        ax3.text(i + 1, max(data) * 1.05, stats_text, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ===== Plot 4: Events per Case Analysis =====
    ax4 = axes[1, 1]
    
    # Calculate events per case for each nurse
    all_nurses['events_per_case'] = all_nurses['events_handled'] / all_nurses['cases_handled']
    
    # Scatter plot: cases handled vs events per case
    recon_mask = all_nurses['job_type'] == 'Medicine Reconciliation'
    admin_mask = all_nurses['job_type'] == 'Medicine Administration'
    
    ax4.scatter(all_nurses.loc[recon_mask, 'cases_handled'], 
               all_nurses.loc[recon_mask, 'events_per_case'],
               s=all_nurses.loc[recon_mask, 'events_handled'] / 10,
               c=COLORS['primary'], alpha=0.7, label='Reconciliation', edgecolors='white')
    
    ax4.scatter(all_nurses.loc[admin_mask, 'cases_handled'], 
               all_nurses.loc[admin_mask, 'events_per_case'],
               s=all_nurses.loc[admin_mask, 'events_handled'] / 10,
               c=COLORS['secondary'], alpha=0.7, label='Administration', edgecolors='white')
    
    # Add nurse labels for reconciliation nurses (there are only 3)
    for _, row in all_nurses[recon_mask].iterrows():
        ax4.annotate(row['nurse_label'], (row['cases_handled'], row['events_per_case']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Cases Handled')
    ax4.set_ylabel('Events per Case')
    ax4.set_title('Nurse Efficiency Analysis\n(Size = Total Events Handled)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_nurse_workload.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("NURSE WORKLOAD SUMMARY")
    print("="*80)
    
    print("\n  MEDICINE RECONCILIATION NURSES:")
    print("  " + "-"*60)
    print(f"  {'Nurse ID':<15} {'Events':>12} {'Cases':>12} {'Events/Case':>15}")
    for _, row in recon_stats.sort_values('events_handled', ascending=False).iterrows():
        print(f"  {row['nurse_label']:<15} {int(row['events_handled']):>12} "
              f"{int(row['cases_handled']):>12} {row['events_handled']/row['cases_handled']:>15.2f}")
    
    print("\n  MEDICINE ADMINISTRATION NURSES (Top 10):")
    print("  " + "-"*60)
    print(f"  {'Nurse ID':<15} {'Events':>12} {'Cases':>12} {'Events/Case':>15}")
    for _, row in admin_stats.sort_values('events_handled', ascending=False).head(10).iterrows():
        print(f"  {row['nurse_label']:<15} {int(row['events_handled']):>12} "
              f"{int(row['cases_handled']):>12} {row['events_handled']/row['cases_handled']:>15.2f}")
    
    print(f"\n  Total Administration Nurses: {len(admin_stats)}")
    print(f"  Total Reconciliation Nurses: {len(recon_stats)}")
    
    return fig


# =============================================================================
# FUNCTION 4: LENGTH OF STAY BY ACUITY LEVEL
# =============================================================================
def plot_los_by_acuity(log: pd.DataFrame):
    """
    Plot Length of Stay (hours) against case acuity level.
    
    Acuity Levels:
    - Level 1: Most Urgent (Resuscitation)
    - Level 2: Emergent
    - Level 3: Urgent
    - Level 4: Less Urgent
    - Level 5: Non-Urgent
    
    Parameters:
    -----------
    log : pd.DataFrame
        Event log with columns: case:concept:name, time:timestamp, case:acuity
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Ensure timestamp is datetime
    log = log.copy()
    if not pd.api.types.is_datetime64_any_dtype(log['time:timestamp']):
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    
    # Calculate LOS for each case
    case_data = log.groupby('case:concept:name').agg({
        'time:timestamp': ['min', 'max'],
        'case:acuity': 'first'
    })
    case_data.columns = ['start_time', 'end_time', 'acuity']
    case_data['los_hours'] = (case_data['end_time'] - case_data['start_time']).dt.total_seconds() / 3600
    case_data = case_data.dropna(subset=['acuity'])
    case_data['acuity'] = case_data['acuity'].astype(int)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ===== Plot 1: Box Plot by Acuity =====
    ax1 = axes[0, 0]
    
    acuity_levels = sorted(case_data['acuity'].unique())
    box_data = [case_data[case_data['acuity'] == level]['los_hours'].values 
                for level in acuity_levels]
    
    bp = ax1.boxplot(box_data, labels=[f'Level {i}' for i in acuity_levels], 
                     patch_artist=True)
    
    for i, (patch, level) in enumerate(zip(bp['boxes'], acuity_levels)):
        patch.set_facecolor(ACUITY_COLORS.get(level, COLORS['neutral']))
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Acuity Level')
    ax1.set_ylabel('Length of Stay (Hours)')
    ax1.set_title('Length of Stay Distribution by Acuity Level\n(Box Plot)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add sample sizes
    for i, level in enumerate(acuity_levels):
        n = len(case_data[case_data['acuity'] == level])
        ax1.text(i + 1, ax1.get_ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=9)
    
    # ===== Plot 2: Violin Plot by Acuity =====
    ax2 = axes[0, 1]
    
    parts = ax2.violinplot(box_data, positions=acuity_levels, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(ACUITY_COLORS.get(acuity_levels[i], COLORS['neutral']))
        pc.set_alpha(0.7)
    
    ax2.set_xlabel('Acuity Level')
    ax2.set_ylabel('Length of Stay (Hours)')
    ax2.set_title('Length of Stay Distribution by Acuity Level\n(Violin Plot)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(acuity_levels)
    ax2.set_xticklabels([f'Level {i}' for i in acuity_levels])
    ax2.grid(axis='y', alpha=0.3)
    
    # ===== Plot 3: Mean and Median Comparison =====
    ax3 = axes[1, 0]
    
    stats_by_acuity = case_data.groupby('acuity')['los_hours'].agg(['mean', 'median', 'std', 'count'])
    
    x = np.arange(len(acuity_levels))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, stats_by_acuity['mean'], width, 
                    yerr=stats_by_acuity['std'], capsize=5,
                    label='Mean ± Std', color=COLORS['primary'], alpha=0.8)
    bars2 = ax3.bar(x + width/2, stats_by_acuity['median'], width,
                    label='Median', color=COLORS['secondary'], alpha=0.8)
    
    ax3.set_xlabel('Acuity Level')
    ax3.set_ylabel('Length of Stay (Hours)')
    ax3.set_title('Mean vs Median LOS by Acuity Level', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Level {i}\n(n={int(stats_by_acuity.loc[i, "count"])})' 
                         for i in acuity_levels])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}h', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}h', ha='center', va='bottom', fontsize=9)
    
    # ===== Plot 4: Scatter with Trend Line =====
    ax4 = axes[1, 1]
    
    # Add jitter to x-axis for better visualization
    jitter = np.random.uniform(-0.2, 0.2, len(case_data))
    
    for level in acuity_levels:
        mask = case_data['acuity'] == level
        x_jittered = case_data.loc[mask, 'acuity'] + jitter[mask]
        ax4.scatter(x_jittered, case_data.loc[mask, 'los_hours'],
                   c=ACUITY_COLORS.get(level, COLORS['neutral']),
                   alpha=0.5, s=30, label=f'Level {level}')
    
    # Add trend line (polynomial fit)
    z = np.polyfit(case_data['acuity'], case_data['los_hours'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(acuity_levels), max(acuity_levels), 100)
    ax4.plot(x_smooth, p(x_smooth), 'k--', linewidth=2, label='Trend (quadratic)')
    
    # Add mean line
    means = stats_by_acuity['mean']
    ax4.plot(acuity_levels, means, 'ro-', linewidth=2, markersize=10, label='Mean LOS')
    
    ax4.set_xlabel('Acuity Level')
    ax4.set_ylabel('Length of Stay (Hours)')
    ax4.set_title('LOS vs Acuity Level\n(Scatter with Trend)', fontsize=12, fontweight='bold')
    ax4.set_xticks(acuity_levels)
    ax4.set_xticklabels([f'Level {i}' for i in acuity_levels])
    ax4.legend(loc='upper right')
    ax4.grid(alpha=0.3)
    
    # Add acuity interpretation
    interpretation = ("← Most Urgent" + " "*20 + "Least Urgent →")
    ax4.text(0.5, -0.12, interpretation, transform=ax4.transAxes, 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('04_los_by_acuity.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Statistical test
    print("\n" + "="*70)
    print("LENGTH OF STAY BY ACUITY LEVEL - STATISTICAL ANALYSIS")
    print("="*70)
    
    print("\n  DESCRIPTIVE STATISTICS (Hours):")
    print("  " + "-"*60)
    print(f"  {'Acuity':<15} {'N':>8} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print("  " + "-"*60)
    for level in acuity_levels:
        data = case_data[case_data['acuity'] == level]['los_hours']
        print(f"  Level {level:<9} {len(data):>8} {data.mean():>10.2f} "
              f"{data.median():>10.2f} {data.std():>10.2f}")
    
    # Kruskal-Wallis test (non-parametric)
    groups = [case_data[case_data['acuity'] == level]['los_hours'].values 
              for level in acuity_levels if len(case_data[case_data['acuity'] == level]) > 0]
    
    if len(groups) > 1:
        h_stat, p_value = stats.kruskal(*groups)
        print(f"\n  KRUSKAL-WALLIS TEST:")
        print(f"    H-statistic: {h_stat:.2f}")
        print(f"    p-value: {p_value:.2e}")
        if p_value < 0.05:
            print("    → Significant difference in LOS across acuity levels (p < 0.05)")
        else:
            print("    → No significant difference in LOS across acuity levels (p ≥ 0.05)")
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv("dataset_for_exam.csv")
    
    # Rename columns to standard format
    log = df.rename(columns={
        "stay_id": "case:concept:name",
        "activity": "concept:name",
        "time": "time:timestamp",
        "acuity": "case:acuity",
        "arrival_transport": "case:arrival_transport",
        "disposition": "case:disposition",
        "gender": "case:gender"
    })
    
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    log['case:concept:name'] = log['case:concept:name'].astype(str)
    
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)
    
    # Run all visualization functions
    print("\n>>> Function 1: Cumulative Lead Time")
    plot_cumulative_lead_time(log)
    
    print("\n>>> Function 2: Domain-Specific KPIs")
    plot_domain_specific_kpis(log)
    
    print("\n>>> Function 3: Nurse Workload")
    plot_nurse_workload(log)
    
    print("\n>>> Function 4: LOS by Acuity")
    plot_los_by_acuity(log)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nOutput files generated:")
    print("  - 04_cumulative_lead_time.png")
    print("  - 04_domain_specific_kpis.png")
    print("  - 04_nurse_workload.png")
    print("  - 04_los_by_acuity.png")
