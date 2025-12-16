import pandas as pd
import pm4py
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import timedelta
import warnings
import numpy as np

# Visualization configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01', 
          'success': '#C73E1D', 'neutral': '#3B1F2B'}

def validate_domain_order(log):
    """
    Check for violations of expected clinical sequences in the ED process.

    Domain knowledge defines expected precedence relationships:
    - Patient must enter before any other activity
    - Triage should precede treatment activities
    - Discharge should be the final activity

    Parameters:
    -----------
    log : pd.DataFrame
        Event log

    Returns:
    --------
    tuple
        (violations_df, start_activities, end_activities)
    """
    print("\n" + "="*80)
    print("STEP 4: DOMAIN-BASED ORDER VALIDATION")
    print("="*80)
    print("""
    Validating expected clinical sequences based on ED domain knowledge...

    Expected precedence rules:
    1. 'Enter the ED' → should be FIRST activity in every case
    2. 'Enter the ED' → 'Triage in the ED' (entry before triage)
    3. 'Triage in the ED' → 'Medicine administration' (triage before treatment)
    4. Any activity → 'Discharge from the ED' (discharge should be LAST)
    """)

    # Define expected precedence rules (A should occur before B)
    EXPECTED_PRECEDENCE = [
        ('Enter the ED', 'Triage in the ED'),
        ('Enter the ED', 'Medicine reconciliation'),
        ('Enter the ED', 'Medicine dispensations'),
        ('Enter the ED', 'Vital sign check'),
        ('Enter the ED', 'Discharge from the ED'),
        ('Triage in the ED', 'Discharge from the ED'),
    ]

    violations = []

    for case_id, case_df in log.groupby('case:concept:name'):
        case_df = case_df.sort_values('time:timestamp')
        activities = case_df['concept:name'].tolist()

        # Build first occurrence index for each activity
        activity_first_idx = {}
        for idx, act in enumerate(activities):
            if act not in activity_first_idx:
                activity_first_idx[act] = idx

        # Check each precedence rule
        for act_before, act_after in EXPECTED_PRECEDENCE:
            if act_before in activity_first_idx and act_after in activity_first_idx:
                if activity_first_idx[act_before] > activity_first_idx[act_after]:
                    violations.append({
                        'case_id': case_id,
                        'rule': f"{act_before} → {act_after}",
                        'expected_first': act_before,
                        'expected_second': act_after,
                        'actual_order': f"{act_after} occurred at position {activity_first_idx[act_after]}, "
                                       f"{act_before} at position {activity_first_idx[act_before]}"
                    })

    violations_df = pd.DataFrame(violations)

    # Get start and end activities
    start_activities = pm4py.get_start_activities(log)
    end_activities = pm4py.get_end_activities(log)

    # Report findings
    print(f"\n✓ RESULTS:")
    print(f"  Start activities: {start_activities}")
    print(f"  End activities: {end_activities}")

    if len(violations_df) == 0:
        print("\n  ✓ No domain-based order violations detected!")
        print("    All cases follow the expected clinical sequence.")
    else:
        n_cases = violations_df['case_id'].nunique()
        print(f"\n  ⚠ Found {len(violations_df)} violations in {n_cases} cases")
        print("\n  Violations by rule:")
        print(violations_df.groupby('rule').size().to_string())

    # Validate start/end activities
    print("\n  Process boundary validation:")
    unexpected_starts = {k: v for k, v in start_activities.items() if k != 'Enter the ED'}
    unexpected_ends = {k: v for k, v in end_activities.items() if k != 'Discharge from the ED'}

    if not unexpected_starts:
        print("    ✓ All cases start with 'Enter the ED'")
    else:
        print(f"    ⚠ Unexpected start activities: {unexpected_starts}")

    if not unexpected_ends:
        print("    ✓ All cases end with 'Discharge from the ED'")
    else:
        print(f"    ⚠ Unexpected end activities: {unexpected_ends}")

    return violations_df, start_activities, end_activities


def detect_temporal_anomalies(log):
    """
    Detect backward timestamps within cases (temporal anomalies).

    A temporal anomaly occurs when event N+1 has an earlier timestamp
    than event N within the same case. This violates the fundamental
    assumption that events are recorded in chronological order.

    Parameters:
    -----------
    log : pd.DataFrame
        Event log with case:concept:name and time:timestamp columns

    Returns:
    --------
    pd.DataFrame
        DataFrame containing details of all detected anomalies
    """
    print("\n" + "="*80)
    print("STEP 2: TEMPORAL ANOMALIES DETECTION")
    print("="*80)
    print("""
    Checking for backward timestamps within cases...
    A temporal anomaly = event where timestamp goes backward in time.
    """)

    anomalies = []

    for case_id, case_df in log.groupby('case:concept:name'):
        # Ensure events are in their original recorded order
        case_df = case_df.sort_index()
        timestamps = case_df['time:timestamp'].values
        activities = case_df['concept:name'].values
        indices = case_df.index.values

        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                anomalies.append({
                    'case_id': case_id,
                    'event_position': i,
                    'previous_activity': activities[i-1],
                    'previous_timestamp': pd.Timestamp(timestamps[i-1]),
                    'current_activity': activities[i],
                    'current_timestamp': pd.Timestamp(timestamps[i]),
                    'time_difference': pd.Timedelta(timestamps[i] - timestamps[i-1])
                })

    anomalies_df = pd.DataFrame(anomalies)

    # Report findings
    if len(anomalies_df) == 0:
        print("✓ RESULT: No temporal anomalies detected")
        print("  All timestamps are in chronological order within each case.")
    else:
        n_cases_affected = anomalies_df['case_id'].nunique()
        print(f"⚠ RESULT: Found {len(anomalies_df)} temporal anomalies")
        print(f"  Affecting {n_cases_affected} cases ({100*n_cases_affected/log['case:concept:name'].nunique():.2f}%)")
        print("\n  Sample anomalies:")
        print(anomalies_df.head(10).to_string(index=False))

    return anomalies_df


def detect_concurrent_activities(log):
    """
    Identify events occurring at exactly the same timestamp within a case.

    Concurrent activities are common in healthcare processes where:
    - Multiple medications are recorded at once
    - Batch processing occurs for administrative tasks
    - Multiple diagnoses are assigned simultaneously

    This affects temporal KPI calculations because:
    - Waiting time between concurrent events = 0
    - Traditional sequential assumptions don't hold

    Parameters:
    -----------
    log : pd.DataFrame
        Event log with case:concept:name, concept:name, and time:timestamp

    Returns:
    --------
    tuple
        (concurrent_df, concurrent_cases, concurrent_pairs)
        - concurrent_df: DataFrame with concurrent activity groups
        - concurrent_cases: Set of case IDs with concurrent activities
        - concurrent_pairs: Counter of activity pair co-occurrences
    """
    print("\n" + "="*80)
    print("STEP 3: CONCURRENT/PARALLEL ACTIVITIES DETECTION")
    print("="*80)
    print("""
    Identifying events at exactly the same timestamp within cases...
    Concurrent activities indicate parallel execution or batch recording.
    """)

    concurrent_events = []
    concurrent_cases = set()

    for case_id, case_df in log.groupby('case:concept:name'):
        # Group events by timestamp
        timestamp_groups = case_df.groupby('time:timestamp')

        for ts, group in timestamp_groups:
            if len(group) > 1:  # More than one event at same timestamp
                concurrent_cases.add(case_id)
                activities = group['concept:name'].tolist()
                concurrent_events.append({
                    'case_id': case_id,
                    'timestamp': ts,
                    'num_concurrent': len(group),
                    'activities': activities,
                    'activity_types': list(set(activities))
                })

    concurrent_df = pd.DataFrame(concurrent_events)

    # Analyze concurrent activity pairs
    concurrent_pairs = Counter()
    if len(concurrent_df) > 0:
        for _, row in concurrent_df.iterrows():
            activities = sorted(row['activities'])
            for i in range(len(activities)):
                for j in range(i+1, len(activities)):
                    concurrent_pairs[(activities[i], activities[j])] += 1

    # Calculate statistics
    total_cases = log['case:concept:name'].nunique()
    pct_concurrent = 100 * len(concurrent_cases) / total_cases

    # Report findings
    print(f"\n✓ RESULTS:")
    print(f"  - Cases with concurrent activities: {len(concurrent_cases):,} / {total_cases:,} ({pct_concurrent:.1f}%)")

    if len(concurrent_df) > 0:
        print(f"  - Total concurrent activity groups: {len(concurrent_df):,}")
        print(f"  - Average activities per concurrent group: {concurrent_df['num_concurrent'].mean():.2f}")
        print(f"  - Maximum concurrent activities: {concurrent_df['num_concurrent'].max()}")

        print("\n  Top 10 most common concurrent activity pairs:")
        for (act1, act2), count in concurrent_pairs.most_common(10):
            print(f"    {act1} ‖ {act2}: {count:,} times")

        print("""
    INTERPRETATION:
    - High concurrency suggests batch recording or parallel clinical activities
    - Same-activity concurrency (e.g., multiple Medicine reconciliation) indicates
      multiple instances of the same task (e.g., multiple medications)
    - Cross-activity concurrency suggests truly parallel processes
        """)

    return concurrent_df, concurrent_cases, concurrent_pairs

def create_concurrent_heatmap(concurrent_pairs, log):
    """
    Create a heatmap showing concurrent activity relationships.
    """
    activities = log['concept:name'].unique()
    n_activities = len(activities)

    # Create co-occurrence matrix
    matrix = pd.DataFrame(0, index=activities, columns=activities)
    for (a1, a2), count in concurrent_pairs.items():
        matrix.loc[a1, a2] = count
        matrix.loc[a2, a1] = count

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use log scale for better visualization of varying magnitudes
    matrix_log = np.log1p(matrix)  # log(1+x) to handle zeros

    sns.heatmap(matrix_log, annot=matrix.values, fmt='g', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'log(1 + frequency)'})

    ax.set_title('Concurrent Activity Co-occurrence Matrix\n(Values show actual counts, colors show log scale)')
    ax.set_xlabel('Activity')
    ax.set_ylabel('Activity')

    plt.tight_layout()
    plt.show()


def plot_activity_frequency(log: pd.DataFrame, N: int = 15):
    """
    ## 📊 Top Activity Frequency

    Generates a horizontal bar chart of the top N most frequent activities.

    - **Input:** Event log (pandas DataFrame), N (integer, default 15) for top activities.
    - **Output:** Matplotlib Figure displaying the bar chart.
    """

    # Create a new figure and axes for the plot
    fig, ax = plt.subplots(figsize=(9, min(6, N * 0.5))) # Adjust height based on N

    activity_freq = log['concept:name'].value_counts()
    
    # Select the top N activities
    activity_freq_top = activity_freq.head(N)

    # Plotting the horizontal bars
    bars = ax.barh(range(len(activity_freq_top)), activity_freq_top.values,
            color=COLORS['primary'], edgecolor='white')

    # Setting y-axis labels
    ax.set_yticks(range(len(activity_freq_top)))
    ax.set_yticklabels(activity_freq_top.index, fontsize=10)
    
    # Setting labels and title
    ax.set_xlabel('Frequency (Total Event Count)')
    ax.set_title("Activity Frequency Distribution")
    ax.invert_yaxis() # Display the highest frequency bar at the top

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, activity_freq.values)):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2, 
                f'{val:,}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()