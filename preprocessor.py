import pandas as pd 
import json


def standardize_dataframe_for_pm4py (log):
    log = log.rename(columns={
    "stay_id": "case:concept:name",
    "activity": "concept:name",
    "time": "time:timestamp",
    "arrival_transport": "case:arrival_transport",
    "disposition" : 'case:disposition',
    "gender": "case:gender",
    "race": "case:race",
    "acuity": "case:acuity",
    "chiefcomplaint": "case:chiefcomplaint" })
    
    # convert the 'Date' column to datetime format
    log['time:timestamp']= pd.to_datetime(log['time:timestamp'])
    
    # The case ID column should be of type string
    log['case:concept:name'] = log['case:concept:name'].astype(str)

    return log

"""
Aggregates concurrent events while preserving true parallel activities.
"""


def aggregate_event_log(df_raw):
    """
    Aggregates event log by combining concurrent events of the same type by the same nurse.
    
    Aggregation Rules:
    ------------------
    1. Medicine reconciliation: Aggregates only when reconciliation_nurse_id is the same
       - Different nurses at same time = TRUE PARALLEL activities (preserved as separate events)
       - Same nurse at same time = DUPLICATED data (aggregated into one event)
    
    2. Medicine dispensations: Aggregates only when administering_nurse_id is the same
       - Different nurses at same time = TRUE PARALLEL activities (preserved as separate events)
       - Same nurse at same time = DUPLICATED data (aggregated into one event)
    
    3. Discharge from ED: Aggregates all diagnosis codes (no nurse distinction)
    
    4. Other activities: No aggregation (typically single events)
    
    """
    
    # Ensure time is datetime
    df = df_raw.copy()
    if df['time:timestamp'].dtype != 'datetime64[ns]':
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    
    # Sort by case and time
    df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    # Define grouping keys and columns to aggregate for each activity type
    aggregation_rules = {
        'Medicine reconciliation': {
            'group_by': ['case:concept:name', 'time:timestamp', 'concept:name', 'reconciliation_nurse_id'],
            'aggregate_cols': ['drug_name', 'generic_drug_code', 'national_drug_code', 
                              'drug_class_code', 'drug_class_classification']
        },
        'Medicine dispensations': {
            'group_by': ['case:concept:name', 'time:timestamp', 'concept:name', 'administering_nurse_id'],
            'aggregate_cols': ['drug_name', 'generic_drug_code']
        },
        'Discharge from the ED': {
            'group_by': ['case:concept:name', 'time:timestamp', 'concept:name'],
            'aggregate_cols': ['diagnosis_sequence', 'diagnosis_code', 'diagnosis_description']
        }
    }
    
    # Process each activity type
    aggregated_dfs = []
    
    for activity_name, rules in aggregation_rules.items():
        activity_df = df[df['concept:name'] == activity_name].copy()
        
        if len(activity_df) == 0:
            continue
            
        group_by_cols = rules['group_by']
        cols_to_aggregate = rules['aggregate_cols']
        
        # Group and aggregate
        grouped = activity_df.groupby(group_by_cols, dropna=False)
        
        for group_key, group_data in grouped:
            if len(group_data) == 1:
                # No aggregation needed - single event
                row = group_data.iloc[0].to_dict()
                row['aggregated_event_count'] = 1
                row['is_aggregated'] = False
                aggregated_dfs.append(pd.DataFrame([row]))
            else:
                # Multiple events - aggregate them
                base_row = group_data.iloc[0].to_dict()
                
                # Aggregate specified columns
                for col in cols_to_aggregate:
                    if col in group_data.columns:
                        # Collect all non-null unique values
                        values = group_data[col].dropna().astype(str).tolist()
                        if len(values) > 0:
                            # Store as JSON array if multiple values, otherwise single value
                            if len(values) > 1:
                                base_row[col] = json.dumps(values)
                            else:
                                base_row[col] = values[0]
                            base_row[f'{col}_count'] = len(values)
                
                # Add aggregation metadata
                base_row['aggregated_event_count'] = len(group_data)
                base_row['is_aggregated'] = True
                
                aggregated_dfs.append(pd.DataFrame([base_row]))
    
    # Handle other activities (no aggregation needed, but group by case:concept:name, time, activity just in case)
    other_activities = ['Enter the ED', 'Triage in the ED', 'Vital sign check']
    for activity_name in other_activities:
        activity_df = df[df['concept:name'] == activity_name].copy()
        
        if len(activity_df) == 0:
            continue
        
        # Group by case:concept:name, time, activity (unlikely to have duplicates, but handle if present)
        grouped = activity_df.groupby(['case:concept:name', 'time:timestamp', 'concept:name'], dropna=False)
        
        for group_key, group_data in grouped:
            if len(group_data) == 1:
                row = group_data.iloc[0].to_dict()
                row['aggregated_event_count'] = 1
                row['is_aggregated'] = False
                aggregated_dfs.append(pd.DataFrame([row]))
            else:
                # Rare case - just take the first one and mark as aggregated
                row = group_data.iloc[0].to_dict()
                row['aggregated_event_count'] = len(group_data)
                row['is_aggregated'] = True
                aggregated_dfs.append(pd.DataFrame([row]))
    
    # Combine all aggregated data
    df_aggregated = pd.concat(aggregated_dfs, ignore_index=True)
    
    # Sort by case and time to maintain chronological order
    df_aggregated = df_aggregated.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    # Ensure aggregation metadata columns exist
    if 'aggregated_event_count' not in df_aggregated.columns:
        df_aggregated['aggregated_event_count'] = 1
    if 'is_aggregated' not in df_aggregated.columns:
        df_aggregated['is_aggregated'] = False
    
    df_aggregated['aggregated_event_count'] = df_aggregated['aggregated_event_count'].fillna(1).astype(int)
    df_aggregated['is_aggregated'] = df_aggregated['is_aggregated'].fillna(False).astype(bool)
    
    return df_aggregated