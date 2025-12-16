import pandas as pd 


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