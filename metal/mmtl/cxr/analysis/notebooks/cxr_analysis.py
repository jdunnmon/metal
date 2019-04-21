import os
import json
import numpy as np
import pandas as pd

def load_log_json(log_json):
    """
    Loads MMTL json-formatted log file
    """
    with open(log_json, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    return data

def load_results_from_log(log_dir):
    """
    Load all json logs from MMTL log dict
    """
    results = {}
    json_files = [a for a in os.listdir(log_dir) if a.endswith('json')]
    for fl in json_files:
        path = os.path.join(log_dir,fl)
        fl_str = fl.split('.')[0]
        results[fl_str] = load_log_json(path)
    return results

def get_task_name(nm):
    return '_'.join(nm.split('_')[1:]).split(":")[0]

def get_labelset_name(nm):
    return '_'.join(nm.split('_')[1:])


def get_cxr14_rocs_from_log(chexnet_results, metrics_dict, col_name = 'experiment', plot_metric='roc-auc', load_slices=True,
                            head=None):
    # Use existing results if already there
    if col_name in chexnet_results.columns:
        output_dict = dict(zip(chexnet_results[col_name].index,chexnet_results[col_name].values))
    else:
        output_dict = {}
        
    for ky, val in metrics_dict.items():
            
        # Current format: task, split, labelset, metric
        task, split, labelset, metric = ky.split('/')

        # Current task format: DATASET_TASKNAME
        task_name = get_task_name(task)
        labelset_name = get_labelset_name(labelset)
        if head is not None:
            labelset_name = f"{labelset_name}:{head}"
        
        # Checking if this is a valid result for comparison
        if (task_name in labelset_name) and (task_name.split(":")[0].upper() in chexnet_results.index) and (metric == plot_metric):
            output_dict[labelset_name.upper()] = val
        
        if (labelset_name.upper() not in chexnet_results.index) and load_slices:
            cols = chexnet_results.columns.tolist()
            ns =  pd.Series([np.nan]*len(cols), index=cols, name=labelset_name.upper())
            chexnet_results = chexnet_results.append(ns)
            
    # Adding to chexnet results
    chexnet_results[col_name] = chexnet_results.index.map(output_dict)
            
    return chexnet_results 