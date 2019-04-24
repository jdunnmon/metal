import os
import pandas as pd
import torch
from torch.utils.data import Dataset

def chest_drain_cnn_neg(dataset: Dataset) -> dict:
    #data_file = os.path.join(os.environ["CXRDATA"],'CXR8-DRAIN-SLICE-NEG',f"{dataset.split}.tsv")
    data_file = os.path.join(os.environ["CXRDATA"],'CXR8-DRAIN-SLICE-NEG-CNN-F1',f"{dataset.split}.tsv")
    slice_data = pd.read_csv(data_file, sep='\t')
    keys = slice_data['data_index'].tolist()
    values = [int(l) for l in slice_data['slice_label'].astype(int)]
    slice_dict = dict(zip(keys, values))
    return slice_dict

def create_slice_labels(dataset, base_task_name, slice_name, verbose=False):
    """Returns a label set masked to include only those labels in the specified slice"""
    # TODO: break this out into more modular pieces one we have multiple slices
    # Uses typed function annotatinos to figure out which way to evaluate the slice
    slice_fn = globals()[slice_name]
    return_type = slice_fn.__annotations__['return']

    # if we pre-load data, use a dict + uids
    if return_type is dict:
        slice_ind_dict = slice_fn(dataset)
        slice_indicators = [slice_ind_dict[dataset.df.index[idx]] for idx in range(len(dataset))]
        
    # if we evaluate at runtime, use index
    else:
        slice_indicators = [slice_fn(dataset, idx) for idx in range(len(dataset))]
 
    base_labels = dataset.labels[base_task_name]
    slice_labels = [
        label * indicator for label, indicator in zip(base_labels, slice_indicators)
    ]
    if verbose:
        print(f"Found {sum(slice_indicators)} examples in slice {slice_name}.")
        if not any(slice_labels):
            warnings.warn("No examples were found to belong to ")

    return slice_labels
