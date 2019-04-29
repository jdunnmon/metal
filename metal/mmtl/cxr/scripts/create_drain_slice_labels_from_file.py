import pandas as pd
import argparse
import os
import copy
import logging
import numpy as np
import os

# Configuring logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_reproduce_chexnet_data(dr):
    dfs = {}
    for split in ["train", "dev", "test"]:
        dfs[split] = pd.read_csv(os.path.join(dr,f"{split}.tsv"), sep='\t')
    return dfs

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcdwd_dir", "-rcd", help="Directory containing modified NIH data (with drains) from reproduce-chexnet", required=True)
    parser.add_argument("--drain_file", "-dd", help="File containing list of confirmed chest drains", required=True)
    parser.add_argument("--out_dir", "-od", help="Output directory for drain slice data")
    parser.add_argument("--positive", "-p", type=int, help="Positive (1) or negative (0)", required=True) 
    parser.add_argument("--split", "-s", help="Split to operate on", required=True)
    parser.add_argument("--only_slice", "-os", help="if True, only labels from slice; else, labels from slice plus all other classes as negative", type=int, required=True)
    args = parser.parse_args()
    
    # Setting seed
    SEED = 1701

    # Getting split
    split = args.split

    # Reading in NIH data
    dfs = read_reproduce_chexnet_data(args.rcdwd_dir)
    
    # Reading in pneumo subset from LOR
    pneumo_subset = pd.read_csv(args.drain_file)

    # Identifying drains
    drains = pneumo_subset[pneumo_subset['drain']==1]
    no_drains = pneumo_subset[pneumo_subset['drain']==0]   

    # Getting pneumos from dev set
    df_split = copy.deepcopy(dfs[split])
    split_pneumos = df_split[df_split["Pneumothorax"]==1]
    split_no_pneumos = df_split[df_split["Pneumothorax"]==0]    

    # Getting drains on split
    split_drains = split_pneumos[split_pneumos["Image Index"].isin(drains["Image Index"])]["Image Index"]
    split_no_drains = split_pneumos[split_pneumos["Image Index"].isin(no_drains["Image Index"])]["Image Index"]

    if args.positive:
        df_split['drains'] = df_split["Image Index"].isin(split_drains).astype(int)
    else:
        df_split['drains'] =  df_split["Image Index"].isin(split_no_drains).astype(int)
        
    if not args.only_slice:
        df_split['drains'][df_split["Image Index"].isin(split_no_pneumos["Image Index"])] = 1

    # Creating dataframe
    slice_df = pd.DataFrame(
                    {'data_index' : list(df_split["Image Index"]),
                     'slice_label' : list(df_split["drains"])
                        })


    slice_df.to_csv(f"{args.out_dir}/{split}_gt_os_{args.only_slice}.tsv",sep='\t',index=False)

    logger.info(f"{np.sum(slice_df['slice_label'])} pos={args.positive} labels in {split} split")
    logger.info(f"Slice data saved for {split} split")
