import pandas as pd
import argparse
import os
import copy
import logging

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
    parser.add_argument("--rcd_dir", "-rcd", help="Directory containing NIH data from reproduce-chexnet", required=True)
    parser.add_argument("--drain_dir", "-dd", help="Directory containing list of confirmed chest drains", required=True)
    parser.add_argument("--out_dir", "-od", help="Output directory for adjusted data files")

    args = parser.parse_args()

    # Reading in NIH data
    dfs = read_reproduce_chexnet_data(args.rcd_dir)

    # Reading in pneumo subset from LOR
    pneumo_subset = pd.read_csv(args.drain_dir)

    # Identifying drains
    drains = pneumo_subset[pneumo_subset['drain']==1]
    no_drains = pneumo_subset[pneumo_subset['drain']==0]

    # Getting pneumos from dev set
    df_dev = copy.deepcopy(dfs["dev"])
    dev_pneumos = df_dev[df_dev["Pneumothorax"]==1]
    
    # Swapping half of drain gt pneumos with pneumos from dev set
    drain_swap_data = drains.sample(frac=0.5).append(no_drains.sample(frac=0.5))
    drain_swap_subset = dfs["test"][dfs["test"]["Image Index"].isin(drain_swap_data["Image Index"])]
    dev_swap_subset = dev_pneumos.sample(min(len(dev_pneumos),len(drain_swap_subset)))
    logger.info(f"Number of drain ground truth to be swapped: {len(drain_swap_subset)}")
    logger.info(f"Number of drain unknown to be swapped: {len(dev_swap_subset)}")

    # Dropping rows from exisitng dataframes
    logger.info(f"Original dev length: {len(dfs['dev'])}")
    logger.info(f"Original test length: {len(dfs['test'])}")
    dfs["dev"] = dfs["dev"][~dfs["dev"]["Image Index"].isin(dev_swap_subset["Image Index"])]
    dfs["test"] = dfs["test"][~dfs["test"]["Image Index"].isin(drain_swap_subset["Image Index"])]
    logger.info(f"Dev length after drop: {len(dfs['dev'])}")
    logger.info(f"Test length after drop: {len(dfs['test'])}")

    # Adding subsets back
    dfs["dev"] = dfs["dev"].append(drain_swap_subset)
    dfs["test"] = dfs["test"].append(dev_swap_subset)
    logger.info(f"Dev length after re-add: {len(dfs['dev'])}")
    logger.info(f"Test length after re-add: {len(dfs['test'])}")

    # Number in original subset in dev set:
    dev_drains = len(dfs["dev"][dfs["dev"]["Image Index"].isin(drains["Image Index"])])
    test_drains = len(dfs["test"][dfs["test"]["Image Index"].isin(drains["Image Index"])])
    logger.info(f"Number of drains in dev: {dev_drains}")
    logger.info(f"Number of drains in test: {test_drains}")

    # Number in original subset in dev set:
    dev_no_drains = len(dfs["dev"][dfs["dev"]["Image Index"].isin(no_drains["Image Index"])])
    test_no_drains = len(dfs["test"][dfs["test"]["Image Index"].isin(no_drains["Image Index"])])
    logger.info(f"Number of no drains in dev: {dev_no_drains}")
    logger.info(f"Number of no drains in test: {test_no_drains}")

    # Saving
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    for split in dfs:
        dfs[split].to_csv(os.path.join(args.out_dir,f"{split}.tsv"), sep='\t', index=False)
