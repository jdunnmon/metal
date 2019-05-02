import argparse
import copy
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuring logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_reproduce_chexnet_data(dr):
    dfs = {}
    for split in ["train", "dev", "test"]:
        dfs[split] = pd.read_csv(os.path.join(dr, f"{split}.tsv"), sep="\t")
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rcdwd_dir",
        "-rcd",
        help="Directory containing modified NIH data (with drains) from reproduce-chexnet",
        required=True,
    )
    parser.add_argument(
        "--drain_file",
        "-dd",
        help="File containing list of confirmed chest drains",
        required=True,
    )
    parser.add_argument(
        "--out_dir", "-od", help="Output directory for drain slice data"
    )
    args = parser.parse_args()

    # Setting seed
    SEED = 1701

    # Reading in NIH data
    dfs = read_reproduce_chexnet_data(args.rcdwd_dir)

    # Reading in pneumo subset from LOR
    pneumo_subset = pd.read_csv(args.drain_file)

    # Identifying drains
    drains = pneumo_subset[pneumo_subset["drain"] == 1]
    no_drains = pneumo_subset[pneumo_subset["drain"] == 0]

    # Getting pneumos from dev set
    df_dev = copy.deepcopy(dfs["dev"])
    dev_pneumos = df_dev[df_dev["Pneumothorax"] == 1]

    # Getting drains in dev set
    dev_drains = dev_pneumos[dev_pneumos["Image Index"].isin(drains["Image Index"])]
    dev_no_drains_subset = dev_pneumos[
        dev_pneumos["Image Index"].isin(no_drains["Image Index"])
    ]

    # Adding drain label
    dev_drains["drain"] = 1
    dev_no_drains_subset["drain"] = 0

    # Creating dev, train, test sets with 80-10-10
    dfs_out = {}
    drain_data_tot = dev_drains.append(dev_no_drains_subset).sample(frac=1)
    drain_train_plus_val, dfs_out["test"] = train_test_split(
        drain_data_tot, test_size=0.1
    )
    dfs_out["train"], dfs_out["dev"] = train_test_split(
        drain_train_plus_val, test_size=0.1
    )

    for split in ["train", "dev", "test"]:
        st = dfs_out[split]
        logger.info(
            f"{split} set: {len(st)} Examples, {100*np.sum(st['drain'])/len(st):0.2f} Percent Positive"
        )
        st.to_csv(os.path.join(args.out_dir, f"{split}.tsv"), sep="\t", index=False)
