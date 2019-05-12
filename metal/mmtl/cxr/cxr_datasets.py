import logging
import os
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from tqdm import tqdm

from metal.mmtl.cxr.cxr_preprocess import get_task_config
from metal.utils import padded_tensor, set_seed

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CXR8Dataset(Dataset):
    """
    Dataset to load NIH Chest X-ray 14 dataset

    Modified from reproduce-chexnet repo
    https://github.com/jrzech/reproduce-chexnet
    """

    def __init__(
        self,
        path_to_images,
        path_to_labels,
        split,
        transform=None,  # Currently no support for this
        subsample=0,
        finding="ALL",
        dataset_name="CXR8",
        pooled=False,
        get_uid=False,
        slice_labels=None,
        sample_dict=None,
        single_task=None,
        return_dict=True,
        add_normal_col=False,
        label_transform={},
        seed=None,
    ):

        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.split = split
        self.df = pd.read_csv(self.path_to_labels, sep="\t")
        self.df.columns = map(str.upper, self.df.columns)
        self.get_uid = get_uid
        self.add_normal_col = add_normal_col
        self.labels = {}
        self.sample_dict = sample_dict
        self.pooled = pooled
        self.single_task = single_task
        self.dataset_name = dataset_name
        self.return_dict = return_dict
        self.label_transform = label_transform
        if seed is None:
            self.seed = 123
        else:
            self.seed = int(seed)
        logger.debug(f"Using dataset seed: {self.seed}")
        
        self.PRED_LABEL = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        if subsample > 0 and subsample < len(self.df):
            self.df = self.df.sample(subsample, random_state=self.seed)
            
        if add_normal_col:
            row_sums = self.df[[a.upper() for a in self.PRED_LABEL]].sum(axis=1)
            logger.info("Adding normal column to dataset")
            self.df['NORMAL'] = [int(a==0) for a in row_sums]
            self.PRED_LABEL+=['Normal']
            
        classes = self.PRED_LABEL
        if slice_labels is not None:
            if isinstance(slice_labels, str):
                slice_labels = [slice_labels]
            classes = classes + slice_labels
            
        if sample_dict:
            assert 'ALL' in self.sample_dict, "Sample dict must have key 'ALL'"
            logger.info("Using pre-specified sampling dictionary")
            df_adj = pd.DataFrame() #columns=self.df.keys()
            for k in classes:
                ky = k.upper()
                #if ky == 'PNEUMOTHORAX':
                #    import pdb; pdb.set_trace()
                df_k = self.df.loc[self.df[ky]==1]
                smp = self.sample_dict[ky] if ky in self.sample_dict else self.sample_dict['ALL']
                smp = min(len(df_k),smp) if smp>0 else len(df_k)
                df_k = df_k.sample(smp, random_state=self.seed)
                df_adj = df_adj.append(df_k)
            self.df = df_adj
            self.df.drop_duplicates(keep="first", inplace=True)

        if (
            not finding == "ALL"
        ):  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    raise ValueError(
                        "No positive cases exist for "
                        + LABEL
                        + ", returning all unfiltered cases"
                    )
            else:
                raise ValueError(
                    "cannot filter on finding "
                    + finding
                    + " as not in data - please check spelling"
                )

        self.uids = self.df["IMAGE INDEX"].tolist()
        self.df = self.df.set_index("IMAGE INDEX")

        # Adding tasks and labels -- right now, we train all labels associated with
        # a given task!
        for cls in classes:
            cls_upper = cls.upper().strip()
            label_vec = (self.df[cls_upper].astype(int) > 0).astype(int)
            # Converting to metal format: 0 abstain, 2 negative
            label_vec[label_vec == 0] = 2
            if cls_upper in self.label_transform.keys():
                logger.info(f"Transforming labels for {cls} class")
                label_vec = [self.label_transform[cls_upper](l) for l in label_vec]
            # label_set = torch.tensor(label_vec).int()
            # if label_set.dim()<2:
            #    label_set = label_set[:,None]

            label_set = np.array(label_vec).astype(int)
            if self.pooled:
                self.labels[cls_upper] = label_set
            else:
                self.labels[f"{self.dataset_name}_{cls_upper}"] = label_set

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.path_to_images, self.df.index[idx]))
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        uid = self.df.index[idx]
        if self.return_dict:
            x = {"data": image}
            if self.get_uid:
                x["uid"] = uid
        else:
            x = image

        # If statement to train classifiers for single tasks outside mmtl
        if self.single_task is not None:
            ky = (
                f"{self.dataset_name}_{self.single_task.upper()}"
                if not self.pooled
                else self.single_task.upper()
            )
            ys = self.labels[ky][idx]
        else:
            ys = {
                task_name: label_set[idx]
                for task_name, label_set in self.labels.items()
            }
        
        if self.get_uid and not self.return_dict:
            return x, ys, uid
        else:
            return x, ys

    def __len__(self):
        return len(self.df)

    def get_dataloader(self, split_prop=None, split_seed=123, **kwargs):
        """Returns a dataloader based on self (dataset). If split_prop is specified,
        returns a split dataset assuming train -> split_prop and dev -> 1 - split_prop."""

        if split_prop:
            assert split_prop > 0 and split_prop < 1

            # choose random indexes for train/dev
            N = len(self)
            full_idx = np.arange(N)
            set_seed(split_seed)
            np.random.shuffle(full_idx)

            # split into train/dev
            split_div = int(split_prop * N)
            train_idx = full_idx[:split_div]
            dev_idx = full_idx[split_div:]

            # create data loaders
            train_dataloader = DataLoader(
                self,
                collate_fn=self._collate_fn,
                sampler=SubsetRandomSampler(train_idx),
                **kwargs,
            )

            dev_dataloader = DataLoader(
                self,
                collate_fn=self._collate_fn,
                sampler=SubsetRandomSampler(dev_idx),
                **kwargs,
            )

            return train_dataloader, dev_dataloader

        else:
            return DataLoader(self, collate_fn=self._collate_fn, **kwargs)

    def _collate_fn(self, batch_list):
        """Collates batch of (images, labels) into padded (X, Ys) tensors

        Args:
            batch_list: a list of tuples containing (images, labels)
        Returns:
            X: images
            Y: a dict of {task_name: labels} where labels[idx] are the appropriate
                labels for that task
        """
        Y_lists = {task_name: [] for task_name in self.labels}
        X_list = []
        uid_list = []
        for instance in batch_list:
            if self.get_uid and not self.return_dict:
                x, ys, uid = instance
            else:
                x, ys = instance

            if self.return_dict:
                image = x["data"]
                if self.get_uid:
                    uid = x["uid"]
            else:
                image = x

            for task_name, y in ys.items():
                Y_lists[task_name].append(y)

            X_list.append(image)

            if self.get_uid:
                uid_list.append(uid)

        if self.return_dict:
            Xs = {"data": torch.stack(X_list)}
            if self.get_uid:
                Xs["uid"] = uid_list
        else:
            Xs = torch.stack(X_list)

        Ys = self._collate_labels(Y_lists)

        if self.get_uid and not self.return_dict:
            uids = uid_list
            return Xs, Ys, uids
        else:
            return Xs, Ys

    def _collate_labels(self, Ys):
        """Collate potentially multiple label_sets 
 
        Args: 
            Ys: a dict of the form {task_name: label_list}, where label_list is a 
                list of individual labels (ints, floats, numpy, or torch) belonging to 
                the same label_set; labels may be a scalar or a sequence. 
        Returns: 
            Ys: a dict of the form {task_name: labels}, with labels containing a torch 
                Tensor (padded if necessary) of labels belonging to the same label_set 
 
 
        Convert each Y in Ys from: 
            list of scalars (instance labels) -> [n,] tensor 
            list of tensors/lists/arrays (token labels) -> [n, seq_len] padded tensor 
        """
        for task_name, Y in Ys.items():
            if isinstance(Y[0], (int, np.int64)):
                Y = torch.tensor(Y, dtype=torch.long)
            elif isinstance(Y[0], torch.Tensor) and len(Y[0].size()) < 2:
                Y = torch.tensor(Y, dtype=torch.float)
            elif isinstance(Y[0], np.integer):
                Y = torch.from_numpy(np.array(Y))
            elif isinstance(Y[0], float):
                Y = torch.tensor(Y, dtype=torch.float)
            elif isinstance(Y[0], np.float):
                Y = torch.from_numpy(Y)
            elif isinstance(Y[0], torch.FloatTensor):
                Y = torch.tensor(Y, dtype=torch.long)
            elif (
                isinstance(Y[0], list)
                or isinstance(Y[0], np.ndarray)
                or isinstance(Y[0], torch.Tensor)
                and len(Y[0].size()) >= 2
            ):
                if isinstance(Y[0][0], (int, np.integer)):
                    dtype = torch.long
                elif isinstance(Y[0][0], (float, np.float)):
                    # TODO: WARNING: this may not handle half-precision correctly!
                    dtype = torch.float
                else:
                    msg = (
                        f"Unrecognized dtype of elements in label_set for task "
                        f"{task_name}: {type(Y[0][0])}"
                    )
                    raise Exception(msg)
                Y, _ = padded_tensor(Y, dtype=dtype)
            else:
                msg = (
                    f"Unrecognized dtype of label_set for task {task_name}: "
                    f"{type(Y[0])}"
                )
                raise Exception(msg)
            # Ensure that first dimension of Y is n
            if Y.dim() == 1:
                Y = Y.view(-1, 1)
            Ys[task_name] = Y
        return Ys


DATASET_CLASS_DICT = {"CXR8": CXR8Dataset, "CXR8-DRAIN": CXR8Dataset}


def get_cxr_dataset(
    dataset_name, split, subsample=None, finding="ALL", pooled=False, **kwargs
):
    """ Create and returns specified cxr dataset based on image path."""

    # MODIFY THIS TO GET THE RIGHT LOCATIONS FOR EACH!!
    if "_" in dataset_name:
        dataset_name = dataset_name.split("_")[0]
    if "_" in finding:
        finding = finding.split("_")[1]
    transform_kwargs = kwargs["transform_kwargs"]
    config = get_task_config(dataset_name, split, subsample, finding, transform_kwargs)
    config["get_uid"] = kwargs.get("get_uid", False)
    config["return_dict"] = kwargs.get("return_dict", True)
    config["seed"] = kwargs.get("seed", None)
    sample_dict=kwargs.get("sample_dict",None)
    add_normal_col=kwargs.get("add_normal_col",False)
    dataset_class = DATASET_CLASS_DICT[dataset_name]

    return dataset_class(
        config["path_to_images"],
        config["path_to_labels"],
        split,
        transform=config["transform"],
        subsample=config["subsample"],
        finding=config["finding"],
        pooled=False,
        get_uid=config["get_uid"],
        dataset_name=dataset_name,
        return_dict=config["return_dict"],
        sample_dict=sample_dict,
        add_normal_col=add_normal_col,
        seed=config["seed"],
    )
