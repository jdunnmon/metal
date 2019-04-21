import argparse
import logging
import os
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm

from metal.mmtl.cxr.cxr_tasks import create_cxr_datasets, create_cxr_dataloaders

# Configuring logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Before running, must set CXRDATA env to appropriate location
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset_name', '-lp', required=True,
                   type=str, help='name of dataset' )
    parser.add_argument('--slice_module_path','-smp', type=str,
                   help='path to saved slice module')
    parser.add_argument('--metal_model', '-mm', type=int, required=True,
                   help='use metal label convention if true')
    parser.add_argument('--slice_label', '-sl', type=int, required=True,
                   help='module label associated with slice, assume softmax')
    parser.add_argument('--load_type', '-lt', type=str,
                   default='torch', choices=['torch','pickle'])
    parser.add_argument('--save_dir', '-sd', type=str, required=True,
                   help='use metal label convention if true')
    parser.add_argument('--base_task', '-bt', type=str, required=True,
                   help='baseline task to use for slice')
    parser.add_argument('--splits', '-sp', type=str, required=True,
                   help='comma separated list')
    args = parser.parse_args()

    # Hard-coding resolution for now
    dataset_kwargs = {
        "transform_kwargs":{
            "res":224
        }
    }
    
    # detecting GPU
    use_cuda = torch.cuda.is_available()

    # Getting slice labels on all subsets
    splits = args.splits.split(',')

    logger.info(f"Getting datasets and dataloaders...")
    # Getting all datasets
    datasets = create_cxr_datasets(
                dataset_name=args.dataset_name,
                splits=splits,
                subsample=-1,
                pooled=False,
                finding=args.base_task,
                verbose=True,
                dataset_kwargs=dataset_kwargs,
                get_uid = True
                )

    # Wrap datasets with DataLoader objects
    data_loaders = create_cxr_dataloaders(
            datasets,
            dl_kwargs={'num_workers':8, 'batch_size':16},
            split_prop=None,
            splits=splits,
            seed=123,
        )

    logger.info(f"Loading saved slice module...")
    # Loading saved slice module; .forward() gives slice label
    if args.load_type == 'torch':
        slice_module = torch.load(args.slice_module_path)
    elif args.save_type == 'pickle':
        with open(args.slice_module_path, 'rb') as fl:
            slice_module = pickle.load(fl)
    else:
        raise ValueError('Unrecognized file type, must be pickle or torch')

    # Creating output directory and saving
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
 
    # Setting cuda if avialable
    if use_cuda:
        slice_module.cuda()

    # Logging metal convenction
    if args.metal_model:
        logger.info(f"Using MeTaL label convention...")
    # Evaluating slice module
    slice_label_dict = {}
    for split in splits:
        logger.info(f"Evaluating {split} split...")
        for ii, batch in tqdm(enumerate(data_loaders[split]), total=len(data_loaders[split])):
            X, y, uids = batch
            X = Variable(X)
            if use_cuda:
                X = X.cuda()
            if args.metal_model:
                slice_labels = slice_module.predict(X)
            else:
                slice_labels = slice_module.predict(X)
            slice_labels = [int(l == args.slice_label) for l in slice_labels]
            batch_dict = dict(zip(uids, slice_labels))
            slice_label_dict.update(batch_dict)

        # Creating dataframe
        slice_df = pd.DataFrame(
                        {'data_index' : list(slice_label_dict.keys()),
                         'slice_label' : list(slice_label_dict.values())
                        })

        
        slice_df.to_csv(f"{args.save_dir}/{split}.tsv",sep='\t',index=False)

    logger.info("Slice data saved")
