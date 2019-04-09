import sys, os
sys.path.append('../../../metal/')

import numpy as np
import argparse
from objdict import ObjDict
import importlib
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import shutil
import pickle

from metal.end_model import EndModel
from metal.tuners import RandomSearchTuner
from metal.logging.tensorboard import TensorBoardWriter
from metal.mmtl.cxr.cxr_datasets import CXR8Dataset
from metal.mmtl.cxr.utils.sampler import ImbalancedDatasetSampler
from metal.mmtl.cxr.cxr_preprocess import transform_for_dataset

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--path_to_images', '-ip', required=True,
                   type=str, help='path to image data' )
parser.add_argument('--path_to_labels', '-lp', required=True,
                   type=str, help='path to label directory' )
parser.add_argument('--config', required=True, type=str,
                    help='path to config dict')

def resnet18_wrap(**kwargs):
        pretrained = kwargs.get('pretrained',False)
        num_classes = kwargs.get('num_classes',-1)
        print(f"Pretrained: {pretrained}")
        print(f"Number of Classes: {num_classes}")
        model = models.resnet18(pretrained=pretrained)
        last_layer_input_size=int(model.fc.weight.size()[1])
        model.fc = torch.nn.Linear(last_layer_input_size, num_classes)
        return model
    
def train_model():
    
    # Parsing arguments

    args = parser.parse_args()   
    config_in = importlib.import_module(args.config)
    em_config = config_in.em_config
    search_space = config_in.search_space
    writer_config = config_in.writer_config
    tuner_config = config_in.tuner_config

    # Setting up additional params for dataloaders
    params = ObjDict()
    params.cuda = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Creating datasetsc
    splits = ['train', 'dev', 'test']
    dataloaders = {}
    for split in splits:
        dataset = CXR8Dataset(
            args.path_to_images,
            os.path.join(args.path_to_labels,f"{split}.tsv"),
            split,
            transform=transform_for_dataset("CXR8",split),
            subsample=0,
            finding="ALL",
            pooled=False,
            get_uid=False,
            slice_labels="DRAIN",
            single_task="DRAIN",
        )
        dataloaders[split] = DataLoader(dataset, **em_config['train_config']['data_loader_config'])

    # Getting sampler
    if em_config['train_config']['data_loader_config']['sampler'] == 'imbalanced':
        sampler=ImbalancedDatasetSampler
    else:
        sampler=None        

    # Fetching dataloaders 
    train_loader = dataloaders['train']
    val_loader = dataloaders['dev']
    test_loader = dataloaders['test']
    
    # Defining network parameters
    input_module = resnet18_wrap
    encode_dim = 512
    num_classes = em_config['num_classes']
    em_config['device']=params.cuda
    metric = em_config['train_config']['validation_metric']


    # Initializing logger to get log config
    log_writer_class = TensorBoardWriter

    # Initializing searcher
    searcher = RandomSearchTuner(
        EndModel,
        module_classes={"input_module": input_module},
        log_writer_class=log_writer_class,
        **writer_config,
        validation_metric = metric,
    )

    em_config['train_config']['checkpoint_config']['checkpoint_dir'] = os.path.join(searcher.log_subdir,'checkpoints')    

   
    init_kwargs = {'layer_out_dims':[encode_dim, num_classes]}
    init_kwargs.update(em_config)
    max_search = tuner_config['max_search']
    
    # Module args & kwargs
    module_args = {}
    module_args["input_module"] = ()
    module_kwargs = {}
    module_kwargs["input_module"] = {
        "pretrained": em_config['pretrained'],
        "num_classes": em_config['num_classes'],
    }
    
    end_model = searcher.search(search_space, val_loader, \
            train_args=[train_loader],
            init_kwargs=init_kwargs, train_kwargs=em_config['train_config'],
            module_args=module_args, module_kwargs=module_kwargs,
            max_search=max_search, clean_up=True)

    # Saving model and dataloaders
    print("Saving model and dataloaders...")
    end_model.save(os.path.join(searcher.log_subdir,"best_model.pkl"))
    with open(os.path.join(searcher.log_subdir,"dataloaders.pkl"),'wb') as fl:
        pickle.dump(dataloaders,fl)

    # Moving config over
    shutil.copy(f"{args.config}.py", searcher.log_subdir)
 
    # Evaluating model
    print("EVALUATING ON DEV SET...")
    end_model.score(val_loader, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])
    
    print("EVALUATING ON TEST SET...")
    end_model.score(test_loader, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])

if __name__ == "__main__":
    train_model()
