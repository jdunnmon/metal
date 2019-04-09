import sys, os
sys.path.append('../../../metal/')

import numpy as np
import argparse
from objdict import ObjDict
import importlib
import torch

from metal.end_model import EndModel
from metal.modules import LSTMModule
from model.sampler import ImbalancedDatasetSampler
from model.dataloader import fetch_dataloader
from metal.tuners import RandomSearchTuner
from metal.contrib.logging.tensorboard import TensorBoardWriter

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--raw_data_file', required=True, 
                   type=str, help='data file containing dev set labels for slice training' )

parser.add_argument('--subset_data_file', required=True,
                   type=str, help='data file containing indices of pneumothoraces with chest drain' )

parser.add_argument('--config', required=True, type=str,
                    help='path to config dict')

def train_model():
    
    # Parsing arguments

    args = parser.parse_args()   
    config_in = importlib.import_module(args.config)
    em_config = config_in.em_config
    search_space = config_in.search_space
    log_config = config_in.log_config
    tuner_config = config_in.tuner_config

    # Setting up additional params for dataloaders
    params = ObjDict()
    params.batch_size = em_config['train_config']['data_loader_config']['batch_size']
    params.num_workers = em_config['train_config']['data_loader_config']['num_workers']
    params.cuda = torch.cuda.is_available()
    
    # Getting sampler
    if em_config['train_config']['data_loader_config']['sampler'] == 'imbalanced':
        sampler=ImbalancedDatasetSampler
    else:
        sampler=None    
    
    # Fetching dataloaders 
    dataloaders = fetch_dataloader(['train','val','test'],args.data_dir,args.file_dir,args.label_dir,args.raw_data_dir,params, max_len=em_config['max_len'], src=em_config['src'], sampler=sampler, norm='patient', train_size=em_config['train_size'], test_gold=True)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Defining network parameters
    num_classes = em_config['num_classes']
    fc_size = em_config['fc_size']
    hidden_size=em_config['hidden_size']
    embed_size=np.shape(train_loader.dataset.__getitem__(0)[0])[1]
    em_config['use_cuda']=params.cuda
    metric = em_config['train_config']['validation_metric']

    # Initializing searcher
    searcher = RandomSearchTuner(
        EndModel,
        module_classes={"input_module": LSTMModule},
        log_writer_class=TensorBoardWriter,
        **log_config, # set this?
        validation_metric = metric,
    )

    init_kwargs = {'layer_out_dims': [2*hidden_size,fc_size,num_classes],
    }
    init_kwargs.update(em_config)
    max_search = tuner_config['max_search']
    
    # LSTMModule args & kwargs
    module_args = {}
    module_args["input_module"] = (embed_size, hidden_size)
    module_kwargs = {}
    module_kwargs["input_module"] = {
        "skip_embeddings": True,
        "seed": 123,
        "lstm_num_layers" : em_config['num_layers'], 
        "bidirectional": True,
        "verbose": True,
        "lstm_reduction": "attention",
        "lstm_reduction":em_config['lstm_reduction']
    }

    end_model = searcher.search(search_space, val_loader, \
            train_args=[train_loader],
            init_kwargs=init_kwargs, train_kwargs=em_config['train_config'],
            module_args=module_args, module_kwargs=module_kwargs,
            max_search=max_search, clean_up=False)

    # Evaluating model
    print("EVALUATING ON DEV SET...")
    end_model.score(val_loader, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])
    
    print("EVALUATING ON TEST SET...")
    end_model.score(test_loader, metric=['accuracy', 'precision', 'recall', 'f1','roc-auc'])

def fetch_dataloaders():
    datasets = create_datasets()

def create_datasets():


if __name__ == "__main__":
    #try:
    train_model()
