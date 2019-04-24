writer_config = {
        'log_dir' : './slice_run_logs/CXR8-DRAIN',
        'run_name' : 'test',
        #'include_config': True,
        }

tuner_config = {
        'max_search': 15,
        'seed':123,
        'include_config':True,
}

search_space = {
        'l2': {'range': [0.000001, 0.001], 'scale':'log'},           # linear range
        'lr': {'range': [0.0001, 0.01], 'scale': 'log'},  # log range
        # 'l2': [1.826899859912767e-05],
        # 'lr': [0.0003294833709327656]
        }

def flip_pos_neg_labs(x):
    if x == 2:
        return 1
    elif x == 1:
        return 2
    elif x == 0:
        return 0
    else:
        raise ValueError(f"Unrecognized label value {x}")

em_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": False,
    # Network
    # The first value is the output dim of the input module (or the sum of
    # the output dims of all the input modules if multitask=True and
    # multiple input modules are provided). The last value is the
    # output dim of the head layer (i.e., the cardinality of the
    # classification task). The remaining values are the output dims of
    # middle layers (if any). The number of middle layers will be inferred
    # from this list.
    # Input layer configs
    "input_layer_config": {
        "input_relu": False,
        "input_batchnorm": False,
        "input_dropout": 0.0,
    },
    # Middle layer configs
    "middle_layer_config": {
        "middle_relu": False,
        "middle_batchnorm": False,
        "middle_dropout": 0.0,
    },
    # Can optionally skip the head layer completely, for e.g. running baseline
    # models...
    "skip_head": True,
    # GPU
    "use_cuda": False,
    # MODEL CLASS
    "pretrained": True,
    "num_classes": 2,
    # DATA
    "label_transform": {"DRAIN": flip_pos_neg_labs},
    # TRAINING
    "train_config": {
         # Loss function config
        "loss_fn_reduction": "sum",
        # Display
        "print_every": 1,  # Print after this many epochs
        "disable_prog_bar": False,  # Disable progress bar each epoch
        # Dataloader
        "data_loader_config": {"batch_size": 16, 
                                "num_workers": 8, 
                                "sampler": None,
        },
       # Loss weights
        "loss_weights": [0.66, 0.33],
        # Train Loop
        "n_epochs": 20,
        # 'grad_clip': 0.0,
         "l2": 0.0,
         #"lr": 0.01,
        "validation_metric": "f1",
        "validation_freq": 1,
        "validation_scoring_kwargs": {},
        "log_valid_metrics":['accuracy','f1'],
        # Evaluate dev for during training every this many epochs
        # Optimizer
        "optimizer_config": {
            "optimizer": "adam",
            "optimizer_common": {"lr": 0.01},
            # Optimizer - SGD
            "sgd_config": {"momentum": 0.9},
            # Optimizer - Adam
            "adam_config": {"betas": (0.9, 0.999)},
        },
        # Scheduler
        "scheduler_config": {
            "scheduler": "reduce_on_plateau",
            # ['constant', 'exponential', 'reduce_on_plateu']
            # Freeze learning rate initially this many epochs
            "lr_freeze": 0,
            # Scheduler - exponential
            "exponential_config": {"gamma": 0.9},  # decay rate
            # Scheduler - reduce_on_plateau
            "plateau_config": {
                "factor": 0.5,
                "patience": 2,
                "threshold": 0.0001,
                "min_lr": 1e-5,
            },
        },
        # Checkpointer
        "checkpoint": True,
        "checkpoint_config": {
            "checkpoint_min": -1,
            "checkpoint_metric":'f1',
            # The initial best score to beat to merit checkpointing
            "checkpoint_runway": 0,
            # Don't start taking checkpoints until after this many epochs
        },
    },
}
