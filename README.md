# Snorkel MeTaL (previously known as _MuTS_)

<img src="assets/logo_01.png" width="150"/>

Snorkel MeTaL: A framework for using multi-task weak supervision (MTS), provided by users in the form of _labeling functions_ applied over unlabeled data, to train multi-task models.
Snorkel MeTaL can use the output of labeling functions developed and executed in [Snorkel](snorkel.stanford.edu), or take in arbitrary _label matrices_ representing weak supervision from multiple sources of unknown quality, and then use this to train auto-compiled MTL networks.

Check out the tutorial on basic usage: https://github.com/HazyResearch/metal/blob/master/Tutorial.ipynb

For more detail, see the **working draft of our technical report on MeTaL: [_Training Complex Models with Multi-Task Weak Supervision_](https://ajratner.github.io/assets/papers/mts-draft.pdf)**

## Setup
[1] Install anaconda3 (https://www.anaconda.com/download/#macos)

[2] Create conda environment:
```
conda create -n metal python=3.6
source activate metal
```

[3] Download dependencies:
```
conda install -q matplotlib numpy pandas pytorch scipy torchvision -c pytorch
conda install --channel=conda-forge nb_conda_kernels
```

[4] Set environment:
```
git clone https://github.com/HazyResearch/metal.git
cd metal
source set_env.sh
```

[5] Test functionality:
```
nosetests
```

[6] View analysis tools:
[launch jupyter notebook] (selecting your metal conda environment as the kernel)

```jupyter notebook```
Navigate to ```notebooks/Tools.ipynb```

restart and run all
