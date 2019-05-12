import torch
import torch.utils.data
import torchvision

import metal

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class ImbalancedMMTLSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, active_tasks, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        
        # dataset to sample from
        self.dataset = dataset

        # weight for each sample
        self._set_weights_and_tasks(active_tasks)
 
    def _set_weights_and_tasks(self, active_tasks):
        # distribution of classes in the dataset
        self.label_to_count = {}
        for idx in self.indices:
            label_dict = self._get_labels(idx)
            for label in label_dict:
                if label_dict[label] == 1:
                    if label in self.label_to_count:
                        self.label_to_count[label] += 1
                    else:
                        self.label_to_count[label] = 1
                        
        self.active_tasks = active_tasks
        weights = [self._get_weight(idx) for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_weight(self, idx):
        labs = self._get_labels(idx)
        min_count = 1e9 if self.active_tasks else len(self.dataset)
        for task, lab in labs.items():
            if not self.active_tasks:
                if lab==1:
                    min_count = min(min_count,self.label_to_count[task])
            elif (task in self.active_tasks) and (lab==1):
                min_count = min(min_count,self.label_to_count[task])
        return 1/min_count

    def _get_labels(self, idx):
        dataset_type = type(self.dataset)
        if dataset_type is metal.mmtl.cxr.cxr_datasets.CXR8Dataset:
            dct = {ky:self.dataset.labels[ky][idx] for ky in self.dataset.labels.keys()}
            return dct
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples
