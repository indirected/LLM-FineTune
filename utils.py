import datasets
from OmegaConf import DictConfig
import hydra
import torch

def prepare_data(datacfg: DictConfig) -> datasets.DatasetDict:
    dataset = hydra.utils.instantiate(datacfg.dataset.HF_cls)
    train_size = int(len(dataset)*datacfg.split_sizes.train)
    val_size = int(len(dataset)*datacfg.split_sizes.val)
    test_size = len(dataset) - train_size - val_size

    train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(datacfg.split_seed))

    dataset = datasets.DatasetDict({
        'train': dataset.select(train.indices),
        'val': dataset.select(val.indices),
        'test': dataset.select(test.indices)
    })

    return dataset