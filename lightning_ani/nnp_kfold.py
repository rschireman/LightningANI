import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional, Type

import torch
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.demos.boring_classes import Net
from pytorch_lightning.demos.mnist_datamodule import MNIST
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn

class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass

@dataclass
class NNPKfoldDataModule(BaseKFoldDataModule):

    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None