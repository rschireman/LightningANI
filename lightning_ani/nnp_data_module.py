from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
import torchani
import torchani.data
from typing import List

class NNPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int=32, use_cuda_extension: bool = False, Rcr: float=5.600, Rca: float=3.50, EtaR: List=[16.00], 
    ShfR: List=[9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00],
    Zeta: List=[32.00], ShfZ: List=[1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00],
    EtaA: List=[8.0000000e+00], ShfA: List=[9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00]
    ):
        
        super().__init__()
    
        self.Rcr = Rcr
        print("Proceeding with Rcr: ", self.Rcr)
        self.Rca = Rca
        self.EtaR = torch.tensor(EtaR)
        self.ShfR = torch.tensor(ShfR)
        self.Zeta = torch.tensor(Zeta)
        self.ShfZ = torch.tensor(ShfZ)
        self.EtaA = torch.tensor(EtaA)
        self.ShfA = torch.tensor(ShfA)
        
        self.use_cuda_extension = use_cuda_extension
        self.species_order = ['H','O']
        self.num_species = len(self.species_order)
        """
        @todo #3 can't use cuda extension when vibrational analysis is called
        """
        self.aev_computer = torchani.AEVComputer(self.Rcr, self.Rca, self.EtaR, self.ShfR, self.EtaA, self.Zeta, self.ShfA, self.ShfZ, self.num_species)
        
        self.data_dir = data_dir

        self.training, self.validation = torchani.data.load(self.data_dir, additional_properties=('forces',)).species_to_indices(self.species_order).shuffle().split(0.8, None)
        self.training = torch.utils.data.DataLoader(list(self.training), batch_size=batch_size,  num_workers=2, pin_memory=True)
        self.validation = torch.utils.data.DataLoader(list(self.validation), batch_size=batch_size, num_workers=2, pin_memory=True)
        self.train_set_size = int(0.5*len(self.validation.dataset))
        self.valid_set_size = int(len(self.validation.dataset) - self.train_set_size)

        self.valid_set, self.test_set = torch.utils.data.random_split(self.validation.dataset, [self.valid_set_size, self.train_set_size])

    def train_dataloader(self):
        return self.training

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set)

    def get_aev_dim(self): 
        self.aev_dim = self.aev_computer.aev_length
        return self.aev_dim
