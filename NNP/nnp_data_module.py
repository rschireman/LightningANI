from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
import torchani
import torchani.data


class NNPDataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = "./", batch_size: int = 32):
            super().__init__()
            
            self.Rcr = 5.2000e+00
            self.Rca = 3.5000e+00
            self.EtaR = torch.tensor([1.6000000e+01])
            self.ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00])
            self.Zeta = torch.tensor([3.2000000e+01])
            self.ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00])
            self.EtaA = torch.tensor([8.0000000e+00])
            self.ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00])

            self.species_order = ['H',"C","S"]
            self.num_species = len(self.species_order)
            self.aev_computer = torchani.AEVComputer(self.Rcr, self.Rca, self.EtaR, self.ShfR, self.EtaA, self.Zeta, self.ShfA, self.ShfZ, self.num_species)
            
            
            self.data_dir = data_dir
            self.training, self.validation = torchani.data.load(
            self.data_dir,
            additional_properties=('forces',)).species_to_indices(self.species_order).shuffle().split(0.8, None)
            
            self.batch_size = batch_size

            self.training = self.training.collate(batch_size=self.batch_size).cache()
            self.validation = self.validation.collate(batch_size=self.batch_size).cache()
        
        def train_dataloader(self):
            return self.training

        def val_dataloader(self):
            return self.validation,

        def get_aev_dim(self): 
            self.aev_dim = self.aev_computer.aev_length
            return self.aev_dim