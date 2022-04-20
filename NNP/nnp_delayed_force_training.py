from argparse import ArgumentParser
import os
from tracemalloc import start
import torch
import pytorch_lightning as pl
import torchani
import torchani.data
from pytorch_lightning.loggers import WandbLogger
import wandb
from NNP.nnp_data_module import NNPDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class NNPLightningModelDF(pl.LightningModule):
        def __init__(self, force_coefficient: int = 1, learning_rate: float=1e-6, batch_size: int=32, aev_dim: int=1, aev_computer: torchani.AEVComputer=None, start_force_training_epoch: int=0):
            super().__init__()
            
            self.H_network = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(240,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30, 1)
            )

            self.C_network = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(240,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30, 1)
            )

            self.S_network = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(240,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30,30),
                torch.nn.Tanh(),
                torch.nn.Linear(30, 1)
            )

            self.batch_size = batch_size
            
            self.mse = torch.nn.MSELoss(reduction='none')
            self.start_force_training_epoch = start_force_training_epoch
            self.nn = torchani.ANIModel([self.H_network, self.C_network, self.S_network])
            self.model = torchani.nn.Sequential(aev_computer, self.nn)
            self.learning_rate = learning_rate
            self.force_coefficient = force_coefficient
               

        
        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument('--learning_rate', type=float, default=1e-6)
            parser.add_argument('--start_force_training_epoch', type=int, default=0)
            return parser        
          
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            # Adam_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=0, verbose=True)
            # return {"optimizer": optimizer, "lr_scheduler": Adam_scheduler, "monitor": "val_force_loss"}
            return optimizer
        
        def forward(self, species, coordinates):
            _, predicted_energies = self.model((species, coordinates))
            return predicted_energies

        def training_step(self, batch, batch_idx):
            species = batch['species']
            coordinates = batch['coordinates'].float().requires_grad_(True)
            true_energies = batch['energies'].float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            energies = self.forward(species, coordinates)
            energy_loss = (self.mse(energies, true_energies) / num_atoms.sqrt()).mean()
            self.log('energy_loss', energy_loss)        
            if self.current_epoch >= self.start_force_training_epoch:                
                true_forces = batch['forces'].float()
                forces = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                force_loss = (self.mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
                loss = energy_loss + self.force_coefficient * force_loss
                self.log('force_loss', force_loss)
            else:
                loss = energy_loss

            return loss.float()

        def validation_step(self, val_batch, val_batch_idx):
            torch.set_grad_enabled(True)
            species = val_batch['species']
            coordinates = val_batch['coordinates'].float().requires_grad_(True)
            true_energies = val_batch['energies'].float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            energies = self.forward(species, coordinates)
            energy_loss = (self.mse(energies, true_energies) / num_atoms.sqrt()).mean()
            self.log('val_energy_loss', energy_loss)
            if self.current_epoch >= self.start_force_training_epoch:  
                true_forces = val_batch['forces'].float()
                forces = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                force_loss = (self.mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
                loss = energy_loss + self.force_coefficient * force_loss
                self.log('val_force_loss', force_loss)
            else:
                loss = energy_loss
            
            torch.save(self.nn.state_dict(), "nnp.pt")    
            return loss.float()

        def test_step(self, test_batch, test_batch_idx):
            torch.set_grad_enabled(True)
            species = test_batch['species']
            coordinates = test_batch['coordinates'].float().requires_grad_(True)
            true_energies = test_batch['energies'].float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            energies = self.forward(species, coordinates)
            energy_loss = (self.mse(energies, true_energies) / num_atoms.sqrt()).mean()
            self.log('test_energy_loss', energy_loss)
            true_forces = test_batch['forces'].float()
            forces = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
            force_loss = (self.mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
            loss = energy_loss + self.force_coefficient * force_loss
            self.log('test_force_loss', force_loss)
     
            return loss.float()

def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--force_coefficient', type=float, default=1)
    parser.add_argument('--use_cuda_extension', type=bool, default=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = NNPLightningModelDF.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # ------------
    # data
    # ------------
    data = NNPDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    aev_dim = data.get_aev_dim()
    aev_computer = data.aev_computer
    
    # ------------
    # model
    # ------------
    nnp = NNPLightningModelDF(learning_rate=args.learning_rate, batch_size=args.batch_size, aev_computer=aev_computer, aev_dim=aev_dim, start_force_training_epoch=args.start_force_training_epoch)
 
    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(dirpath="runs", save_top_k=20, monitor="val_force_loss")
    early_stopping = EarlyStopping(monitor="val_force_loss", mode="min", patience=100)
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, max_epochs=1000, callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(nnp, data)

    # ------------
    # testing
    # ------------
    trainer.test(ckpt_path="best", dataloaders=data.validation)


if __name__ == '__main__':
    cli_main()            