from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
import torchani
import torchani.data
from pytorch_lightning.loggers import WandbLogger
import wandb
from lightning_ani.nnp_data_module import NNPDataModule

class NNPLightningModel(pl.LightningModule):
    def __init__(self, force_coefficient: int = 10, learning_rate: float=1e-4, aev_dim: int=1, aev_computer: torchani.AEVComputer=None):
        super().__init__()
        self.save_hyperparameters()
        self.log('learning_rate',self.hparams.learning_rate)
        self.H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        self.C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )

        self.S_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1)
        )
        self.mse = torch.nn.MSELoss(reduction='none')
        self.nn = torchani.ANIModel([self.H_network, self.C_network, self.S_network])
        self.model = torchani.nn.Sequential(aev_computer, self.nn)
        self.force_coefficient = force_coefficient

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
        
    def forward(self, species, coordinates):
        _, predicted_energies = self.model((species, coordinates))
        return predicted_energies

    def training_step(self, batch, batch_idx):
        species = batch['species']
        coordinates = batch['coordinates'].float().requires_grad_(True)
        true_energies = batch['energies'].float()
        true_forces = batch['forces'].float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        energies = self.forward(species, coordinates)
        forces = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        energy_loss = (self.mse(energies, true_energies) / num_atoms.sqrt()).mean()
        force_loss = (self.mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
        loss = energy_loss + self.force_coefficient * force_loss
        self.log('energy_loss', energy_loss)
        self.log('force_loss', force_loss)
        return loss.float()

    def validation_step(self, val_batch, val_batch_idx):
        torch.set_grad_enabled(True)
        species = val_batch['species']
        coordinates = val_batch['coordinates'].float().requires_grad_(True)
        true_energies = val_batch['energies'].float()
        true_forces = val_batch['forces'].float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        energies = self.forward(species, coordinates)
        forces = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        energy_loss = (self.mse(energies, true_energies) / num_atoms.sqrt()).mean()
        force_loss = (self.mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
        loss = energy_loss + self.force_coefficient * force_loss
        self.log('val_energy_loss', energy_loss)
        self.log('val_force_loss', force_loss)
        torch.save(self.nn.state_dict(), "./nnp.pt.epoch_" + str(self.current_epoch))    
        return loss.float()        



def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--force_coefficient', type=float, default=10)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = NNPLightningModel.add_model_specific_args(parser)
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
    nnp = NNPLightningModel(learning_rate=args.learning_rate, aev_computer=aev_computer, aev_dim=aev_dim)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, max_epochs=10000)
    trainer.fit(nnp, data)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
