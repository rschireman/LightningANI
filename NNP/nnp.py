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
   

class NNPLightningModel(pl.LightningModule):
    def __init__(self, force_coefficient: int = 10, learning_rate: float=1e-4, aev_dim: int=1, aev_computer: torchani.AEVComputer=None):
        super().__init__()
        self.save_hyperparameters()

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
        print(self.hparams.learning_rate)
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
        torch.save(self.nn.state_dict(), "./nnp.pt")    
        return loss.float()        



def cli_main():
    pl.seed_everything(1234)

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
    trainer = pl.Trainer.from_argparse_args(args, gpus=1)
    trainer.fit(nnp, data)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
