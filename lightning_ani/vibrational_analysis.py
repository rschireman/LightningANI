import glob
from tabnanny import check
from pytorch_lightning.callbacks import Callback
import torchani
from ase.optimize import BFGSLineSearch
from ase.io import read, write
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

class VibrationalAnalysis(Callback):
    def __init__(self, checkpoints_dir, model, aev_computer, vib_ref_data_path, pdb_path, species_order):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.model = model
        self.aev_computer = aev_computer
        self.vib_ref_data_path = vib_ref_data_path
        self.pdb_path = pdb_path
        self.species_order = species_order
    def on_train_end(self, trainer, pl_module):
        print("Running Vibrational Analysis")

        ckpts = glob.glob(self.checkpoints_dir + "/*.ckpt")
        sorted_ckpts = sorted(ckpts, key=lambda x: int(x.split("=")[1].replace("-step","")))
        print(sorted_ckpts)
        vib_error_list = []
        epoch_list = []
        for ckpt_model in sorted_ckpts:
            ckpt_nnp = self.model.load_from_checkpoint(ckpt_model)  
            nn = torchani.ANIModel([ckpt_nnp.H_network, ckpt_nnp.C_network, ckpt_nnp.S_network])
            test_model = torchani.nn.Sequential(self.aev_computer, nn).to("cuda:0")
            molecule = read(self.pdb_path)
            ase_calc = torchani.ase.Calculator(model=test_model, species=self.species_order)
            molecule.calc = ase_calc


            dyn = BFGSLineSearch(molecule)
            try:
                dyn.run(fmax=0.05,steps=1000)
            except RuntimeError:
                vib_error_list.append(0.0)
                epoch_list.append(int(ckpt_model.split("=")[1].replace("-step","")))
                continue
            

            species = torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long).unsqueeze(0)
            coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True).float().to('cuda:0')
            masses = torchani.utils.get_atomic_masses(species).to('cuda:0')
            cell = torch.tensor(molecule.get_cell()).float().to('cuda:0')
            pbc = torch.tensor([True,True,True]).to('cuda:0')
            """
            @todo #4 add species as input to callback
            """
            species = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	2,	2,	2,	2,
            ]],dtype=torch.long).to('cuda:0')

            energies = test_model((species, coordinates),cell=cell, pbc=pbc).energies

            hessian = torchani.utils.hessian(coordinates,energies=energies).to('cuda:0')

            freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
            torch.set_printoptions(precision=3, sci_mode=False)
            print('Force Constants (mDyne/A):', fconstants)
            print('Reduced masses (AMU):', rmasses)
            print('Modes:', modes)
            print('Frequencies (cm^-1):', freq)

            mode, true_freqs = np.loadtxt(self.vib_ref_data_path,skiprows=1, unpack=True)
            freq = torch.nan_to_num(freq).cpu().numpy()
            vib_error = np.sqrt(np.mean((freq-true_freqs)**2)) 

            epoch_list.append(int(ckpt_model.split("=")[1].replace("-step","")))	
            vib_error_list.append(vib_error)
            print(vib_error_list)
            print(epoch_list)
          

        fig, ax = plt.subplots()
        ax.plot(epoch_list,vib_error_list)
        wandb.log({"vib_error": fig})
     
