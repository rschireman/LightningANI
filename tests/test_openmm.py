import torch
import torchani
from typing import Tuple, Optional
from torch import Tensor
# from openmmtorch import TorchForce
from ase.io import read,write

molecule = read("C:\\Users\\ray\\Dropbox\\ML\\datasets\\BTBT_300K_83K_100K_60K_NNP\\btbt_0_1_2.pdb")
print(molecule)
species = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	2,	2,	2,	2]],dtype=torch.long).to('cuda:0')

coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True).float().to('cuda:0')
print(species)
masses = torchani.utils.get_atomic_masses(species).to('cuda:0')
cell = torch.tensor(molecule.get_cell()).float().to('cuda:0')
pbc = torch.tensor([True,True,True]).to('cuda:0')


loaded_compiled_model = torch.jit.load('compiled_model.pt')
energies_single_jit = loaded_compiled_model((species, coordinates)).energies