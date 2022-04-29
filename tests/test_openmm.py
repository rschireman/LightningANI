import torch
import torchani
from typing import Tuple, Optional
from torch import Tensor
from openmmtorch import TorchForce
from ase.io import read,write
from openmm.app import *
from openmm import *
from openmm.unit import *

class CustomModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.jit.load('compiled_model.pt')
        # self.model = torchani.models.ANI1x(periodic_table_index=True).double()
        # self.model = torchani.models.ANI1x(periodic_table_index=True)[0].double()
        # self.model = torchani.models.ANI1ccx(periodic_table_index=True).double()

    def forward(self, species: Tensor, coordinates: Tensor, return_forces: bool = False,
                return_hessians: bool = False) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if return_forces or return_hessians:
            coordinates.requires_grad_(True)

        energies = self.model(species, coordinates)

        forces: Optional[Tensor] = None  # noqa: E701
        hessians: Optional[Tensor] = None
        if return_forces or return_hessians:
            grad = torch.autograd.grad([energies.sum()], [coordinates], create_graph=return_hessians)[0]
            assert grad is not None
            forces = -grad
            if return_hessians:
                hessians = torchani.utils.hessian(coordinates, forces=forces)
        return energies, forces, hessians


molecule = read("btbt_0_1_2.pdb")

custom_model = CustomModule()
species = torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long).unsqueeze(0)
print(species)
masses = torchani.utils.get_atomic_masses(species).to('cuda:0')
species = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	2,	2,	2,	2]],dtype=torch.long)
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True).float()

energies, forces, hessians = custom_model(species, coordinates, True, True)
print(energies,forces,hessians)
