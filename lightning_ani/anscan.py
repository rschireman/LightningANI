import torchani
from ase.optimize import BFGSLineSearch
from ase.io import read, write
import torch

loaded_compiled_model = torch.jit.load('compiled_model.pt')
print(loaded_compiled_model)

molecule = read("btbt_0_1_2.pdb")
ase_calc = torchani.ase.Calculator(model=loaded_compiled_model, species=["H", "C", "S"])
molecule.calc = ase_calc

dyn = BFGSLineSearch(molecule)
dyn.run(fmax=1e-3)
print("Optimized Structure: ")
print(molecule.get_total_energy())
species = torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long).unsqueeze(0)
masses = torchani.utils.get_atomic_masses(species)
species = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	2,	2,	2,	2,]],dtype=torch.long)
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True).float()

cell = torch.tensor(molecule.get_cell()).float()
pbc = torch.tensor([True,True,True])
energies = loaded_compiled_model(species, coordinates,cell=cell, pbc=pbc)
hessian = torchani.utils.hessian(coordinates,energies=energies)

freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MWN', unit='cm^-1')
torch.set_printoptions(precision=3, sci_mode=False)
print(freq)
print(modes)

# Note that the normal modes are the COLUMNS of the eigenvectors matrix