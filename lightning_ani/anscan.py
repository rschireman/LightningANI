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
dyn.run()
print("Optimized Structure: ")
print(molecule.get_total_energy())