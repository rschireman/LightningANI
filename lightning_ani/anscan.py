import torchani
from ase.optimize import BFGSLineSearch
from ase.io import read, write
import torch
import numpy as np
from ase.vibrations import Vibrations
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy.polynomial.polynomial as poly
from nnp_delayed_force_training import NNPLightningModelDF
from nnp_data_module import NNPDataModule


mode = 140

# data = NNPDataModule()
# aev_dim = data.get_aev_dim()
# print(aev_dim)
# aev_computer = data.aev_computer
loaded_compiled_model = torch.jit.load('model.pt')
# model = NNPLightningModelDF(aev_computer=aev_computer, aev_dim=aev_dim)
# ckpt_nnp = model.load_from_checkpoint("./runs/epoch=4-step=174.ckpt")  
# nn = torchani.ANIModel([ckpt_nnp.H_network, ckpt_nnp.C_network, ckpt_nnp.S_network])
# test_model = torchani.nn.Sequential(aev_computer, nn)

# loaded_compiled_model = torch.load('model.pt')

molecule = read("btbt_0_1_2.pdb")
ase_calc = torchani.ase.Calculator(model=loaded_compiled_model, species=["H", "C", "S"])
molecule.calc = ase_calc



dyn = BFGSLineSearch(molecule)
dyn.run()
print("Optimized Structure: ")
print(molecule.get_total_energy())
write("opt.pdb", molecule)

# vib = Vibrations(molecule)
# vib.run()
# vib.write_jmol()

species = torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long).unsqueeze(0)
masses = torchani.utils.get_atomic_masses(species)
species = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	2,	2,	2,	2,]],dtype=torch.long)
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True).float()

cell = torch.tensor(molecule.get_cell()).float()
pbc = torch.tensor([True,True,True])
energies = loaded_compiled_model((species, coordinates), cell=cell, pbc=pbc)
hessian = torchani.utils.hessian(coordinates,energies=energies)

freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MWN', unit='cm^-1')
torch.set_printoptions(precision=5, sci_mode=False)
# Note that the normal modes are the COLUMNS of the eigenvectors matrix
print(freq)

polOrder =  3
# best results if numpoints is set to half the total displacements
numPoints = 3
stepsize = 1
freq_cm =  freq[mode-1]

clas_sdene = (freq_cm/8065.5)*(1/27.21)

displacement_vectors = modes[mode-1]

initial_geom = molecule


energies_vs_Q = []

steps = np.arange(0,numPoints+1,1)
for step in steps:
    new_positions = []
    coordinates = torch.from_numpy(initial_geom.get_positions())
    coordinates += step*stepsize*displacement_vectors
    molecule.set_positions(coordinates.numpy())   
    energy = molecule.get_total_energy()
    energies_vs_Q.append((step,energy))
    write("structure_step_" + str(step) + ".pdb", molecule)
    del coordinates

steps = np.arange(-1*numPoints-1,1)
for step in steps:
    new_positions = []
    coordinates = torch.from_numpy(initial_geom.get_positions())
    coordinates += step*stepsize*displacement_vectors
    molecule.set_positions(coordinates.numpy())   
    energy = molecule.get_total_energy()
    energies_vs_Q.append((step,energy))
    write("structure_step_" + str(step) + ".pdb", molecule)
    del coordinates

x = []
y = []
for value in energies_vs_Q:
    x.append(value[0])
    y.append(value[1]/(27.21*8065.5))
plt.scatter(x,y)
plt.savefig("Q_vs_E")    
# plt.show()

## use numpy polyfit to get coefficients, returns highest power first
results = poly.polyfit(x,y,polOrder)
# results =np.flip(results)
print(results)


omega0 = clas_sdene
omegam1 = 0.5/omega0
omegam1 = omegam1/1822.88

amat = np.zeros([len(results),len(results)])

iv=0

for x in range(len(results)-1):
    n=1
    for y in range(len(results)-1):
        if n == 1:
            if (iv+1 <= len(results)-1):
                amat[iv,iv+1]=(iv+1)**0.5*(omegam1)**0.5*results[1]
        if n == 2:
            amat[iv,iv]=2*(iv+0.5)*(0.25*omega0+omegam1*results[2])
            if (iv+2 <= len(results)-1):
                amat[iv,iv+2]=((iv+1)*(iv+2))**0.5*(omegam1*results[2]-0.25*omega0)    
        if n == 3:
            if (iv+1 <= len(results)-1):
                amat[iv,iv+1] = amat[iv,iv+1]+3*(iv+1)**1.5*(omegam1)**(1.5)*results[3]    
            if ((iv+3) <= len(results)-1):
                amat[iv,iv+3]=((iv+1)*(iv+2)*(iv+3))**0.5*(omegam1)**(1.5)*results[3]
        if n == 4:
            amat[iv,iv]=amat[iv,iv]+6*((iv+0.5)**2+0.25)*(omegam1)**2*results[3]
            if ((iv+2 )<= len(results)-1):
                amat[iv,iv+2] = amat[iv,iv+2]+4*(iv+1.5*((iv+1)*(iv+2)))**0.5*(omegam1)**2*results[4]
            if((iv+4) <= len(results)-1):
                amat[iv,iv+4]=((iv+1)*(iv+2)*(iv+3)*(iv+4))**0.5*(omegam1)**2*results[4]
        if n == 5:
            if(iv+1 <= len(results)-1):
                amat[iv,iv+1] = amat[iv,iv+1]+10*((iv+1)**2+0.5)*(iv+1)**0.5*(omegam1)**(2.5)*results[5]

            if((iv+3) <= len(results)-1):
                amat[iv,iv+3] = amat[iv,iv+3]+5*(iv+2)*((iv+1)*(iv+2)*(iv+3))**0.5*(omegam1)**(2)*results[5]

            if((iv+5) <= len(results)-1):
                amat[iv,iv+5]=((iv+1)*(iv+2)*(iv+3)*(iv+4)*(iv+5))**0.5*(omegam1)**(2.5)*results[5]
        if n == 6:
            amat[iv,iv]=amat[iv,iv]+5*(iv+0.5)*(4*(iv+0.5)**2+5)*(omegam1)**3*results[6]   
            if ((iv+2 )<= len(results)-1):
                amat[iv,iv+2]=amat[iv,iv+2]+15*((iv+1.5)**2+0.75)*((iv+1)*(iv+2))**(0.5)*(omegam1)**3*results[6]
            if((iv+4) <= len(results)-1):
                amat[iv,iv+4] = amat[iv,iv+4]+6*(iv+2.5)*((iv+1)*(iv+2)*(iv+3)*(iv+4))**0.5*(omegam1)**3*results[6]
            if((iv+6) <= len(results)-1):
                amat[iv,iv+6]=5*((iv+1)*(iv+2)*(iv+3)*(iv+4)*(iv+5)*(iv+6))**0.5*(omegam1)**3*results[6]
        if n == 7: 
            if(iv+1 <= len(results)-1):
                amat[iv,iv+1] = amat[iv,iv+1]+35*(iv+1)*((iv+1)**2+2)*(iv+1)**0.5*(omegam1)**(3.5)*results[7]
            if(iv+3 <= len(results)-1):
                amat[iv,iv+3] = amat[iv,iv+3]+21*((iv+2)**2+1)*((iv+1)*(iv+2)*(iv+3))**0.5*(omegam1)**(3.5)*results[7]
            if(iv+5 <= len(results)-1):
                amat[iv,iv+5] = amat[iv,iv+5]+7*(iv+3)*((iv+1)*(iv+2)*(iv+3)*(iv+4)*(iv+5))**0.5*(omegam1)**(3.5)*results[7]
            if(iv+7 <= len(results)-1):
                amat[iv,iv+7] = ((iv+1)*(iv+2)*(iv+3)*(iv+4)*(iv+5)*(iv+6)*(iv+7))**0.5*(omegam1)**(3.5)*results[7]
        if n == 8:
            amat[iv,iv]=amat[iv,iv]+((35/8)*(16.0*(iv+0.5)**4+56*(iv+0.5)**2+9))*(omegam1)**4*results[8]
            if ((iv+2 )<= len(results)-1):
                amat[iv,iv+2]=amat[iv,iv+2]+56*(iv+1.5)*((iv+1.5)**2+11/4)*((iv+1)*(iv+2))**0.5*(omegam1)**4*results[8]
            if ((iv+4 )<= len(results)-1):
                amat[iv,iv+4] = amat[iv,iv+4]+28*((iv+2.5)**2+5/4)*((iv+1)*(iv+2)*(iv+3)*(iv+4))**0.5*(omegam1)**4*results[8]
            if ((iv+6 )<= len(results)-1):    
                amat[iv,iv+6]=amat[iv,iv+6]+8*(iv+7/2)*((iv+1)*(iv+2)*(iv+3)*(iv+4)*(iv+5)*(iv+6))**0.5*(omegam1)**4*results[8]
            if ((iv+8 )<= len(results)-1):
                amat[iv,iv+8]=((iv+1)*(iv+2)*(iv+3)*(iv+4)*(iv+5)*(iv+6)*(iv+7)*(iv+8))**0.5*(omegam1)**4*results[8]

        n+=1
    iv+=1

amat = amat + amat.T - np.diag(np.diag(amat))

w, v = LA.eig(amat)
w.sort()
print(amat)


t00 = 27.21*8065.5*w[1]
t11 = 27.21*8065.5*w[2]
t22 = 27.21*8065.5*w[3]

w01 = t11-t00
print("Fundamental Anharmonic Frequency:  ", w01 , " cm-1")