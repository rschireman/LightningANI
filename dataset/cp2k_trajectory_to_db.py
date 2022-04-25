from operator import index
import ase.db
import ase.io
import os
import numpy as np

def listToString(s): 
    # initialize an empty string
    str1 = " " 
    # return string  
    return (str1.join(s))

os.system("rm *.db")
# cp2k produces bad pdb file

coord_frames = ase.io.read("btbt_0_1_2-pos-1.pdb",index="10000:")
# print(coord_frames)

force_energy_frames = ase.io.read("btbt_0_1_2-frc-1.xyz", index="10000:")
# print(force_energy_frames)


db = ase.db.connect("btbt_0_1_2_300K.db")
print(len(db))
db.metadata = {"Simulation Overview": "P1 structure of BTBT with 2 molecules", "Steps": 100000, "Timestep": 0.2}

for i,frame in enumerate(coord_frames):

    symbols = frame.get_chemical_symbols()

    energy = force_energy_frames[i].info['E']
    
    forces = force_energy_frames[i].positions

    bohr_to_angstrom = np.array(0.529177249)

    forces = forces / bohr_to_angstrom

    # print(forces)
    db.write(frame, cp2k_energy=energy,cp2k_forces=np.array2string(forces),symbols=listToString(symbols))
