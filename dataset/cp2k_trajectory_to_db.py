import glob
import ase.db
import ase.io
import os
import numpy as np

def listToString(s): 
    # initialize an empty string
    str1 = " " 
    # return string  
    return (str1.join(s))


def cp2k_trajectory_to_db(data_dir):

    os.system("rm *.db")

    data_dict = {}

    coord_frames_list = glob.glob(data_dir + "*-pos-1*")

    for coord_frames in coord_frames_list:
        coord_data = ase.io.read(coord_frames, index=":")
        data_dict[coord_frames] = coord_data

        
    force_energy_frames_list = glob.glob(data_dir + "*-frc-1*")

    for force_energy_frame in force_energy_frames_list:
        force_energy_data = ase.io.read(force_energy_frame, index=":")
        data_dict[force_energy_frame] = force_energy_data


    db = ase.db.connect("btbt_0_1_2_300K.db")
# print(len(db))
# db.metadata = {"Simulation Overview": "P1 structure of BTBT with 2 molecules", "Steps": 100000, "Timestep": 0.2}

# for i,frame in enumerate(coord_frames):

#     symbols = frame.get_chemical_symbols()

#     energy = force_energy_frames[i].info['E']
    
#     forces = force_energy_frames[i].positions

#     bohr_to_angstrom = np.array(0.529177249)

#     forces = forces / bohr_to_angstrom

#     # print(forces)
#     db.write(frame, cp2k_energy=energy,cp2k_forces=np.array2string(forces),symbols=listToString(symbols))

if __name__ == "__main__":
    cp2k_trajectory_to_db(data_dir="/home/ray/Desktop/BTBT_300K_83K_100K_60K_NNP/btbt_0_1_2_100K_60k/")