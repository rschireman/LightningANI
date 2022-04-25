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


def cp2k_trajectory_to_db(data_dir, db_name):

    data_dict = {}
    db = ase.db.connect(db_name)
    

    coord_frames_list = glob.glob(data_dir + "*-pos-1*")

    for coord_frames in coord_frames_list:
        coord_data = ase.io.read(coord_frames, index=":")
        data_dict[coord_frames] = coord_data

        
    force_energy_frames_list = glob.glob(data_dir + "*-frc-1*")

    for force_energy_frame in force_energy_frames_list:
        force_energy_data = ase.io.read(force_energy_frame, index=":")
        data_dict[force_energy_frame] = force_energy_data
        db.metadata = {'Force Data': force_energy_frame}


    for key,value in data_dict.items():
        print(value)
        

# for i,frame in enumerate(coord_frames):

#     symbols = frame.get_chemical_symbols()

#     energy = force_energy_frames[i].info['E']
    
#     forces = force_energy_frames[i].positions

#     bohr_to_angstrom = np.array(0.529177249)

#     forces = forces / bohr_to_angstrom

#     # print(forces)
#     db.write(frame, cp2k_energy=energy,cp2k_forces=np.array2string(forces),symbols=listToString(symbols))

if __name__ == "__main__":
    cp2k_trajectory_to_db(data_dir="C:\\Users\\ray\\Dropbox\\ML\\datasets\\BTBT_300K_83K_100K_60K_NNP\\", db_name="test.db")