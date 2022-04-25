import enum
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
    coord_data = []
    for coord_frames in coord_frames_list:
        coord_data.append(ase.io.read(coord_frames, index=":"))
        db.metadata = {'Coord Data': coord_frames_list}
        
    force_energy_frames_list = glob.glob(data_dir + "*-frc-1*")
    force_energy_data = []
    for force_energy_frame in force_energy_frames_list:
        force_energy_data.append(ase.io.read(force_energy_frame, index=":"))
        db.metadata = {'Force Data': force_energy_frames_list}

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