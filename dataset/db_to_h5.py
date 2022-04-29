import h5py
import numpy as np
import ase.db

def db_to_h5(h5_path, ase_db_path):

    hf = h5py.File(h5_path,'w')
    db = ase.db.connect(ase_db_path)
    print("Number of Structures in Database: ", len(db))

    np_energies = np.empty(len(db))
    # TODO 
    # Get number of atoms and declare arrays  
    # hardcode number of atoms for now
    np_forces = np.empty([len(db), 48, 3])
    np_coords = np.empty([len(db), 48, 3])
    np_cell = np.empty([len(db), 3, 3])

    for i,row in enumerate(db.select()):
        symbols = row.get('symbols')  
        cell = row.get('cell')
        np_cell[i] = cell
        energy = row.get('cp2k_energy')
        np_energies[i] = float(energy) 
        forces = row.get('cp2k_forces').replace("]","").replace("[","").strip()
        np_forces[i] = np.array_split(forces.split(),48)
        coords = row.get('positions')
        np_coords[i] = coords
        


    grp = hf.create_group("BTBT") # some unique id for this molecule group
    grp.create_dataset("energies", data=np_energies)
    grp.create_dataset("coordinates", data=np_coords)
    grp.create_dataset("forces", data=np_forces)
    grp.create_dataset("species", data=np.bytes_(symbols))  # species = eg; ['C','H','O'...]
    grp.create_dataset("cell", data=np_cell)
    hf.flush()
    hf.close()

    print(np_forces.shape)
    print(np_coords.shape)
    print(np_energies.shape)


if __name__ == "__main__":
    db_to_h5(h5_path = "test.h5", ase_db_path="test.db")