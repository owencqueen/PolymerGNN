import os, pickle
import qml
import numpy as np
import pandas as pd
from tqdm import tqdm
from polymerlearn.utils.comparison_rep.element_info import *

import ase
from dscribe.descriptors import SOAP
from dscribe.descriptors import MBTR as MBTR_

import pyarrow.feather as feather

def Atom_XYZ(xyz_list):
    atoms = []
    charges = []
    coords = np.empty((0,3), float)
    
    for j in xyz_list:
        while j.startswith(" "):
            j = j[1:]
        while "\t" in j:
            j = j.replace("\t", " ")
        while "  " in j:
            j = j.replace("  ", " ")
        temp = j[:-1].split(" ")
        print(temp)

        atoms.append(temp[0])
        coords = np.append(coords, np.array([[temp[1], temp[2], temp[3]]]), axis=0)
        charges.append(AtomicNumber(temp[0]))
    
    return atoms, charges, coords

def CM(new_struct):
    '''
    new_struct: one chunk of XYZ file
    '''
    atoms, charges, coords = Atom_XYZ(new_struct)
    mol = qml.representations.generate_coulomb_matrix(nuclear_charges=charges, 
                                                        coordinates=coords, 
                                                        size=len(atoms), 
                                                        sorting='row-norm'
                                                        )
    return mol

def BOB(new_struct):
    '''
    new_struct: one chunk of XYZ file
    '''
    atoms, charges, coords = Atom_XYZ(new_struct)
    atom_dict = {}

    for j in atoms:
        if j not in atom_dict:
            atom_dict[j] = 1
        else:
            atom_dict[j] += 1

    mol = qml.representations.generate_bob(nuclear_charges=charges, 
                                            coordinates=coords, 
                                            atomtypes=np.unique(np.asarray(atoms)),
                                            size=len(atoms), 
                                            asize=atom_dict
                                            )
    return mol

def mySOAP(new_struct):
    '''
    new_struct: one chunk of XYZ file
    '''

    atoms, charges, coords = Atom_XYZ(new_struct)
    species = np.unique(np.asarray(atoms))

    soap = SOAP(species=species, 
                periodic=False, 
                rcut=3.0,
                nmax=5,
                lmax=4
                )

    mol = soap.create(system=ase.Atoms(positions=coords, numbers=charges),
                positions=coords,
                n_jobs=1,
                )

    return mol.flatten()

def MBTR(new_struct):

    atoms, charges, coords = Atom_XYZ(new_struct)
    species = np.unique(np.asarray(atoms))

    mbtr = MBTR_(species=species,
                k1={
                    "geometry": {"function": "atomic_number"},
                    "grid": {"min": 0, "max": 8, "n": 100, "sigma": 0.1},
                    },
                k2={
                    "geometry": {"function": "inverse_distance"},
                    "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                    "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
                    },
                k3={
                    "geometry": {"function": "cosine"},
                    "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
                    "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
                    },
                periodic=False,
                normalization="l2_each", 
                flatten=True
                )
    mol = mbtr.create(system=ase.Atoms(positions=coords, numbers=charges),
                        n_jobs=1) # Hardcoded number of jobs

    return mol 

def get_one_top_xyz(filename):
    '''
    Gets the top XYZ chunk for the top conformer in file. Ready to input to representation generator
    '''
    with open(filename) as f:
        ff = f.readlines()
    natoms = int(ff[0])

    end = natoms + 1
    file_chunk = ff[2:end]
    return file_chunk

def screen_build(all_AG, 
        xyz_loc = '../../../Structures/AG/xyz', 
        rep_dir_loc = '../../../Representations',
        reps_to_screen = ['CM', 'SOAP', 'BOB']):
    '''
    Screens all acids/glycols in a dataframe, builds representations of a given type
    '''

    gen_dict = {
        'CM': CM,
        'SOAP': mySOAP,
        'BOB': BOB,
        'MBTR': MBTR
    }

    xyzpath = lambda x: os.path.join(xyz_loc, x)

    for rep in reps_to_screen:
        print('REP', rep)
        for ag in all_AG:
            rloc_rep = os.path.join(rep_dir_loc, rep, 'AG')
            pickle_path = os.path.join(rloc_rep, ag.lower() + '.pickle')
        
            if not os.path.exists(pickle_path):
                fchunk = get_one_top_xyz(xyzpath(ag + '.xyz'))
                F = gen_dict[rep]
                mol = F(fchunk)
                pickle.dump(mol, open(pickle_path, 'wb'))
                #feather.write_feather(mol, feather_path)


if __name__ == '__main__':

    data = pd.read_csv('../../../dataset/pub_data.csv')
    ac = (20,33); gc = (34,46)
    acid_names = [c[1:] for c in data.columns[ac[0]:ac[1]].tolist() if '95% trans' not in c]
    glycol_names = [c[1:] for c in data.columns[gc[0]:gc[1]].tolist()]

    full_names = acid_names + glycol_names
    xyz_loc = '../../../Structures/AG'

    screen_build(full_names)