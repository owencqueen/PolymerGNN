import os, glob, random
import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from polymerlearn.utils.xyz2mol import int_atom, xyz2mol

def read_xyz_file_top_conformer(filename, look_for_charge=True):
    """
    Reads an xyz file and parses the first conformer at the top
    """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0

    with open(filename, "r") as file:

        for line_number, line in enumerate(file):
            #print(line)
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            elif line_number >= num_atoms + 2:
                break # Break after first conformation
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates

def convert_xyz_to_mol(filename):
    atoms, charge, xyz_coordinates = read_xyz_file_top_conformer(filename)

    mols = xyz2mol(
            atoms, 
            xyz_coordinates,
            charge = charge,
            use_graph=True,
            allow_charged_fragments=True,
            embed_chiral=False,
            use_huckel=False)

    return mols[0]

# Citation (C) = https://towardsdatascience.com/practical-graph-neural-networks-for-molecular-machine-learning-5e6dee7dc003
def get_atom_features(mol):
    '''
    Make atom features
        - Can be made more robust with background work
    '''
    # Cite: C
    features = []
    
    for atom in mol.GetAtoms():
        #atomic_number.append(atom.GetAtomicNum())
        charge = atom.GetFormalCharge()
        degree = atom.GetDegree()
        mass = atom.GetMass()
        is_aromatic = atom.GetIsAromatic()
        anum = atom.GetAtomicNum()
        explicit_hs = atom.GetNumExplicitHs()
        num_valence = atom.GetTotalValence()
        num_rad_electrons = atom.GetNumRadicalElectrons()

        #features.append([charge, degree, mass, is_aromatic, anum, explicit_hs, num_rad_electrons])
        features.append([charge, degree, mass, is_aromatic, explicit_hs, num_valence])
        #num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        
    return torch.tensor(features).float()

def get_edge_index(mol, get_edge_attr = False):
    '''
    Gets edge index for a molecule
    '''
    # Cite: C
    row, col = [], []
    edge_attr = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

        if get_edge_attr:
            btype = bond.GetBondTypeAsDouble()
            inring = int(bond.IsInRing())
            edge_attr.append([btype, inring])

    eidx = torch.tensor([row, col], dtype=torch.long)

    if get_edge_attr:
        edge_attr = torch.tensor(edge_attr).float()
        return eidx, edge_attr
        
    return eidx

def prepare_dataloader_graph_AG(
        A_mol_list, 
        G_mol_list, 
        Y = None, 
        add_A = None, 
        add_G = None, 
        get_edge_attr = False,
        device = None,
        atom_feat=None):
    '''
    Prepares a dataloader given a list of molecules

    Args:
        A_mol_list (list of lists): List of lists of RDKit Mol objects for each sample.
            Should look something like:
                [[A_11, A_12], [A_21, A_22, A_23, A_24], ..., [A_n1]]
        G_mol_list (list of lists): Same as A_mol_list but for Glycols
        Y (iterable, optional): Y (ground truth values) for each sample
        add_A (dict of lists, optional): Dictionary of lists of values to add for each acid. 
            Should be keyed on strings of names of variables with list values 
            corresponding to numerical values to add to the Data objects in the
            DataLoader.
        add_G (dict of lists, optional): Same as add_A but for glycols.
        atom_feat (Callable[[RdKit.Mol], torch.Tensor], optional): Function to output the 
            feature matrix for a given molecule.
    '''

    if atom_feat == None:
        atom_feat = get_atom_features

    assert len(A_mol_list) == len(G_mol_list), 'A and G mol (RDKit) lists not same length'

    # Cite: C
    data_list = []

    i = 0 # Counts total Amols, Gmols that we've added (i.e. whole samples)

    for Amols, Gmols in zip(A_mol_list, G_mol_list):

        acid_graphs = []
        j = 0 # Counts total number of acids for this sample
        for Amol in Amols:
            Ax = atom_feat(Amol)
            if get_edge_attr:
                Aedge_index, Aedge_attr = get_edge_index(Amol, get_edge_attr=True)
            else:
                Aedge_index = get_edge_index(Amol)

            # Support for additional arguments:
            add_args = {}
            if add_A is not None:
                for key, val in add_A.items():
                    add_args[key] = torch.tensor([val[i][j]]).float().to(device)

            if Y is not None:
                add_args['y'] = torch.tensor(Y[i]).float().to(device)
            if get_edge_attr:
                add_args['edge_attr'] = Aedge_attr.to(device)

            acid_data = Data(
                x=Ax.to(device), 
                edge_index = Aedge_index.to(device), 
                **add_args)
            # All acid data should be in device
            acid_graphs.append(acid_data)
            j += 1


        glycol_graphs = []
        j = 0 # Counts total number of glycols for this sample
        for Gmol in Gmols:
            Gx = atom_feat(Gmol)

            if get_edge_attr:
                Gedge_index, Gedge_attr = get_edge_index(Gmol, get_edge_attr=True)
            else:
                Gedge_index = get_edge_index(Gmol)

            add_args = {}
            if add_A is not None:
                for key, val in add_G.items():
                    add_args[key] = torch.tensor([val[i][j]]).float()

            # Support for adding multiple
            if Y is not None:
                add_args['y'] = torch.tensor(Y[i]).float().to(device)
            if get_edge_attr:
                add_args['edge_attr'] = Gedge_attr.to(device)

            glycol_data = Data(
                x=Gx.to(device), 
                edge_index=Gedge_index.to(device), 
                **add_args)
            # All glycol data should be in device
            glycol_graphs.append(glycol_data)
            j += 1

        data_list.append((acid_graphs, glycol_graphs))

        i += 1

    return data_list

def graph_dataloader_z_pos(
        A_charge_coords, 
        G_charge_coords,
        Y,
    ):
    '''
    Gets dataloader by nuclear charges (z) and positions (pos)
    - Useful for SchNet architecture
    '''

    data_list = []

    for A, G in zip(A_charge_coords, G_charge_coords):
        # List all atoms and their coordinates:
        acid_data_list = []
        for z, pos in A:
            adata = Data(
                z = torch.as_tensor(z).long(),
                pos = torch.as_tensor(pos),
            )
            acid_data_list.append(adata)

        glycol_data_list = []
        for z, pos in G:
            gdata = Data(
                z = torch.as_tensor(z).long(),
                pos = torch.as_tensor(pos),
            )
            glycol_data_list.append(gdata)
            
        data_list.append((acid_data_list, glycol_data_list))

    return data_list


def get_AG_info(data, ac = (20,33), gc = (34,46)):
    '''
    Gets acid/glycol info from a dataframe containing input in the Eastman fashion
    '''

    ac_tuple = False
    gc_tuple = False
    if type(ac) == tuple:
        ac_tuple = True
    
    if type(gc) == tuple:
        gc_tuple = True

    # Decompose the data into included names
    if ac_tuple:
        acid_names = pd.Series([c[1:] for c in data.columns[ac[0]:ac[1]].tolist()])
    else:
        acid_names = pd.Series([c[1:] for c in data[ac].columns.tolist()])
    if gc_tuple:     
        glycol_names = pd.Series([c[1:] for c in data.columns[gc[0]:gc[1]].tolist()])
    else:
        glycol_names = pd.Series([c[1:] for c in data[gc].columns.tolist()])

    # Holds all names of acids and glycols
    acid_included = []
    glycol_included = []

    # Keep track of percents in each acid, glycol
    acid_pcts = []
    glycol_pcts = []

    # Get relevant names and percentages of acid/glycols
    for i in range(data.shape[0]):

        if ac_tuple:
            acid_hit = (data.iloc[i,ac[0]:ac[1]].to_numpy() > 0)
        else:
            acid_hit = (data[ac].iloc[i].to_numpy() > 0)
        if gc_tuple:
            glycol_hit = (data.iloc[i,gc[0]:gc[1]].to_numpy() > 0)
        else:
            glycol_hit = (data[gc].iloc[i].to_numpy() > 0)

        # Add to percentage lists:
        if ac_tuple:
            acid_pcts.append(data.iloc[i,ac[0]:ac[1]][acid_hit].tolist())
        else:
            acid_pcts.append(data[ac].iloc[i][acid_hit].tolist())
        if gc_tuple:
            glycol_pcts.append(data.iloc[i,gc[0]:gc[1]][glycol_hit].tolist())
        else:
            glycol_pcts.append(data[gc].iloc[i][glycol_hit].tolist())

        acid_pos = acid_names[np.argwhere(acid_hit).flatten()].tolist()
        glycol_pos = glycol_names[np.argwhere(glycol_hit).flatten()].tolist()

        acid_included.append(acid_pos)
        glycol_included.append(glycol_pos)

    return acid_included, glycol_included, acid_pcts, glycol_pcts

def list_mask(L, mask):
    '''
    Transform a list with a boolean mask
    Args:
        L (list): List to be masked
        mask (iterable): Mask to apply on list L
    '''
    return [L[int(i)] for i in range(len(mask)) if mask[i]]

def split_validation(train_mask, val_num):

    train_mask, val_mask = train_test_split(train_mask, test_size=val_num)

    return train_mask, val_mask



# base_structure_dir = os.path.join('/home/sai/Eastman_Project',
#     'ReadyToEnsemble',
#     'Structures',
#     'AG',
#     'xyz')
base_structure_dir = os.path.join('..',
    'Structures',
    'AG',
    'xyz')

class GraphDataset:
    '''
    Generates a graph dataset based on input from Eastman data

    Args:
        data (pd.DataFrame): DataFrame containing Eastman data format.
        Y_target (pd.DataFrame): Target values to predict with the dataset.
        structure_dir (str, optional): Location of base structures, i.e. xyz files.
            Assumes that all xyz files are named exactly like the molecules in the
            input data given from Eastman.
        add_features (np.array or torch.Tensor): Additional features to be added to
            each sample's final embedding before processing. These are GLOBAL features
            and are not per-atom features (TODO: make per-atom features).
        ac (tuple, len 2): Bottom and top bounds for all acids on the table given as
            data. Must be column indices.
        ac (list, str): List of columns for all acids
        gc (tuple, len 2): Bottom and top bounds for all glycols on the table given as
            data. Must be column indices.
        gc (list, str): List of columns for all glycols
        test_size (float): Proportion of data to be used as testing data (if using 
            train/test split).
        val_size (float): Proportion of the data to be used as validation data. If None,
            does not make a validation split.
        get_edge_attr (bool): Option to use predefined edge features on the graphs.
        bound_filter (list, length 2): Filters the dataset by some bound on Y value, 
            i.e. controls for outliers
            TODO: implementation for multiple Y values
        exclude_inds (list of ints): List of indices to exclude in the dataframe.
        device (str): Device name at which to run torch calculations on. Supports GPU.
        standard_scale (bool, optional): Whether to perform standard scaling for the 
            add_features at split time. Cannot be done as a preprocessing step (i.e. 
            incorporated to add_features) because the scales should depend only on trainig
            data. Therefore, scaling parameters must be recomputed at each split. 
            Default False.
        ss_mask (list/ndarray of bools): Mask over the variables that need to be standard
            scaled. May be used if you want to scale some variables (like AN, OHN) but not
            others (like Mw).
    '''

    def __init__(self,
            data,
            Y_target,
            structure_dir = base_structure_dir,
            add_features = None,
            ac = (20,33),
            gc = (34,46),
            test_size = 0.25,
            val_size = None,
            get_edge_attr  = False,
            bound_filter = None,
            exclude_inds = None,
            device = None,
            standard_scale = False,
            ss_mask = None,
            z_pos_loaders = False,
            kelvin=False
        ):

        self.add_features = add_features
        self.get_edge_attr = get_edge_attr
        self.val_size = val_size
        self.test_size = test_size
        self.device = device
        self.standard_scale = standard_scale
        self.ss_mask = None
        self.z_pos_loaders = z_pos_loaders
        if self.add_features is not None:
            if self.add_features.ndim == 1:
                self.add_features = self.add_features[:, np.newaxis] # Turn to column vector

        if type(ac) == tuple:
            self.ac_tuple = True
        else:
            self.ac_tuple = False
        
        if type(gc) == tuple:
            self.gc_tuple = True
        else:
            self.gc_tuple = False
        
        # Decompose the data into included names
        if self.ac_tuple:
            self.acid_names = pd.Series([c[1:] for c in data.columns[ac[0]:ac[1]].tolist()])
        else:
            self.acid_names = pd.Series([c[1:] for c in data[ac].columns.tolist()])
        if self.gc_tuple:     
            self.glycol_names = pd.Series([c[1:] for c in data.columns[gc[0]:gc[1]].tolist()])
        else:
            self.glycol_names = pd.Series([c[1:] for c in data[gc].columns.tolist()])

        # Holds all names of acids and glycols
        acid_included = []
        glycol_included = []

        # Keep track of percents in each acid, glycol
        acid_pcts = []
        glycol_pcts = []

        # Get relevant names and percentages of acid/glycols
        for i in range(data.shape[0]):

            if self.ac_tuple:
                acid_hit = (data.iloc[i,ac[0]:ac[1]].to_numpy() > 0)
            else:
                acid_hit = (data[ac].iloc[i].to_numpy() > 0)
            if self.gc_tuple:
                glycol_hit = (data.iloc[i,gc[0]:gc[1]].to_numpy() > 0)
            else:
                glycol_hit = (data[gc].iloc[i].to_numpy() > 0)

            # Add to percentage lists:
            if self.ac_tuple:
                acid_pcts.append(data.iloc[i,ac[0]:ac[1]][acid_hit].tolist())
            else:
                acid_pcts.append(data[ac].iloc[i][acid_hit].tolist())
            if self.gc_tuple:
                glycol_pcts.append(data.iloc[i,gc[0]:gc[1]][glycol_hit].tolist())
            else:
                glycol_pcts.append(data[gc].iloc[i][glycol_hit].tolist())

            acid_pos = self.acid_names[np.argwhere(acid_hit).flatten()].tolist()
            glycol_pos = self.glycol_names[np.argwhere(glycol_hit).flatten()].tolist()

            acid_included.append(acid_pos)
            glycol_included.append(glycol_pos)

        # Read all xyz files into generators, get lowest energy conformation (index 0)

        self.acid_mols = []
        self.glycol_mols = []

        if self.z_pos_loaders: # Load the Z-pos structure

            for i in range(len(acid_included)):
                A_sub = []
                for j in range(len(acid_included[i])):
                    Acharge, _, Acoords = read_xyz_file_top_conformer(os.path.join(structure_dir, acid_included[i][j] + '.xyz'))
                    A_sub.append((Acharge, Acoords))

                self.acid_mols.append(A_sub)

                G_sub = []
                for j in range(len(glycol_included[i])):
                    Gcharge, _, Gcoords = read_xyz_file_top_conformer(os.path.join(structure_dir, glycol_included[i][j] + '.xyz'))
                    G_sub.append((Acharge, Acoords))
                
                self.glycol_mols.append(G_sub)

        else:

            for i in range(len(acid_included)):
                self.acid_mols.append(
                    [convert_xyz_to_mol(os.path.join(structure_dir, acid_included[i][j] + '.xyz')) for j in range(len(acid_included[i]))]
                )

                self.glycol_mols.append(
                    [convert_xyz_to_mol(os.path.join(structure_dir, glycol_included[i][j] + '.xyz')) for j in range(len(glycol_included[i]))]
                )

        # Set Y (target)
        Y = data.loc[:,Y_target]
        if kelvin:
            if 'Tg' in Y_target:
                Y['Tg'] = Y['Tg'] + 273.15


        # Mask data for empty entries
        non_nan_mask = Y.notna()

        self.exclude_inds = exclude_inds

        if self.exclude_inds is not None:
            # Set up exclude-by-index mask:
            inds_lookup = set(self.exclude_inds)
            exclude_by_index = [not (i in inds_lookup) for i in range(Y.shape[0])]
        else:
            exclude_by_index = [True] * Y.shape[0]

        self.bound_filter = bound_filter
        
        if self.bound_filter is not None:
            non_nan_mask = non_nan_mask & (Y > bound_filter[0]) & (Y < bound_filter[1]) & exclude_by_index
        
        non_nan_mask['res_bool'] = False
        non_nan_mask.loc[non_nan_mask[Y_target].all(1), 'res_bool'] = True

        non_nan_mask = non_nan_mask['res_bool'].values
        self.total_samples = sum(non_nan_mask)

        # Mask acid, glycols:
        self.acid_mols = list_mask(self.acid_mols, non_nan_mask)
        self.glycol_mols = list_mask(self.glycol_mols, non_nan_mask)

        # Mask Y:
        self.Y = Y[non_nan_mask].values

        # Mask data:
        self.data = data.loc[non_nan_mask,:]

        # Mask percentages of acids and glycols:
        self.acid_pcts = list_mask(acid_pcts, non_nan_mask)
        self.glycol_pcts = list_mask(glycol_pcts, non_nan_mask)

        # Mask additional features:
        if self.add_features is not None: # Mask additional features, if needed
            self.add_features = list_mask(self.add_features, non_nan_mask)

        # Make dataloader
        rangeL = list(range(len(self.acid_mols)))
        train_mask, test_mask = train_test_split(rangeL, test_size = test_size)

        # Support validation splitting:
        if self.val_size is not None:
            adj_valsize = (self.val_size) / (self.val_size + (1 - self.test_size))
            train_mask, val_mask = train_test_split(train_mask, test_size = adj_valsize)
        else:
            val_mask = None

        self.split_by_indices(train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)

        # if self.standard_scale:
        #     add_features = np.array(self.add_features)

        print(Y)

    def get_train_batch(self, size: int):
        '''
        Perform manual batching of graph dataset

        Args:
            size (int): Size of the batch to be retrieved
        '''

        # Randomly sample the training data
        sample_inds = random.sample(list(np.arange(len(self.Ytrain))), k = size)

        if self.add_features is None:
            return [self.train_data[i] for i in sample_inds], torch.tensor([self.Ytrain[i] for i in sample_inds]).float(), None
        else:
            train_masked = [self.train_data[i] for i in sample_inds]
            Y_masked = torch.tensor(np.array([self.Ytrain[i] for i in sample_inds])).float()
            add_masked = [self.add_train[i] for i in sample_inds]
            return train_masked, Y_masked, add_masked

    def get_test(self, test_inds = None):
        '''
        Get test data nbased on the current internal split

        Args:
            test_inds (any, optional): If provided, is returned with all other
                values.
        '''
        if test_inds is not None:
            return self.test_data, \
                torch.tensor(np.array(self.Ytest)).float().to(self.device), \
                self.add_test, \
                test_inds
        else:
            return self.test_data, \
                torch.tensor(np.array(self.Ytest)).float().to(self.device), \
                torch.tensor(self.add_test).float().to(self.device)

    def get_validation(self):
        '''
        Get the validation data based on current internal splits
        '''
        if self.val_data is None: # No validation split present
            return None
        else:
            # val_data should already be on device
            return self.val_data, \
                torch.tensor(self.Yval).float().to(self.device), \
                torch.tensor(self.add_val).float().to(self.device)

    def Kfold_CV(self, folds, val = False, val_size = None):
        '''
        Generator that wraps SKLearn's K-fold cross validation

        Note that the yield of this function is the testing data, you must perform batching
            of the dataset object (get_train_batch) to get the training data. Rationale 
            behind this is to allow you to train multiple epochs while repeatedly batching
            the training data under one iteration of the Kfold_CV function. 
        
        Args:
            folds (int): Number of folds for the cross validation.
            val (bool): Should be set to True if using validation split.
            val_size (float, 0<=x<=1): Proportion of whole dataset that is used for
                validation split on each fold.

        Yield:
            tuple(tuple(train_data, Ytrain, add_train), tuple(test_data, Ytest, add_test))
        '''

        inds = np.arange(self.total_samples)
        kfold = KFold(n_splits=folds, shuffle = True)

        for train_inds, test_inds in kfold.split(inds):
            if val: # Split the validation:
                val_size = val_size if val_size is not None else self.val_size
                val_adj = val_size / (val_size + (len(train_inds) / len(test_inds)))
                train_inds, val_inds = train_test_split(train_inds, test_size = val_adj)
                self.val_inds = val_inds # No set if val is not true
            else:
                val_inds = None

            self.split_by_indices(train_mask = train_inds, test_mask = test_inds, 
                val_mask = val_inds)
            self.train_inds = train_inds
            self.test_inds = test_inds
            yield self.get_test(test_inds)

    def make_dataloader_by_mask(self, mask):
        '''
        Makes an internal dataloader based on some given list of indices
            - Not technically a mask
            - Makes no internal updates
        '''

        # Perform all train masking: -------------------------------
        Ymask = [self.Y[int(i)] for i in mask]
        mask_Amols = [self.acid_mols[int(i)] for i in mask]
        mask_Gmols = [self.glycol_mols[int(i)] for i in mask]


        if self.z_pos_loaders:
            data = graph_dataloader_z_pos(mask_Amols, mask_Gmols, Ymask)
        else:
            add_A = {'pct': [self.acid_pcts[i] for i in mask]}
            add_G = {'pct': [self.glycol_pcts[i] for i in mask]}

            data = prepare_dataloader_graph_AG(mask_Amols, mask_Gmols, Ymask,
                            add_A = add_A, add_G = add_G, device = self.device)

        return data

    def get_additional_by_mask(self, mask):
        '''
        Get additional elements based on given list of indices
        '''
        return [self.add_features[int(i)] for i in mask]

    def get_Y_by_mask(self, mask):
        return [self.Y[int(i)] for i in mask]

    def split_by_indices(self, train_mask, test_mask, val_mask = None):
        '''
        Resets train_data, test_data, Ytrain, and Ytest for internal use
        Splits the data given train_mask and test_mask and stores dataloaders in
            self.train_data and self.test_data
        '''

        self.train_mask = train_mask
        self.test_mask = test_mask
        self.val_mask = val_mask

        self.Ytrain = [self.Y[int(i)] for i in train_mask]
        self.Ytest = [self.Y[int(i)] for i in test_mask]

        self.train_data = self.make_dataloader_by_mask(train_mask)
        self.test_data = self.make_dataloader_by_mask(test_mask)

        if self.val_mask is not None:
            self.Yval = [self.Y[int(i)] for i in val_mask]
            self.val_data = self.make_dataloader_by_mask(val_mask)
        else:
            self.val_data = None

        # Perform all test masking:  --------------------------------
        # self.Ytest = [self.Y[int(i)] for i in test_mask]
        # self.test_Amols = [self.acid_mols[int(i)] for i in test_mask]
        # self.test_Gmols = [self.glycol_mols[int(i)] for i in test_mask]

        # add_A = {'pct': [self.acid_pcts[i] for i in test_mask]}
        # add_G = {'pct': [self.glycol_pcts[i] for i in test_mask]}
        # self.test_data = prepare_dataloader_graph_AG(self.test_Amols, self.test_Gmols, self.Ytest,
        #                 add_A = add_A, add_G = add_G)

        if self.add_features is not None:
            self.add_train = [self.add_features[int(i)] for i in train_mask]
            self.add_test = [self.add_features[int(i)] for i in test_mask]
            if self.val_mask is not None:
                self.add_val = [self.add_features[int(i)] for i in val_mask]

            if self.standard_scale:
                if self.ss_mask is not None:
                    self.add_train = np.array(self.add_train)
                    self.add_test = np.array(self.add_test)

                    # Scale only the variables masked in by the ss_mask:
                    ss = StandardScaler().fit(self.add_train[:,self.ss_mask])
                    self.add_train[:,self.ss_mask] = ss.transform(self.add_train[:,self.ss_mask])
                    self.add_test[:,self.ss_mask] = ss.transform(self.add_test[:,self.ss_mask])

                    if self.val_mask is not None:
                        self.add_val[:,self.ss_mask] = ss.transform(self.add_val[:,self.ss_mask]) 
                        self.add_val = list(self.add_val)

                    self.add_train = list(self.add_train)
                    self.add_test = list(self.add_test)

                else:
                    # Standard scale based on new split
                    ss = StandardScaler().fit(np.array(self.add_train))
                    #   Fit to only training data, as is customary
                    self.add_train = list(ss.transform(self.add_train))
                    self.add_test = list(ss.transform(self.add_test))
                    if self.val_mask is not None:
                        self.add_val = list(ss.transform(self.add_val))

        else:
            self.add_train = None
            self.add_test = None
            if self.val_mask is not None:
                self.add_val = None

# Misc. testing functions:
def test_xyz2mol():
    print(read_xyz_file_top_conformer(os.path.join(base_structure_dir, 'IPA.xyz')))

    mol = convert_xyz_to_mol(os.path.join(base_structure_dir, 'IPA.xyz'))

    print([a.GetAtomicNum() for a in mol.GetAtoms()])

def test_dataset():

    data = pd.read_csv('../../dataset/pub_data.csv')

    print(data)

    base_structure_dir = os.path.join('..', '..',
        'Structures',
        'AG',
        'xyz'
    )

    dataset = GraphDataset(data = data, Y_target=['IV'], z_pos_loaders = True,
        structure_dir = base_structure_dir, kelvin=True)

    #print(dataset.Y)

if __name__ == '__main__':

    test_dataset()
