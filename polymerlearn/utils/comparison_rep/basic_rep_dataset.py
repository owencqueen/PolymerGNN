import torch
import os, pickle
import numpy as np
import pandas as pd

from polymerlearn.utils.graph_prep import get_AG_info, list_mask
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as R2, mean_absolute_error as MAE
from sklearn.utils import shuffle

base_rep_dir = os.path.join('../../..',
    'Representations')

def load_pickle(dir, mol_name):
    f = os.path.join(dir, mol_name.lower() + '.pickle')
    rep = pickle.load(open(f, 'rb'))

    return rep

class BinaryDataset:
    def __init__(self,
            data,
            Y_target, 
            add_features = None,
            ac = (20,33),
            gc = (34,46),
            #test_size = 0.25,
            standard_scale = False,
            device = None
        ):

        self.add_features = add_features
        self.standard_scale = standard_scale
        self.device = device

        Y = data.loc[:,Y_target]
        non_nan_mask = Y.notna()
        if type(Y_target) == list:
            assert Y_target.index('IV') < Y_target.index('Tg'), 'IV must come before Tg'
            non_nan_mask['res_bool'] = False
            non_nan_mask.loc[non_nan_mask[Y_target].all(1), 'res_bool'] = True
            non_nan_mask = non_nan_mask['res_bool'].values
        
        self.Y = Y[non_nan_mask].values # Get Y values
        self.data = data.loc[non_nan_mask,:]

        # Extract binary presence:
        acids = self.data.iloc[:,ac[0]:ac[1]]
        self.binary_acids = acids.notna().to_numpy(dtype = float)
        glycols = self.data.iloc[:,gc[0]:gc[1]]
        self.binary_glycols = glycols.notna().to_numpy(dtype = float)

        if self.add_features is not None:
            self.add_features = list_mask(self.add_features, list(non_nan_mask))
            self.add_features = np.array(self.add_features, dtype = float)

        self.acid_len = self.binary_acids.shape[1]
        self.glycol_len = self.binary_glycols.shape[1]

        # Get entire dataset:
        # Structure: [([A, A], [G]), ..., ([A, A, A], [G, G, G, G])]
        #   len == length of dataset for this value

    def __len__(self):
        # Sum size of both A and G along with add_features
        af_shape = 0 if self.add_features is None else self.add_features.shape[1]
        return af_shape + self.acid_len + self.glycol_len

    def Kfold_CV(self, folds):
        
        if self.add_features is not None:
            X = np.concatenate([self.binary_acids, self.binary_glycols, self.add_features], axis = 1)
        else:
            X = np.concatenate([self.binary_acids, self.binary_glycols], axis = 1)
        y = self.Y
        #y = torch.from_numpy(self.Y).to(self.device)
        kfold = KFold(n_splits=folds, shuffle = True)

        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx] 

            if self.standard_scale and (self.add_features is not None):
                endlen = np.asarray(self.add_features).shape[1]
                # Scale only additional values:
                ss = StandardScaler().fit(X_train[:,-endlen:])
                X_train[:,-endlen:] = ss.transform(X_train[:,-endlen:])
                X_test[:,-endlen:] = ss.transform(X_test[:,-endlen:])

            yield torch.from_numpy(X_train).float(), \
                torch.from_numpy(X_test).float(), \
                torch.from_numpy(y_train).float(), \
                torch.from_numpy(y_test).float(), \
                torch.from_numpy(train_idx).float(), \
                torch.from_numpy(test_idx).float()    

if __name__ == '__main__':
    from polymerlearn.utils.train_graphs import get_IV_add_nolog
    data = pd.read_csv('../../../dataset/pub_data.csv')
    add_features = get_IV_add_nolog(data)

    dataset = BinaryDataset(
            data = data,
            Y_target = 'IV', 
            add_features = add_features,
            ac = (20,33),
            gc = (34,46),
            standard_scale = True,
            device = None
        )

    for Xtrain, Xtest, _, _, _, _ in dataset.Kfold_CV(5):
        print(Xtrain)