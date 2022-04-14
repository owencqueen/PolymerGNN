from multiprocessing import pool
import os, pickle
import numpy as np
import pandas as pd

from polymerlearn.utils.graph_prep import get_AG_info, list_mask
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

base_rep_dir = os.path.join('../../..',
    'Representations')

def load_pickle(dir, mol_name):
    f = os.path.join(dir, mol_name.lower() + '.pickle')
    rep = pickle.load(open(f, 'rb'))

    return rep

class RepDataset:
    def __init__(self,
            data,
            Y_target, 
            rep_dir = base_rep_dir,
            add_features = None,
            ac = (20,33),
            gc = (34,46),
            #test_size = 0.25,
            rep = 'CM',
        ):

        self.add_features = add_features

        rep = rep.upper()
        assert rep in ['CM', 'BOB', 'SOAP'], "Representation must be in ['CM', 'BOB', 'SOAP']"

        Y = data.loc[:,Y_target]
        non_nan_mask = Y.notna()
        self.Y = Y[non_nan_mask].values # Get Y values
        self.data = data.loc[non_nan_mask,:]

        self.acid_included, self.glycol_included, self.acid_pcts, self.glycol_pcts = \
            get_AG_info(self.data, ac, gc)

        if self.add_features is not None:
            self.add_features = list_mask(self.add_features, list(non_nan_mask))

        # Get entire dataset:
        # Structure: [([A, A], [G]), ..., ([A, A, A], [G, G, G, G])]
        #   len == length of dataset for this value

        dirlook = os.path.join(rep_dir, rep, 'AG')

        self.dataset = []
        max_size = 0
        for A, G in zip(self.acid_included, self.glycol_included):
            Asamples = [load_pickle(dirlook, a).flatten()  for a in A]
            Gsamples = [load_pickle(dirlook, g).flatten()  for g in G]

            Asizes = [A.shape[0] for A in Asamples]
            Gsizes = [G.shape[0] for G in Gsamples]

            max_size = max(max(Asizes), max(Gsizes), max_size)

            self.dataset.append((Asamples, Gsamples))

        # Pad all representations with the size of the largest in either grouping
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset[i])):
                for k in range(len(self.dataset[i][j])):
                    cshape = self.dataset[i][j][k].shape[0]
                    self.dataset[i][j][k] = \
                        np.concatenate([self.dataset[i][j][k], np.zeros(max_size - cshape)])

    def pool_dataset(self, pool_method = np.max):
        # Go through and pool each
        pooled = []
        i = 0
        for A, G in self.dataset:
            poolA = np.sum(np.stack(A), axis = 0)
            poolG = np.sum(np.stack(G), axis = 0)
            if self.add_features is not None:
                toconcat_list = [poolA.flatten(), poolG.flatten(), self.add_features[i]]
            else:
                toconcat_list = [poolA.flatten(), poolG.flatten()]
            pooled.append(np.concatenate(toconcat_list))

            i += 1

        return pooled

    def cross_val_predict(self, model, pool_method = np.sum, verbose = 0, shuf = False):
        X = self.pool_dataset(pool_method)

        if shuf:
            X, y = shuffle(X, self.Y)
        else:
            y = self.Y

        r2_scores = cross_val_score(model, X, y, verbose = verbose, scoring = 'r2')
        mae_scores = -1.0 * cross_val_score(model, X, y, verbose = verbose, scoring = 'neg_mean_absolute_error')
        #scores = cross_val

        return r2_scores, mae_scores

def test_dataset():
    data = pd.read_csv('../../../dataset/pub_data.csv')
    dataset = RepDataset(data, Y_target = 'IV', rep = 'SOAP')

    X = dataset.pool_dataset()

    print(X[0])
    print(X[0].shape)

def test_cv():
    from polymerlearn.utils.train_graphs import get_IV_add
    data = pd.read_csv('../../../dataset/pub_data.csv')
    to_add = get_IV_add(data)
    dataset = RepDataset(data, Y_target = 'IV', rep = 'SOAP', add_features=to_add)

    from sklearn.ensemble import RandomForestRegressor
    #from sklearn.neural_network import MLPRegressor

    # model = MLPRegressor(
    #     hidden_layer_sizes=(128, 128, 64),
    #     solver = 'adam',
    #     learning_rate_init = 0.001,
    #     batch_size = 32,
    #     max_iter = 500
    # )

    model = RandomForestRegressor()
    scores = dataset.cross_val_predict(model, verbose = 2)

    print(np.mean(scores))

if __name__ == '__main__':
    #test_dataset()
    test_cv()