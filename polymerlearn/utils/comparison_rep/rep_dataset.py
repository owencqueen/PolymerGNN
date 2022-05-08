from multiprocessing import pool
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
            standard_scale = False,
        ):

        self.add_features = add_features
        self.standard_scale = standard_scale

        rep = rep.upper()
        assert rep in ['CM', 'MBTR', 'SOAP'], "Representation must be in ['CM', 'MBTR', 'SOAP']"

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
            poolA = np.max(np.stack(A), axis = 0)
            poolG = np.max(np.stack(G), axis = 0)
            if self.add_features is not None:
                toconcat_list = [poolA.flatten(), poolG.flatten(), self.add_features[i]]
            else:
                toconcat_list = [poolA.flatten(), poolG.flatten()]
            pooled.append(np.concatenate(toconcat_list))

            i += 1

        return np.asarray(pooled)

    def cross_val_predict(self, model, pool_method = np.sum, verbose = 0, shuf = True, folds = 5):
        X = self.pool_dataset(pool_method)

        if shuf:
            X, y = shuffle(X, self.Y)
        else:
            y = self.Y

        kf = KFold(n_splits = folds)

        r2_scores = []
        mae_scores = []

        for train_idx, test_idx in kf.split(X):
            #print(train_idx)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.standard_scale:
                endlen = np.asarray(self.add_features).shape[1]
                # Scale only additional values:
                ss = StandardScaler().fit(np.array(X_train[:,-endlen:]))
                X_train[:,-endlen:] = ss.transform(X_train[:,-endlen:])
                X_test[:,-endlen:] = ss.transform(X_test[:,-endlen:])

            #M = model.copy()

            model.fit(X_train, y_train)

            yhat = model.predict(X_test)
            r2_scores.append(R2(y_test, yhat))
            mae_scores.append(MAE(y_test, yhat))

        #r2_scores = cross_val_score(model, X, y, verbose = verbose, scoring = 'r2')
        #mae_scores = -1.0 * cross_val_score(model, X, y, verbose = verbose, scoring = 'neg_mean_absolute_error')
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