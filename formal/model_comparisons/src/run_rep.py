import argparse, os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from tqdm import trange

from polymerlearn.utils.comparison_rep.rep_dataset import RepDataset
from polymerlearn.utils.train_graphs import get_IV_add, get_Tg_add


parser = argparse.ArgumentParser()
parser.add_argument('--num_cv', type = int, required = True,
    help = 'Number of cross validations')
parser.add_argument('--rep', type = str)
parser.add_argument('--target', type = str)
parser.add_argument('--start_fold', required = True, type = int,
    help = 'Starting fold number. i.e. we could have 5 folds, 30-34, this number would be 30. Used for parallelization of the job script.')
parser.add_argument('--results_save_dir', type = str, 
    help = 'Directory in which to save results as we evaluate folds in the CV.',
    default = None)
#parser.add_argument('--Tg', action = 'store_true')

args = parser.parse_args()

assert args.rep in ['CM', 'MBTR', 'SOAP']
assert args.target in ['Tg', 'IV']

data = pd.read_csv('../../dataset/pub_data.csv')
add_features = get_IV_add(data) if args.target == 'IV' else get_Tg_add(data)

name = 'IV_results_fold={}.pickle' if args.target == 'IV' else 'Tg_results_fold={}.pickle'

dataset = RepDataset(
    data = data,
    Y_target = args.target,
    rep_dir = os.path.join('../../Representations'),
    add_features = add_features,
    rep = args.rep
)

def save_to_loc(obj, cur_name):
    pickle.dump(obj, open(os.path.join(args.results_save_dir, cur_name), 'wb'))

for i in trange(args.num_cv):

    # Use RF for all
    model = KernelRidge()
    r2, mae = dataset.cross_val_predict(model)

    fold = i + args.start_fold

    if args.results_save_dir is not None:
        save_to_loc((r2, mae), cur_name = name.format(fold))
