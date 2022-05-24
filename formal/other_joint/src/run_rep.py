import argparse, os, pickle
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from tqdm import trange
from functools import partial

from polymerlearn.utils.comparison_rep.rep_dataset import RepDataset
from polymerlearn.utils.train_graphs import get_IV_add, get_Tg_add
from polymerlearn.utils.train_graphs import get_IV_add_nolog, get_Tg_add_nolog
from polymerlearn.utils.comparison_rep.train_reps import CV_eval, CV_eval_joint
from polymerlearn.models.vector.networks import Vector_Joint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--num_cv', type = int, required = True,
    help = 'Number of cross validations')
parser.add_argument('--rep', type = str)
parser.add_argument('--start_fold', required = True, type = int,
    help = 'Starting fold number. i.e. we could have 5 folds, 30-34, this number would be 30. Used for parallelization of the job script.')
parser.add_argument('--results_save_dir', type = str, 
    help = 'Directory in which to save results as we evaluate folds in the CV.',
    default = None)
parser.add_argument('--standard_scale', action = 'store_true', 
    help = 'If included, standard scale all variables before input')
#parser.add_argument('--Tg', action = 'store_true')

args = parser.parse_args()

assert args.rep in ['CM', 'MBTR', 'SOAP', 'PI']

data = pd.read_csv('../../dataset/pub_data.csv')
if args.standard_scale:
    add_features = get_IV_add_nolog(data)
else:
    add_features = get_IV_add(data)

name = 'J_results_fold={}.pickle'

dataset = RepDataset(
    data = data,
    Y_target = ['IV', 'Tg'],
    rep_dir = os.path.join('../../Representations'),
    add_features = add_features,
    rep = args.rep,
    standard_scale=args.standard_scale,
)

model_generator = Vector_Joint
model_generator_kwargs = {
    'input_feat': len(dataset), # Gets number of features per sample
    'hidden_channels': 32
}

def save_to_loc(obj, cur_name):
    pickle.dump(obj, open(os.path.join(args.results_save_dir, cur_name), 'wb'))

CV = partial(CV_eval_joint,
    dataset = dataset,
    model_generator = model_generator,
    model_generator_kwargs = model_generator_kwargs,
    criterion = torch.nn.MSELoss(),
    optimizer_generator = torch.optim.AdamW,
    optimizer_kwargs = {'lr': 0.0001, 'weight_decay':0.01},
    epochs = 1000,
    batch_size = 64,
    verbose = 1,
    gamma = 5e5,
    get_scores = True,
    device = device,
    save_state_dicts = True,
)


scores = {
    'IV': ([], []),
    'Tg': ([], []) 
}

for i in trange(args.num_cv):
    # if args.save_history:
    #     results_dict, state_dicts = CV()
    # else:
    #     results_dict = CV()
    history = CV()

    scores['IV'][0].append(history['IV'][0])
    scores['IV'][1].append(history['IV'][1])
    scores['Tg'][0].append(history['Tg'][0])
    scores['Tg'][1].append(history['Tg'][1])

    fold = i + args.start_fold # Set actual fold number wrt whole experiment

    if (args.results_save_dir is not None):
        save_to_loc(scores, cur_name = name.format(fold))

    # if args.save_history and (args.history_loc is not None):
    #     hist_loc = os.path.join(args.history_loc, 'joint_model_fold={}.pickle'.format(fold))
    #     pickle.dump(history, open(hist_loc, 'wb'))