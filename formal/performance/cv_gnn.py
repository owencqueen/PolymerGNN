import os, argparse, pickle
import torch
import pandas as pd
import numpy as np
from tqdm import trange
from polymerlearn.utils import get_IV_add, get_Tg_add, GraphDataset
from polymerlearn.models.gnn import PolymerGNN_IV, PolymerGNN_Tg, PolymerGNN_Joint
from polymerlearn.utils import train, CV_eval, get_add_properties
from polymerlearn.utils import CV_eval_joint

from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_transform_lists(raw_args):
    use_log = []
    additionals = []
    for a in raw_args:
        use_log.append( a[:3] == 'log' )
        if a[:3] == 'log':
            a = a[3:]
        additionals.append(a) # Will be filtered if log

    return additionals, use_log

structure_dir = '/lustre/haven/proj/UTK0022/PolymerGNN/Structures/AG/xyz'

# Load data from local path:
data = pd.read_csv(os.path.join('../../dataset', 
            'pub_data.csv'))

parser = argparse.ArgumentParser()
parser.add_argument('--num_cv', type = int, required = True,
    help = 'Number of cross validations')
parser.add_argument('--IV', action = 'store_true')
parser.add_argument('--Tg', action = 'store_true')
parser.add_argument('--properties', type = str, 
    default = ['default'], nargs = '+',
    help = 'Resin-wide properties to include in the model. \n\
        Strings will directly query the pandas dataframe, so spell and use upper/lowercase correctly. \n\
        Use "default" if you want to explicitly ensure usage of default properties; not using this argument also ensures default is used. \n\
        Use "log<>" for some <> property with log transform.')
parser.add_argument('--results_save_dir', type = str, 
    help = 'Directory in which to save results as we evaluate folds in the CV.',
    default = None)
parser.add_argument('--save_state_dicts', action = 'store_true',
    help = 'Stores state dicts for all cross validated models')
parser.add_argument('--save_state_dict_loc', default = None,
    type = str,
    help = 'Location to which to save state dicts; MUST BE PROVIDED if trying to save state dicts; should be absolute.')

args = parser.parse_args()

# General training:
optimizer_gen = torch.optim.AdamW
criterion = torch.nn.MSELoss()

joint_model = False

if args.IV and args.Tg:
    # Joint model
    joint_model = True

    targets = ['IV', 'Tg']
    name = 'joint_results_fold={}.pickle'

    # Decide additional vectors:
    if 'default' in args.properties:
        pass
    else:
        prop, use_log = build_transform_lists(args.properties)
        add = get_add_properties(data, prop, use_log)
    
    dataset = GraphDataset(
        data = data,
        structure_dir = structure_dir,
        Y_target=targets,
        add_features=add,
        device = device
    )

    # Model generator kwargs:
    model_generator_kwargs = {
        'input_feat': 6,
        'hidden_channels': 32,
        'num_additional': add.shape[1], 
    }

    model_gen = PolymerGNN_Joint

    # Cross validation function (with parameters):
    CV = partial(CV_eval_joint,
        dataset = dataset,
        model_generator = PolymerGNN_Joint,
        optimizer_generator = optimizer_gen,
        criterion = criterion,
        model_generator_kwargs = model_generator_kwargs,
        optimizer_kwargs = {'lr': 0.0001, 'weight_decay':0.01},
        epochs = 1000,
        batch_size = 64,
        verbose = 0,
        gamma = 1e5,
        get_only_scores = True,
        device = device,
        save_state_dicts = args.save_state_dicts
    )



if args.IV: # we're predicting IV
    targets = ['IV']
    name = 'IV_results_fold={}.pickle'

    # Decide additional vectors:
    if 'default' in args.properties:
        add = get_IV_add(data)
    else:
        prop, use_log = build_transform_lists(args.properties)
        add = get_add_properties(data, prop, use_log)

    dataset = GraphDataset(
        data = data,
        structure_dir = structure_dir,
        Y_target=targets,
        add_features=add
    )

    # Model generator kwargs:
    model_generator_kwargs = {
        'input_feat': 6,
        'hidden_channels': 32,
        'num_additional': add.shape[1], 
    }

    model_gen = PolymerGNN_IV

    # Cross validation function (with hyperparameters)
    CV = partial(CV_eval, 
        dataset = dataset,
        model_generator = model_gen,
        optimizer_generator = optimizer_gen,
        criterion = criterion,
        model_generator_kwargs = model_generator_kwargs,
        optimizer_kwargs = {'lr': 0.0001, 'weight_decay':0.01},
        epochs = 800,
        batch_size = 64,
        verbose = 0,
        use_val = False,
        get_only_scores = True,
        device = device,
        save_state_dicts = args.save_state_dicts
    )

if args.Tg: # We're predicting Tg:
    targets = ['Tg']
    name = 'Tg_results_fold={}.pickle'

    # Decide additional vectors:
    if 'default' in args.properties:
        add = get_Tg_add(data)
    else:
        prop, use_log = build_transform_lists(args.properties)
        add = get_add_properties(data, prop, use_log)

    dataset = GraphDataset(
        data = data,
        structure_dir = structure_dir,
        Y_target=targets,
        add_features=add,
        device = device
    )

    # Model generator kwargs:
    model_generator_kwargs = {
        'input_feat': 6,
        'hidden_channels': 32,
        'num_additional': add.shape[1], 
    }

    model_gen = PolymerGNN_Tg

    # Cross validation function (with hyperparameters)
    CV = partial(CV_eval, 
        dataset = dataset,
        model_generator = model_gen,
        optimizer_generator = optimizer_gen,
        criterion = criterion,
        model_generator_kwargs = model_generator_kwargs,
        optimizer_kwargs = {'lr': 0.0001, 'weight_decay':0.01},
        epochs = 1000,
        batch_size = 64,
        verbose = 0,
        use_val = False,
        get_only_scores = True,
        device = device,
        save_state_dicts = args.save_state_dicts
    )

def save_to_loc(obj, cur_name):
    pickle.dump(obj, open(os.path.join(args.results_save_dir, name), 'wb'))

if joint_model:
    # Run the CV separately for the joint model
    scores = {
        'IV': ([], []),
        'Tg': ([], []) 
    }

    for fold in trange(args.num_cv):
        if args.save_state_dicts:
            results_dict, state_dicts = CV()
        else:
            results_dict = CV()

        scores['IV'][0].append(results_dict['IV'][0])
        scores['IV'][1].append(results_dict['IV'][1])
        scores['Tg'][0].append(results_dict['Tg'][0])
        scores['Tg'][1].append(results_dict['Tg'][1])

        if fold % 5 == 0 and (args.results_save_dir is not None):
            save_to_loc(scores, cur_name = name.format(fold))

        if args.save_state_dicts and (args.save_state_dict_loc is not None):
            for i in range(len(state_dicts)):
                sd = state_dicts[i]
                torch.save(sd, open(os.path.join(args.save_state_dict_loc, \
                    'joint_model_fold={}_sd={}.pt'.format(fold, i)), 'wb'))


else:
    r2_scores = []
    mae_scores = []

    save_indicator = 'Tg' if args.Tg else 'IV'

    for fold in trange(args.num_cv):
        if args.save_state_dicts:
            avg_r2, avg_mae, state_dicts = CV()
        else:
            avg_r2, avg_mae = CV()

        r2_scores.append(avg_r2)
        mae_scores.append(avg_mae)

        if fold % 5 == 0 and (args.results_save_dir is not None):
            save_to_loc((r2_scores, mae_scores), cur_name = name.format(fold))

        if args.save_state_dicts and (args.save_state_dict_loc is not None):
            for i in range(len(state_dicts)):
                sd = state_dicts[i]
                torch.save(sd, open(os.path.join(args.save_state_dict_loc, \
                    '{}_model_fold={}_sd={}.pt'.format(save_indicator, fold, i)), 'wb'))

    print('R2: {} +- {}'.format(
        np.mean(r2_scores),
        np.std(r2_scores) / np.sqrt(len(r2_scores))
    ))

    print('MAE: {} +- {}'.format(
        np.mean(mae_scores),
        np.std(mae_scores) / np.sqrt(len(mae_scores))
    ))