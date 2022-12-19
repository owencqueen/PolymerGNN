import os, pickle, argparse
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from polymerlearn.utils import get_IV_add, get_Tg_add, GraphDataset
from polymerlearn.utils.train_graphs import get_IV_add_nolog
from polymerlearn.utils.train_graphs import get_add_properties
from polymerlearn.explain import PolymerGNN_IV_EXPLAIN, PolymerGNN_IVMono_EXPLAIN
from polymerlearn.explain import PolymerGNN_Tg_EXPLAIN, PolymerGNNExplainer

def agg_exps(exp_list, add_data_keys = ['Mw', 'AN', 'OHN', '%TMP']):

    agg_addkeys = {a:[] for a in add_data_keys}
    acid_scores = []
    glycol_scores = []

    for i in range(len(exp_list)):
        for ad in add_data_keys:
            agg_addkeys[ad] += exp_list[i][3][ad]

        for j in range(len(exp_list[i][0])):
            acid_scores.append(torch.sum(exp_list[i][0][j]['A']).item())
            glycol_scores.append(torch.sum(exp_list[i][0][j]['G']).item())

    return acid_scores, glycol_scores, agg_addkeys

parser = argparse.ArgumentParser()
parser.add_argument('--history_loc', type = str, required = True,
    help = 'Location of histories to explain')
parser.add_argument('--target', type = str, required = True,
    help = 'Either Tg or IV, whichever is the target to predict')
parser.add_argument('--num_add', type = int, default = None,
    help = 'Number of additional resin properties for the model')
parser.add_argument('--tmp_experiment', action='store_true', 
    help = 'If added, will run the TMP experiment shown in the paper')
parser.add_argument('--tmp_rank_experiment', action = 'store_true',
    help = 'If marked, runs the rank of tmp experiment in the paper. Does not override tmp_experiment, please only use one.')
parser.add_argument('--save_path_tmp_experiment', type = str, default = None,
    help = 'Path in which to save tmp experiment results')
parser.add_argument('--monomertype', type = str, default = 'G',
    help = 'A (acid) or G (glycol). Used only for TMP rank experiment.')
parser.add_argument('--mono', action = 'store_true', help ='Use if using Mono variant for IV')
parser.add_argument('--mw_experiment', action = 'store_true', help ='Run Mw Experiment (partitioning and observing attribution values)')
parser.add_argument('--debug', action = 'store_true')
args = parser.parse_args()

# Load dataset:
data = pd.read_csv(os.path.join('/Users/owenqueen/Desktop/Eastman_Project-confidential',
            'Eastman_Project/PolymerGNN/dataset', 
            'pub_data.csv'))

if args.target == 'IV':
    add = get_IV_add_nolog(data)
    add_data_keys = ['Mw', 'AN', 'OHN', '%TMP']
    if args.num_add is not None:
        add_data_keys = add_data_keys[:args.num_add]
        for_add_prop = ['Mw (PS)', 'AN', 'OHN', '%TMP'][:args.num_add]
        add = get_add_properties(data, prop_names = for_add_prop)
    name_list = ['A', 'G', 'Mw', 'AN', 'OHN', '%TMP']
    mgen = PolymerGNN_IVMono_EXPLAIN if args.mono else PolymerGNN_IV_EXPLAIN
else: 
    add = get_Tg_add(data)
    add_data_keys = ['Mw']
    name_list = ['A', 'G', 'Mw']
    mgen = PolymerGNN_Tg_EXPLAIN

dataset = GraphDataset(
    data = data,
    Y_target = [args.target],
    structure_dir = '../../Structures/AG/xyz',
    test_size = 0.2,
    add_features = add,
    standard_scale = (args.target == 'IV')
)

# ASSUME model_kwargs STAYS THE SAME
model_kwargs = {
    'input_feat': 6,
    'hidden_channels': 32,
    'num_additional': len(add_data_keys)
}

exps = []
ref_inds = []
count = 0
for f in tqdm(os.listdir(args.history_loc)):
    # Gather all relevant histories:
    history = pickle.load(open(os.path.join(args.history_loc, f), 'rb'))

    # Make splits of the reference indices:
    kfgen = KFold(n_splits=5, shuffle=False).split(history['all_reference_inds'])
    split_ref_inds = [k[1] for k in kfgen]

    ref_inds.append(history['all_reference_inds'])

    # Iterate over the splits, since state dictionaries are separate by kfold splits
    for i in range(len(history['model_state_dicts'])):

        mexplain = mgen(**model_kwargs)
        mexplain.load_state_dict(history['model_state_dicts'][i])

        explainer = PolymerGNNExplainer(mexplain, use_mono = args.mono)

        # Provide arguments based on history
        exp_out = explainer.get_testing_explanation(
            dataset,
            test_inds = split_ref_inds[i],
            add_data_keys=add_data_keys
        )

        exps.append(exp_out)
    
    count += 1
    if args.debug and (count > 1): # Breaks after 2
        break

if args.tmp_experiment:
    tmp_importance = []
    for i in range(len(exps)):
        tmp_importance += list(exps[i][2]['TMP']) # Glycol key -> TMP importance

    print('TMP mean importance: {} +- {}'.format(
        np.mean(tmp_importance),
        np.std(tmp_importance) / np.sqrt(len(tmp_importance))
    ))

    if args.save_path_tmp_experiment is not None:
        pickle.dump(tmp_importance, open(args.save_path_tmp_experiment, 'wb'))

elif args.mw_experiment:

    mw_vals = []
    mw_attr = []

    for i in range(len(ref_inds)):
        # Each ref_inds[i] is a sub-list
        for j in range(len(ref_inds[i])):
            mw_attr.append(exps[i][-1]['Mw'][0])
            mw_vals.append(data.iloc[ref_inds[i][j],:].loc['Mw (PS)'])

    # Partition by mw_vals:
    mw_inds = np.argsort(mw_vals)
    upper_quartile = mw_inds[-int(len(mw_inds) / 4):]
    lower_quartile = mw_inds[:int(len(mw_inds) / 4)]

    upper_attr = [mw_attr[i] for i in upper_quartile]
    lower_attr = [mw_attr[i] for i in lower_quartile]

    print('Upper Quartile Attr: {:.4f} +- {:.4f}'.format(np.mean(upper_attr), np.std(upper_attr) / np.sqrt(len(upper_attr))))
    print('Lower Quartile Attr: {:.4f} +- {:.4f}'.format(np.mean(lower_attr), np.std(lower_attr) / np.sqrt(len(lower_attr))))

    plt.rcParams["font.family"] = "serif"
    fig = plt.gcf()
    fig.set_size_inches(5, 5)

    #plt.hlines(0, xmin=0, xmax=len(name_list) + 1, colors = 'black', linestyles='dashed')
    plt.boxplot([lower_attr, upper_attr])
    plt.ylabel('Attribution')
    plt.xticks([1, 2], ['Lower Quartile', 'Upper Quartile'])
    plt.tight_layout()
    plt.show()

elif args.tmp_rank_experiment:

    ind = 2 if args.monomertype == 'G' else 1

    example_dict = exps[0][ind]
    attr_scores = {g:[] for g in example_dict.keys()}

    for i in range(len(exps)):
        for a in attr_scores.keys():
            attr_scores[a] += list(exps[i][ind][a])

    if args.save_path_tmp_experiment is not None:
        pickle.dump(attr_scores, open(args.save_path_tmp_experiment, 'wb'))

else:

    acid_scores, glycol_scores, agg_addkeys = agg_exps(exps, add_data_keys=add_data_keys)

    plt.figure(dpi=150, figsize=(4, 5))
    all_data = []
    all_data.append(acid_scores)
    all_data.append(glycol_scores)

    for k, v in agg_addkeys.items():
        if k == '%TMP':
            m = np.array(v)[np.abs(v) > 1e-9]
        else:
            m = v
        print(f'{k} mean importance: {np.mean(m)}')

        all_data.append(m)

    print('Acid embedding mean importance:', np.mean(acid_scores))
    print('Glycol embedding mean importance', np.mean(glycol_scores))

    plt.rcParams["font.family"] = "serif"
    fig = plt.gcf()
    fig.set_size_inches(5, 5)

    plt.hlines(0, xmin=0, xmax=len(name_list) + 1, colors = 'black', linestyles='dashed')
    plt.boxplot(all_data)
    plt.ylabel('Attribution')
    if args.target == 'IV': 
        plt.yscale('symlog', linthresh = 0.5)
    plt.xticks(list(range(1, len(name_list) + 1)), name_list)
    plt.tight_layout()
    plt.show()