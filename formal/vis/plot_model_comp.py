import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys; sys.path.append('..')
from summarize_results import get_r2_mae, get_r2_mae_joint

IV_paths = [
    # '../model_comparisons/CM/iv',
    # '../model_comparisons/MBTR/iv',
    # '../model_comparisons/SOAP/iv',
    '../arch_ablation/saved_scores/iv',
    '../performance/saved_scores/joint'
]

tg_paths = [
    # '../model_comparisons/CM/tg',
    # '../model_comparisons/MBTR/tg',
    # '../model_comparisons/SOAP/tg',
    '../ss_ablation/saved_scores/tg',
    '../performance/saved_scores/joint'
]

FIGSIZE = (6, 4)

def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150, sep = 200, color = None):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
 
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    curves = []
    ys = []
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i*(1.0-overlap) * sep
        ys.append(y)
        curve = pdf(xx)
        if fill:
            c = color[i] if color is not None else fill
            plt.fill_between(xx, np.ones(n_points)*y, 
                             curve+y, zorder=len(data)-i+1, color=c)
        plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
    if labels:
        plt.yticks(ys, labels)

    return ys

def get_data(paths):

    data_r2 = []
    data_mae = []

    for p in paths:
        r2, mae, count = get_r2_mae(p)

        data_r2.append(r2)
        data_mae.append(mae)

    return data_r2, data_mae

def get_data_joint(paths):

    r2IVs = []
    r2Tgs = []
    maeIVs = []
    maeTgs = []

    for p in paths:
        r2IV, maeIV, r2Tg, maeTg, _ = get_r2_mae_joint(p)

        r2IVs.append(r2IV)
        r2Tgs.append(r2Tg)
        maeIVs.append(maeIV)
        maeTgs.append(maeTg)

    return r2IVs,maeIVs,r2Tgs,maeTgs

def filter_csv(df, comp = 'iv'):

    # Parse through the csv dataframe and filter out relevant data

    if comp == 'iv':
        mask = [False, False, True, True] * int(df.shape[1] / 4)

    else:
        mask = [True, True, False, False] * int(df.shape[1] / 4)

    relevant = df.loc[:,mask]

    # OH
    OH_r2 = relevant.iloc[:,0].tolist()
    OH_mae = relevant.iloc[:,1].tolist()

    # simp
    simp_r2 = relevant.iloc[:,2].tolist()
    simp_mae = relevant.iloc[:,3].tolist()

    # OHS
    OHS_r2 = relevant.iloc[:,4].tolist()
    OHS_mae = relevant.iloc[:,5].tolist()

    r2 = [OH_r2, simp_r2, OHS_r2]
    mae = [OH_mae, simp_mae, OHS_mae]

    return r2, mae

def filter_csv_2(df, comp = 'iv'):

    # Parse through the csv dataframe and filter out relevant data

    if comp == 'iv':
        mask = [False, False, True, True] * int(df.shape[1] / 4)

    else:
        mask = [True, True, False, False] * int(df.shape[1] / 4)

    relevant = df.loc[:,mask]

    # simp
    simp_r2 = relevant.iloc[:,0].tolist()
    simp_mae = relevant.iloc[:,1].tolist()

    # OH
    OH_r2 = relevant.iloc[:,2].tolist()
    OH_mae = relevant.iloc[:,3].tolist()

    # OHS
    OHS_r2 = relevant.iloc[:,4].tolist()
    OHS_mae = relevant.iloc[:,5].tolist()

    # CM
    CM_r2, CM_mae = relevant.iloc[:,6].tolist(), relevant.iloc[:,7].tolist()

    # BOB
    BOB_r2, BOB_mae = relevant.iloc[:,8].tolist(), relevant.iloc[:,9].tolist()
    #10,11
    #12,13
    #14,15

    # SOAP - skip PI * 3
    SOAP_r2, SOAP_mae = relevant.iloc[:,16].tolist(), relevant.iloc[:,17].tolist()

    # MBTR
    MBTR_r2, MBTR_mae = relevant.iloc[:,18].tolist(), relevant.iloc[:,19].tolist()

    # r2 = [OH_r2, simp_r2, OHS_r2]
    # mae = [OH_mae, simp_mae, OHS_mae]

    r2 = [simp_r2, OH_r2, OHS_r2, CM_r2, BOB_r2, SOAP_r2, MBTR_r2]
    mae = [simp_mae, OH_mae, OHS_mae, CM_mae, BOB_mae, SOAP_mae, MBTR_mae]

    return r2, mae
    
def plot_IV(opt = 1):

    #print(IV_paths[-1])

    r2, mae = get_data(IV_paths[:-1])
    r2J, maeJ, _, _ = get_data_joint([IV_paths[-1]]) 

    #print(list(np.array(r2J).flatten()))

    r2.append(list(np.array(r2J).flatten())); mae.append(list(np.array(maeJ).flatten()))
    
    # Get data from Gavin's experiments:

    if opt == 1:
        other = pd.read_csv('CV_data2.csv')
        r2O, maeO = filter_csv(other, comp = 'iv')

        lab = [
            'OH',
            'Properties',
            'OH + \nProperties',
            'CM',
            'MBTR',
            'SOAP',
            'PGNN IV',
            'PGNN Joint'
        ]

        #c = ['red', 'blue', 'green', 'yellow']
        c = [
            'darkorange', 
            'm',
            'teal',
            'indigo', 
            'y', 
            'mediumblue', 
            'darkgreen', 
            'darkred']

    elif opt == 2:
        other = pd.read_csv('05-16_gavin/results_clean.csv')
        r2O, maeO = filter_csv_2(other, comp = 'iv')

        lab = [
            'Mw',
            'One-Hot',
            'Ont-Hot\n + Mw',
            'CM',
            'BOB',
            'SOAP',
            'MBTR',
            'PGNN IV',
            'PGNN Joint'
        ]

        #c = ['red', 'blue', 'green', 'yellow']
        c = [
            'darkorange', 
            'm',
            'teal',
            'indigo', 
            'y', 
            'mediumblue', 
            'darkgreen', 
            'darkred',
            'sienna',
            ]

    r2 = r2O + r2
    mae = maeO + mae

    # Sort by R2:
    args = np.argsort([np.mean(r) for r in r2])

    def apply_args(L):
        return [L[i] for i in args]

    r2 = apply_args(r2)
    mae = apply_args(mae)
    lab = apply_args(lab)
    c = apply_args(c)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=FIGSIZE)
    ridgeline(r2, overlap =0, fill = 'y', sep = 10,
        labels = lab, color = c)
    plt.xlabel('$R^2$')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=FIGSIZE)
    ridgeline(mae, overlap =0, fill = 'y', sep = 250,
        labels = lab, color = c)
    plt.xlabel('MAE')
    plt.tight_layout()
    plt.show()

def plot_Tg(opt = 1):

    print(tg_paths[-1])

    r2, mae = get_data(tg_paths[:-1])
    _, _ , r2J, maeJ = get_data_joint([tg_paths[-1]]) 

    r2.append(list(np.array(r2J).flatten())); mae.append(list(np.array(maeJ).flatten()))

    # Get data from Gavin's experiments:
    if opt == 1:
        other = pd.read_csv('CV_data2.csv')
        r2O, maeO = filter_csv(other, comp = 'tg')
        lab = [
            'OH',
            'Properties',
            'OH + \nProperties',
            'CM',
            'MBTR',
            'SOAP',
            'PGNN $T_g$',
            'PGNN Joint'
        ]

    #c = ['red', 'blue', 'green', 'yellow']
        c = [
            'darkorange', 
            'm',
            'teal',
            'indigo', 
            'y', 
            'mediumblue', 
            'darkgreen', 
            'darkred']

    elif opt == 2:
        other = pd.read_csv('05-16_gavin/results_clean.csv')
        r2O, maeO = filter_csv_2(other, comp = 'tg')
        lab = [
            'Mw',
            'One-Hot',
            'Ont-Hot\n + Mw',
            'CM',
            'BOB',
            'SOAP',
            'MBTR',
            'PGNN IV',
            'PGNN Joint'
        ]

        c = [
            'darkorange', 
            'm',
            'teal',
            'indigo', 
            'y', 
            'mediumblue', 
            'darkgreen', 
            'darkred',
            'black',
            ]

    r2 = r2O + r2
    mae = maeO + mae

    # Sort by R2:
    args = np.argsort([np.mean(r) for r in r2])

    def apply_args(L):
        return [L[i] for i in args]

    r2 = apply_args(r2)
    mae = apply_args(mae)
    lab = apply_args(lab)
    c = apply_args(c)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=FIGSIZE)
    ridgeline(r2, overlap =0, fill = 'y', sep = 10,
        labels = lab, color = c)
    plt.xlabel('$R^2$')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=FIGSIZE)
    ridgeline(mae, overlap =0, fill = 'y', sep = 0.5,
        labels = lab, color = c)
    plt.xlabel('MAE')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_IV(opt = 2)

