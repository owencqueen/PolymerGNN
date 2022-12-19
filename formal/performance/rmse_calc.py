import pickle, os
import argparse
import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error

RMSE = partial(mean_squared_error, squared = False)

def main_single(args):

    rmse_scores = []
    for f in os.listdir(args.dir):
        try:
            h = pickle.load(open(os.path.join(args.dir, f), 'rb'))
            score = RMSE(h['all_y'], h['all_predictions'])
            rmse_scores.append(score)
        except: # Control for stray files (e.g. __pycache__)
            continue

    print('Score = {:.4f} +- {:.4f}'.format(np.mean(rmse_scores), np.std(rmse_scores) / np.sqrt(len(rmse_scores))))

def main_joint(args):
    rmse_scores_tg = []
    rmse_scores_iv = []
    for f in os.listdir(args.dir):
        try:
            h = pickle.load(open(os.path.join(args.dir, f), 'rb'))
            
            # Tg calc:
            tgpred = [l[1] for l in h['all_predictions']]
            tgy = [l[1] for l in h['all_y']]
            score = RMSE(tgy, tgpred)
            rmse_scores_tg.append(score)

            # IV calc:
            ivpred = [l[0] for l in h['all_predictions']]
            ivy = [l[0] for l in h['all_y']]
            score = RMSE(ivy, ivpred)
            rmse_scores_iv.append(score)
        except: # Control for stray files (e.g. __pycache__)
            continue

    print('Tg Score = {:.4f} +- {:.4f}'.format(np.mean(rmse_scores_tg), np.std(rmse_scores_tg) / np.sqrt(len(rmse_scores_tg))))
    print('IV Score = {:.4f} +- {:.4f}'.format(np.mean(rmse_scores_iv), np.std(rmse_scores_iv) / np.sqrt(len(rmse_scores_iv))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, help = 'director containing histories')
    parser.add_argument('--joint', action = 'store_true')

    args = parser.parse_args()

    if args.joint:
        main_joint(args)
    else:
        main_single(args)