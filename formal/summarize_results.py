import os, pickle, argparse
import numpy as np

def summarize(path):

    r2 = []
    mae = []
    count = 0

    for f in os.listdir(path):
        if f[-6:] != 'pickle':
            continue
        r2_score, mae_score = pickle.load(open(os.path.join(path, f), 'rb'))

        r2.append(np.mean(r2_score))
        mae.append(np.mean(mae_score))

        count += 1

    print('Num folds: {}'.format(count))

    print('R2: {:.4f} +- {:.4f}'.format(
        np.mean(r2),
        np.std(r2) / np.sqrt(len(r2))
    ))

    print('MAE: {:.4f} +- {:.4f}'.format(
        np.mean(mae),
        np.std(mae) / np.sqrt(len(mae))
    ))

def summarize_joint(path):

    r2IV = []
    maeIV = []
    r2Tg = []
    maeTg = []
    count = 0

    for f in os.listdir(path):
        if f[-6:] != 'pickle':
            continue
        result_dict = pickle.load(open(os.path.join(path, f), 'rb'))

        r2IV.append(result_dict['IV'][0])
        maeIV.append(result_dict['IV'][1])
        r2Tg.append(result_dict['Tg'][0])
        maeTg.append(result_dict['Tg'][1])

        count += 1

    scores = [(r2IV, maeIV), (r2Tg, maeTg)]
    names = ['IV', 'Tg']

    print('Num folds: {}'.format(count))
    
    for i in range(len(scores)):
        r2, mae = scores[i]

        print('{} -----------------------'.format(names[i]))
        print('R2: {:.4f} +- {:.4f}'.format(
            np.mean(r2),
            np.std(r2) / np.sqrt(len(r2))
        ))

        print('MAE: {:.4f} +- {:.4f}'.format(
            np.mean(mae),
            np.std(mae) / np.sqrt(len(mae))
        ))


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True,
    help = 'Path to summarize')
parser.add_argument('--joint', action='store_true',
    help = 'Use if trying to summarize joint model (different storage system).')
args = parser.parse_args()

if __name__ == '__main__':
    if args.joint:
        summarize_joint(args.path)
    else:
        summarize(args.path)