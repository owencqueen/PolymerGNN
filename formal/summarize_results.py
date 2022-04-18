import os, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt

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

def plot_dist(path):

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


    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(r2)
    ax1.set_title('R2')

    ax2.hist(mae)
    ax2.set_title('MAE')

    plt.show()


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

def plot_dist_joint(path):

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

    fig, ax = plt.subplots(2, 2)

    ax[0][0].hist(r2IV, bins = 50)
    ax[0][0].set_title('R2 IV')

    ax[0][1].hist(maeIV, bins = 50)
    ax[0][1].set_title('MAE IV')

    ax[1][0].hist(r2Tg, bins = 50)
    ax[1][0].set_title('R2 Tg')

    ax[1][1].hist(maeTg, bins = 50)
    ax[1][1].set_title('MAE Tg')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
        help = 'Path to summarize')
    parser.add_argument('--joint', action='store_true',
        help = 'Use if trying to summarize joint model (different storage system).')
    parser.add_argument('--plot', action = 'store_true',
        help = 'Plots histograms of the r2 and mae for each fold')
    args = parser.parse_args()

    if args.plot:
        if args.joint:
            plot_dist_joint(args.path)
        else:
            plot_dist(args.path)

    if args.joint:
        summarize_joint(args.path)
    else:
        summarize(args.path)