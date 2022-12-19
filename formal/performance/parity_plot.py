import os, pickle, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')
font = {'family' : 'serif',
        'size'   : 12}

matplotlib.rc('font', **font)

def median(arr):
    inds = np.argsort(arr)
    return inds[len(inds) // 2]

def main(args):
    
    # Open files:
    d = args.path
    yhat_iv, y_iv = [], []
    yhat_tg, y_tg = [], []
    iv_mae, tg_mae = [], []
    iv_r2, tg_r2 = [], []
    for f in os.listdir(d):
        h = pickle.load(open(os.path.join(d, f), 'rb'))

        pred = h['all_predictions']
        y = h['all_y']

        if args.iv and args.tg:
            iv_mae.append(h['IV'][1])
            iv_r2.append(h['IV'][0])
            yhat_iv.append([pred[i][0] for i in range(len(pred))])
            y_iv.append([y[i][0] for i in range(len(y))])

            tg_mae.append(h['Tg'][1])
            tg_r2.append(h['Tg'][0])
            yhat_tg.append([pred[i][1] for i in range(len(pred))])
            y_tg.append([y[i][1] for i in range(len(y))])

        elif args.iv:
            iv_mae.append(h['mae'])
            iv_r2.append(h['r2'])
            yhat_iv.append(pred)
            y_iv.append(y)
        elif args.tg:
            tg_mae.append(h['mae'])
            tg_r2.append(h['r2'])
            yhat_tg.append(pred)
            y_tg.append(y)

    i = None
    if (args.iv and args.tg): # Joint model
        tog = np.array(iv_r2) + np.array(tg_r2)
        i = median(tog)

        # Plot fig

        #plt.savefig('joint_iv_parity.pdf')

    if args.iv: # IV-only model
        if i is None:
            i = median(iv_r2)
        y = y_iv[i]
        yhat = yhat_iv[i]

        plt.plot([min(y), max(y)], [min(y), max(y)], color = 'black', linestyle = '--')
        plt.scatter(y, yhat, color = '#006C93')
        plt.ylabel('Predicted IV', c = 'black')
        plt.xlabel('Actual IV', c = 'black')
        lx, rx = plt.xlim()
        by, ty = plt.ylim()

        plt.text(rx*0.75, 0.05, s = '$R^2$ = {:.4f}'.format(iv_r2[i]))
        plt.xticks(c = 'black')
        plt.yticks(c = 'black')
        if args.iv and args.tg:
            plt.savefig('joint_iv_parity.pdf')
        else:
            plt.savefig('iv_parity.pdf')
        plt.show()
        


    if args.tg: # Tg-only model
        if i is None:
            i = median(tg_r2)
        y = y_tg[i]
        yhat = yhat_tg[i]

        plt.plot([min(y), max(y)], [min(y), max(y)], color = 'black', linestyle = '--')
        plt.scatter(y, yhat, color = '#FF8200')
        plt.ylabel('Predicted $T_g$', c = 'black')
        plt.xlabel('Actual $T_g$', c = 'black')
        lx, rx = plt.xlim()
        by, ty = plt.ylim()

        plt.text(rx*0.7, by*0.9, s = '$R^2$ = {:.4f}'.format(tg_r2[i]))
        plt.xticks(c = 'black')
        plt.yticks(c = 'black')

        if args.iv and args.tg:
            plt.savefig('joint_tg_parity.pdf')
        else:
            plt.savefig('tg_parity.pdf')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to directory with history files')
    parser.add_argument('--tg', action='store_true')
    parser.add_argument('--iv', action='store_true')
    args = parser.parse_args()

    main(args)