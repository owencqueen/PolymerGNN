import pickle
import numpy as np
import matplotlib.pyplot as plt

tmp = pickle.load(open('attr_scores_G/tmp.pickle', 'rb'))

# Aggregate scores from dictionary:
tmp_biglist = []
static_tmp_keys = list(tmp.keys())
for k in static_tmp_keys:
    tmp_biglist.append(tmp[k])

tmp_means = [np.median(h) for h in tmp_biglist]
args = np.argsort(tmp_means)
static_tmp_keys = [static_tmp_keys[i] for i in args]

tmp_biglist = sorted(tmp_biglist, key = np.median)

plt.rcParams['font.family'] = 'serif'
plt.boxplot(tmp_biglist)
plt.ylabel('Attribution')
plt.xticks(list(range(1, len(tmp_biglist) + 1)), static_tmp_keys)
plt.tight_layout()
plt.show()

no_tmp = pickle.load(open('attr_scores_G/no_tmp.pickle', 'rb'))
#no_tmp = pickle.load(open('attr_scores_G/tmp.pickle', 'rb'))

# Aggregate scores from dictionary:
no_tmp_biglist = []
#static_no_tmp_keys = list(no_tmp.keys())
for k in static_tmp_keys:
    no_tmp_biglist.append(no_tmp[k])

tmp_means = [np.median(h) for h in no_tmp_biglist]
args = np.argsort(tmp_means)
static_tmp_keys = [static_tmp_keys[i] for i in args]
no_tmp_biglist = sorted(no_tmp_biglist, key = np.median)

plt.rcParams['font.family'] = 'serif'
plt.boxplot(no_tmp_biglist)
plt.ylabel('Attribution')
plt.xticks(list(range(1, len(no_tmp_biglist) + 1)), static_tmp_keys)
plt.tight_layout()
plt.show()