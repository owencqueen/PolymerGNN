import os, pickle
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

no_tmp = pickle.load(open('tmp_exp_results/no_tmp.pickle', 'rb'))
tmp = pickle.load(open('tmp_exp_results/tmp.pickle', 'rb'))

# Perform t-test:
tstat, pval = ttest_ind(tmp, no_tmp, alternative='less')
print('Hypothesis test of difference between means: {}, p={}'.format(tstat, pval))

plt.rcParams['font.family'] = 'serif'

plt.boxplot([tmp, no_tmp])
plt.ylabel('Attribution')
plt.xticks([1, 2], ['TMP %', 'No TMP %'])
plt.tight_layout()
plt.show()