import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.utils import shuffle

plt.rcParams["font.family"] = "serif"

data = pd.read_csv('../pub_data.csv')

IVmask = data['IV'].notna()
Tgmask = data['Tg'].notna()
non_nan_mw = data['Mw (PS)'].notna()
IVmask &= non_nan_mw
Tgmask &= non_nan_mw
#medmw = data['Mw (PS)'].loc[non_nan_mw].median()

IV = data['IV'].loc[IVmask]
mwIV = data['Mw (PS)'].loc[IVmask]

Tg = data['Tg'].loc[Tgmask] + 273.15
mwTg = data['Mw (PS)'].loc[Tgmask]

# mwIV.loc[mwIV.isna()] = medmw
# mwTg.loc[mwTg.isna()] = medmw

fig, ax = plt.subplots(1, 2)

# ax[0,0].scatter(mwIV, IV)
# ax[0,1].scatter(mwTg, Tg)

# ax[1,0].scatter(np.log(mwIV), np.log(IV))
# ax[1,1].scatter(np.log(mwTg), np.log(Tg))

ax[0].scatter(mwIV, IV, c = 'green')
ax[1].scatter(np.log(mwIV), np.log(IV), c = 'green')

ax[0].xaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
#ax[1].xaxis.set_major_formatter(FormatStrFormatter('%1.0e'))

ax[0].set_xlabel('Mw')
ax[0].set_ylabel('IV')
ax[1].set_xlabel('log(Mw)')
ax[1].set_ylabel('log(IV)')

# X = np.expand_dims(np.log(mwIV).to_numpy(), axis = 1)
# y = np.log(IV).to_numpy()

X = np.log(np.expand_dims((mwIV).to_numpy(), axis = 1))
y = (IV).to_numpy()

trials = 50
means = []
mae_means = []
for i in range(trials):
    X, y = shuffle(X, y)

    reg = LinearRegression()
    #reg.fit(np.log(mwIV).to_numpy(), y = np.log(IV).to_numpy())
    cv = cross_val_predict(reg, X, y) #scoring = 'r2')
    print(cv)
    exit()
    cvmae = cross_val_score(reg, X, y, scoring = 'neg_mean_absolute_error')
    means.append(np.mean(cv))
    mae_means.append(-1.0 * np.mean(cvmae))

print(np.mean(means))
print(np.std(means) / np.sqrt(trials))

print(np.mean(mae_means))
print(np.std(mae_means) / np.sqrt(trials))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(mwIV).to_numpy(), y = np.log(IV).to_numpy())
print('log-log', r_value)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress((mwIV).to_numpy(), y = (IV).to_numpy())
print('no log', r_value)

#print(np.log(mwIV).to_numpy())
#print('Log-log', np.power(np.corrcoef(np.log(mwIV).to_numpy(), y = np.log(IV).to_numpy()), 2))

plt.tight_layout()
plt.show()