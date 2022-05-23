import pandas as pd


iter_methods = ['Simple', 'OH', 'OH+Mw', 'CM', 'BOB', 'PI (20)', 'PI (50)', 'PI (100)', 'SOAP', 'MBTR']
iter_targets = ['tg', 'iv']
iter_metrics = ['r2', 'mae']

# Construct names:
names = []
for method in iter_methods:
    for target in iter_targets:
        for metric in iter_metrics:
            n = '{}_{}_{}'.format(method, target, metric)
            names.append(n)

df = pd.read_csv('results_raw.csv', sep =',', names = names)

print(df)
df.to_csv('results_clean.csv', index = False)