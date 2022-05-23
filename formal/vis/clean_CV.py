import pandas as pd

df = pd.read_csv('CV_data2.csv')

print(df)

f = open('CV_data2.csv', 'r')
fw = open('CV_data2_NEW.csv', 'w')

for l in f:
    fw.write(l.replace('"', ''))

f.close()
fw.close()

#df.to_csv('CV_data2.csv', index = False)