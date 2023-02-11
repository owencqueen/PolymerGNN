import numpy as np
import pandas as pd

from polymerlearn.utils.graph_prep import get_AG_info

def main():
    pub_data = pd.read_csv('../PolymerGNN/dataset/pub_data.csv')
    mono_smiles = pd.read_csv('monomer_smiles.csv', sep = '\t')
    smiles_ref = {mono_smiles['abbr'].iloc[i].lower().replace('"',''): mono_smiles['smiles'].iloc[i] for i in range(mono_smiles.shape[0])}

    Y_target = 'IV' # Change here to alter labels %%%%%%%%%%%%%%%%
    Y = pub_data.loc[:,Y_target]
    non_nan_mask = Y.notna()
    if type(Y_target) == list:
        assert Y_target.index('IV') < Y_target.index('Tg'), 'IV must come before Tg'
        non_nan_mask['res_bool'] = False
        non_nan_mask.loc[non_nan_mask[Y_target].all(1), 'res_bool'] = True
        non_nan_mask = non_nan_mask['res_bool'].values
    
    Y = Y[non_nan_mask].values # Get Y values
    data = pub_data.loc[non_nan_mask,:]

    print(data.loc[:,['Tg', 'IV']])
    print(data.shape)

    acid_included, glycol_included, acid_pcts, glycol_pcts = get_AG_info(data)

    # print('Acid included', acid_included)
    # print('Acid pcts', acid_pcts)

    # Construct input dataframe:
    # monomer1.<>.monomerN|pct1|...|pctN|,label1,label2,...

    df_skeleton = {'master_chemprop_input':[]}
    Y_target =  Y_target if isinstance(Y_target, list) else [Y_target]
    for y in Y_target:
        df_skeleton[y] = []

    for i in range(data.shape[0]):
        yi = [Y[i]] # Yvals

        # All monomers:
        acids, acid_pct = acid_included[i], acid_pcts[i]
        glycols, glycol_pct = glycol_included[i], glycol_pcts[i]

        monomers = [smiles_ref[s.lower()] for s in (acids + glycols)]
        mstr = ''
        for j, m in enumerate(monomers):
            mstr += m
            if j < (len(monomers) - 1):
                mstr += '.'

        pcts = acid_pct + glycol_pct
        pcts = [v / sum(pcts) for v in pcts] # Normalize total

        pct_str = '|'
        for j, p in enumerate(pcts):
            pct_str += '{:.4f}'.format(p)
            # if j < (len(pcts) - 1):
            pct_str += '|'
        
        full_inp = mstr + pct_str
        df_skeleton['master_chemprop_input'].append(full_inp)
        for j, y in enumerate(Y_target):
            df_skeleton[y].append(yi[j])

    df = pd.DataFrame(df_skeleton)

    if len(Y_target) > 1:
        df.to_csv('wdmpnn_IV_Tg.csv', index = False)
    else:
        df.to_csv('wdmpnn_{}.csv'.format(Y_target[0]), index = False)




if __name__ == '__main__':
    main()