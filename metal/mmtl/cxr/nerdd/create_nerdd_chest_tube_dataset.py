import pandas as pd
import numpy as np
import os

def split_dataframe(df, seed=1701):
    """
    Creates train, test, dev split from single dataframe
    """
    return np.split(df.sample(frac=1, random_state=seed), [int(.6*len(df)), int(.8*len(df))])

def print_class_balance(df, lab_col, split):
    counts = df.groupby(lab_col)[lab_col].count()
    bals = {k:v/len(df) for k,v in counts.items()}
    for k,v in bals.items():
        print(f'Split {split}, Class {k}: {np.round(v*100,2)} Percent')
    
# LOADING DATA AND OUTPUTTING TRAIN/TEST/DEV
dfs = {}
splits = ['train','dev','test']
nerdd_data_path = os.path.join(os.environ['CXRDATA'],'NERDD-CHEST-TUBE')
df_tube = pd.read_csv(os.path.join(nerdd_data_path,'chest_tube_master.csv'), index_col=0)
df_tube['img_path'] = df_tube['img_path'].apply(lambda x: '/'.join(x.split('/')[-2:]))
dfs['train'], dfs['dev'], dfs['test'] = split_dataframe(df_tube)
_ = [print_class_balance(dfs[split], 'chest_tube', split) for split in splits]

# Outputting to CSV
for split in splits:
    dfs[split].to_csv(os.path.join(nerdd_data_path,f'{split}.tsv'),sep='\t')