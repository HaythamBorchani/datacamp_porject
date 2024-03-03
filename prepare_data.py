import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

# 1 - read in or download the data.
df = pd.read_csv(os.path.join('data','fr-esr-parcoursup.csv'),sep=';')

low_threshold = 20
high_threshold = 80
#create a new column 'selectivity' based on the 'taux_acces_ens' column
def add_selectivity(df):
    #drop rows with NaN in the 'taux_acces_ens' column
    df = df.dropna(subset=['taux_acces_ens'])

    conditions = [
        (df['taux_acces_ens'] <= low_threshold),
        (df['taux_acces_ens'] > low_threshold) & (df['taux_acces_ens'] <= high_threshold),
        (df['taux_acces_ens'] > high_threshold)
    ]

    choices = ['tres selective', 'selective', 'peu selective']
    df['selectivity'] = np.select(conditions, choices, default='Unknown')

    return df

df = add_selectivity(df) 
df = df.dropna(subset=['taux_acces_ens'])
# 2 - Perform any data cleaning and split into private train/test subsets,
# if required. Neither steps required in this case.

# 3 - Split Public/Private data.
df_public, df_private = train_test_split(df, test_size=0.7, random_state=57, stratify=df['selectivity'])
    # specify the random_state to ensure reproducibility

# 4 - Split public train/test subsets. 

df_public_train, df_public_test = train_test_split(
    df_public, test_size=0.2, random_state=57, stratify=df_public['selectivity'])
    # specify the random_state to ensure reproducibility

# Define the directory path
public_path = 'data\\public'

# Check if the directory exists, if not, create it
if not os.path.exists(public_path):
    os.makedirs(public_path)

df_public_train.to_csv(os.path.join(public_path, 'train.csv'), index=False)
df_public_test.to_csv(os.path.join(public_path, 'test.csv'), index=False)


# 5 - Split private train/test subsets.

# Define the directory path
private_path = 'data'

# Check if the directory exists, if not, create it
if not os.path.exists(private_path):
    os.makedirs(private_path)

df_private_train, df_private_test = train_test_split(
    df_private, test_size=0.2, random_state=57, stratify=df_private['selectivity'])
    # specify the random_state to ensure reproducibility
df_private_train.to_csv(os.path.join(private_path, 'train.csv'), index=False)
df_private_test.to_csv(os.path.join(private_path, 'test.csv'), index=False)
