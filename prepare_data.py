import pandas as pd
from sklearn.model_selection import train_test_split
import os


# 1 - read in or download the data.
df = pd.read_csv(os.path.join('data','fr-esr-parcoursup.csv'),sep=';')
# 2 - Perform any data cleaning and split into private train/test subsets,
# if required. Neither steps required in this case.

# 3 - Split Public/Private data.
df_public, df_private = train_test_split(df, test_size=0.7, random_state=57, stratify=df['select_form'])
    # specify the random_state to ensure reproducibility

# 4 - Split public train/test subsets. 

df_public_train, df_public_test = train_test_split(
    df_public, test_size=0.2, random_state=57)
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
    df_private, test_size=0.2, random_state=57)
    # specify the random_state to ensure reproducibility
df_private_train.to_csv(os.path.join(private_path, 'train.csv'), index=False)
df_private_test.to_csv(os.path.join(private_path, 'test.csv'), index=False)
