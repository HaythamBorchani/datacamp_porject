import pandas as pd
from sklearn.model_selection import train_test_split
import os


# 1 - read in or download the data.
df = pd.read_csv(os.path.join('data','fr-esr-parcoursup.csv'),sep=';')

# Step 2: Creation of the Classification Target

# The goal here is to refine the understanding of selectivity among educational formations.
# Initial data classification ('select_form') provides a binary distinction:
# - "formation non sélective": Indicating programs with less competitive admission processes.
# - "formation sélective": Denoting programs with more stringent admission criteria.
# Given the imbalance in the dataset, with approximately 13,000 "formation sélective" and only 3,000 "formation non sélective",
# there's a need for a more nuanced categorization to better capture the spectrum of selectivity.

# The enhancement involves expanding the binary classification into three distinct categories:
# 1. "Non sélective": Programs that are broadly accessible, with high acceptance rates.
# 2. "Peu sélective": Programs with moderate selectivity, indicating a balanced level of competition.
# 3. "Très sélective": Highly competitive programs, characterized by low acceptance rates.

# This categorization relies on the 'Taux_acces_ens', a key metric representing the access rate.
# It's calculated as the ratio of the number of applicants whose ranking is lower than or equal to the ranking of the last admitted applicant in their group,
# to the total number of applicants who expressed a preference for the program during the principal phase of application.
# Implementing these three labels allows for a more detailed analysis of selectivity across educational formations, aiming to provide insights that the original binary classification could not capture.

def categorize_by_access_rate(row, high_threshold):
    # If select_form is 0 or the value is missing, it's automatically 'non selective'
    if row['select_form'] == 0:
        return 'Non sélective'
    # If select_form is 1, then check the 'taux_acces_ens' against the high_threshold
    elif row['select_form'] == 1:
        if row['taux_acces_ens'] > high_threshold:
            return 'Peu sélective'
        else:
            return 'Très sélective":'
    else:
        # Handle cases where 'select_form' is not 0, 1, or NaN, if necessary
        return 'unknown'

# Drop rows where 'select_form' is null
df = df.dropna(subset=['select_form'])
select_form_mapping ={'formation sélective': 1, 'formation non sélective': 0}

# Map the string descriptions to numeric values
df['select_form'] = df['select_form'].map(select_form_mapping)

high_threshold = 70
# Apply the function to the DataFrame
df['selectivity_category'] = df.apply(lambda row: categorize_by_access_rate(row, high_threshold), axis=1)


# 3 - Perform any data cleaning
list_columns_to_be_removed =[ "region_etab_aff" ,"acad_mies", "form_lib_voe_acc", "dep", "dep_lib", "fil_lib_voe_acc","ville_etab", "lib_comp_voe_ins","detail_forma", "g_olocalisation_des_formations","lib_for_voe_ins","lien_form_psup","etablissement_id_paysage","composante_id_paysage",'nb_voe_pp_internat', 'nb_cla_pp_internat', 'nb_cla_pp_pasinternat','acc_internat', 'acc_term', 'acc_term_f', 'lib_grp1', 'ran_grp1','lib_grp2', 'ran_grp2', 'lib_grp3', 'ran_grp3', 'detail_forma2',"lib_for_voe_ins","g_ea_lib_vx","cod_uai", "taux_acces_ens" , "select_form","session"]
df= df.drop(list_columns_to_be_removed, axis=1)

# 4 - Create a composite key
# For categorical features
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
df['stratify_key'] = df[categorical_features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# In order to split each split, stratify_key count must be greater than 5
grouped_sorted = df.groupby('stratify_key').size().reset_index(name='count').sort_values(by='count', ascending=False)
keys_to_keep = grouped_sorted[grouped_sorted['count'] >= 5]['stratify_key']
df_filtered = df[df['stratify_key'].isin(keys_to_keep)]

# 5 - Split Public/Private data.
df_public, df_private = train_test_split(df_filtered, test_size=0.7, random_state=57, stratify=df_filtered['stratify_key'])    # specify the random_state to ensure reproducibility

# 4 - Split public train/test subsets. 

df_public_train, df_public_test = train_test_split(
    df_public, test_size=0.2, random_state=57, stratify=df_public['stratify_key'])
    # specify the random_state to ensure reproducibility

# Define the directory path
public_path = 'data/public'

# Check if the directory exists, if not, create it
if not os.path.exists(public_path):
    os.makedirs(public_path)
df_public_train = df_public_train.drop('stratify_key', axis=1)
df_public_test = df_public_test.drop('stratify_key', axis=1)
df_public_train.to_csv(os.path.join(public_path, 'train.csv'), index=False)
df_public_test.to_csv(os.path.join(public_path, 'test.csv'), index=False)


# 5 - Split private train/test subsets.

# Define the directory path
private_path = 'data'

# Check if the directory exists, if not, create it
if not os.path.exists(private_path):
    os.makedirs(private_path)

df_private_train, df_private_test = train_test_split(
    df_private, test_size=0.2, random_state=57, stratify=df_private['stratify_key'])
    # specify the random_state to ensure reproducibility
df_private_train = df_private_train.drop('stratify_key', axis=1)
df_private_test = df_private_test.drop('stratify_key', axis=1)
df_private_train.to_csv(os.path.join(private_path, 'train.csv'), index=False)
df_private_test.to_csv(os.path.join(private_path, 'test.csv'), index=False)
