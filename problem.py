import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Selectivity of higher education programs in France'
_target_column_name = 'selectivity_category'
_ignore_column_names = ['prop_tot_bt_brs', 'pct_tbf', 'pct_b', 'pct_acc_debutpp', 'nb_voe_pp_at', 'nb_voe_pc_bp', 'pct_acc_datebac', 'acc_tbf', 'pct_bours', 'pct_acc_finpp', 'nb_voe_pc_bt', 'pct_mention_nonrenseignee', 'pct_ab', 'acc_tb', 'nb_voe_pp', 'pct_aca_orig', 'voe_tot', 'pct_aca_orig_idf', 'voe_tot_f', 'acc_mention_nonrenseignee', 'pct_f']
_prediction_label_names = ["Très sélective", "Peu sélective", "Non sélective"]

cat_to_int = {'Très sélective': 2, 'Peu sélective': 1, 'Non sélective': 0}
int_to_cat = {2: 'Très sélective', 1: 'Peu sélective', 0: 'Non sélective'}

_prediction_label_int = [cat_to_int[cat] for cat in _prediction_label_names]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_int)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(name='bal_acc', precision=3, adjusted=False),
    rw.score_types.Accuracy(name='acc', precision=3),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].map(cat_to_int).fillna(-1).astype("int8").values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='./data/public'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='./data/public'):
    f_name = 'test.csv'
    return _read_data(path, f_name)