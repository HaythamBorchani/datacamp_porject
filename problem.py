import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Selectivity of higher education programs in France'
_target_column_name = 'select_form'
_ignore_column_names = ['session']
_prediction_label_names = ["formation non sélective", "formation sélective"]

cat_to_int = {'formation sélective': 1, 'formation non sélective': 0}
int_to_cat = {1: 'formation sélective', 0: 'formation non sélective'}

_prediction_label_int = [cat_to_int[cat] for cat in _prediction_label_names]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_int)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(name='bal_acc', precision=3, adjusted=False),
    rw.score_types.ROCAUC(name='auc', precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].map(cat_to_int).fillna(-1).astype("int8").values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.\\data\\public'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.\\data\\public'):
    f_name = 'test.csv'
    return _read_data(path, f_name)