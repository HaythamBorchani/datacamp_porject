from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['contrat_etab', 'fili', 'list_com', 'tri']
num_cols = ['capa_fin', 'nb_voe_pp_bg', 'nb_voe_pp_bg_brs', 'nb_voe_pp_bt', 'nb_voe_pp_bt_brs', 'nb_voe_pp_bp', 'nb_voe_pp_bp_brs', 'nb_voe_pc', 'nb_voe_pc_bg', 'nb_voe_pc_at', 'nb_cla_pp', 'nb_cla_pc', 'nb_cla_pp_bg', 'nb_cla_pp_bg_brs', 'nb_cla_pp_bt', 'nb_cla_pp_bt_brs', 'nb_cla_pp_bp', 'nb_cla_pp_bp_brs', 'nb_cla_pp_at', 'prop_tot', 'acc_tot', 'acc_tot_f', 'acc_pp', 'acc_pc', 'acc_debutpp', 'acc_datebac', 'acc_finpp', 'acc_brs', 'acc_neobac', 'acc_bg', 'acc_bt', 'acc_bp', 'acc_at', 'acc_sansmention', 'acc_ab', 'acc_b', 'acc_bg_mention', 'acc_bt_mention', 'acc_bp_mention', 'acc_aca_orig', 'acc_aca_orig_idf', 'pct_etab_orig', 'pct_neobac', 'pct_sansmention', 'pct_tb', 'pct_bg', 'pct_bg_mention', 'pct_bt', 'pct_bt_mention', 'pct_bp', 'pct_bp_mention', 'prop_tot_bg', 'prop_tot_bg_brs', 'prop_tot_bt', 'prop_tot_bp', 'prop_tot_bp_brs', 'prop_tot_at', 'cod_aff_form', 'part_acces_gen', 'part_acces_tec', 'part_acces_pro']
class Classifier(BaseEstimator):
    def __init__(self):
    
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
                ])
        self.model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        self.pipe = Pipeline(steps=[('preprocessor', self.preprocessor),
                           ('model', self.model)])

    def fit(self, X, y):
        return self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)