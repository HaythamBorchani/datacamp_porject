from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Classifier(BaseEstimator):
    def __init__(self):
        self.transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        self.model = LogisticRegression(max_iter=500)
        self.pipe = make_pipeline(self.transformer, self.model)

    def fit(self, X, y):
        X = X.select_dtypes(include="number")
        print(y)
        self.pipe.fit(X, y)

    def predict(self, X):
        X = X.select_dtypes(include="number")
        print(self.pipe.predict(X))
        return self.pipe.predict(X)

    def predict_proba(self, X):
        X = X.select_dtypes(include="number")
        return self.pipe.predict_proba(X)