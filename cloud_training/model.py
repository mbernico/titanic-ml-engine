from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FareTransformer(BaseEstimator, TransformerMixin):

    def transform(self, fare, **transform_params):
        fare = fare.clip(0, np.percentile(fare, 99))
        return fare.astype('int')

    def fit(self, X, y=None, **fit_params):
        return self


def build_model():
    sex_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehotencoder', OneHotEncoder())
    ])

    embarked_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehotencoder', OneHotEncoder())
    ])

    other_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    age_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
        # we can add a custom model transformer later
    ])

    fare_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fare_transformer', FareTransformer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('sex', sex_transformer, ['Sex']),
            ('age', age_transformer, ['Age']),
            ('embarked', embarked_transformer, ['Embarked']),
            ('other variables', other_transformer, ['Pclass', 'SibSp', 'Parch']),
            ('fare', fare_transformer, ['Fare']),
        ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('randomForest', RandomForestClassifier(n_estimators=100))
    ])

    return model