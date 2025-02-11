import numpy as np
import pandas as pd
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


def main(X_train, y_train):

    model = MultinomialNB()

    params = {'alpha': np.arange(0, 10, 0.5),
              'fit_prior': [True, False]
              }

    gs = GridSearchCV(model, params, scoring='f1', cv=3, n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_

    joblib.dump(best_model, 'model/best_model.pkl')


if __name__ == '__main__':
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    main(X_train, y_train)
