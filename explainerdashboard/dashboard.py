import joblib
import pandas as pd

from explainerdashboard import ClassifierExplainer, ExplainerDashboard


def main(X_test, y_test):
    model = joblib.load('model/best_model.pkl')
    explainer = ClassifierExplainer(model, X_test, y_test)
    db = ExplainerDashboard(explainer)
    db.to_yaml("dashboard.yaml", explainerfile="explainer.dill", dump_explainer=True)


if __name__ == '__main__':
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    main(X_test, y_test)
