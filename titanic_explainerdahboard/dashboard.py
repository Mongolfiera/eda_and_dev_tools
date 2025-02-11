from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import *
from explainerdashboard.datasets import *

X_train, y_train, X_test, y_test = titanic_survive()
model = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test,
                                cats=["Sex", 'Deck', 'Embarked'],
                                labels=['Not Survived', 'Survived'],
                                descriptions=feature_descriptions)

db = ExplainerDashboard(explainer)
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
