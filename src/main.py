import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
# print(data)

X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(x_train.shape)
# print(x_test.shape)
n_estimators=50
max_depth=100

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    mlflow.log_metric('accuracy_score', accuracy)
    mlflow.log_param('Estimators', n_estimators)
    mlflow.log_param('Max-depth', max_depth)

    print(accuracy)

