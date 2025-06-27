import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import mlflow.sklearn
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_wine()
# print(data)
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(x_train.shape)
# print(x_test.shape)
n_estimators=100
max_depth=100

mlflow.set_experiment('Wine-Experiment')
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    mlflow.log_metric('accuracy_score', accuracy)
    mlflow.log_param('Estimators', n_estimators)
    mlflow.log_param('Max-depth', max_depth)


    confusion_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix, annot=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(rf, "Random Forest Classifier")
    mlflow.set_tags({"author": "Noman", "Project":"Wine Classification"})
    print(accuracy)

