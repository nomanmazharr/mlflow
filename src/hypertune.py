import mlflow.data
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub

data = load_breast_cancer()
dagshub.init(repo_owner='nomanmazharr', repo_name='mlflow', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/nomanmazharr/mlflow.mlflow')
x = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

params = {
    'n_estimators' : [20, 50, 100],
    'max_depth' : [None, 20, 30, 50]
}
rf = RandomForestClassifier()
gridsearch = GridSearchCV(cv=5, param_grid=params, n_jobs=-1, verbose=2, estimator=rf)


mlflow.set_experiment('breast-cancer-random-forest')

with mlflow.start_run() as parent:
    gridsearch.fit(x_train, y_train)

    # log all the child runs
    for i in range(len(gridsearch.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(gridsearch.cv_results_["params"][i])
            mlflow.log_metric("accuracy", gridsearch.cv_results_["mean_test_score"][i])
    
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_

    mlflow.log_params(best_params)

    mlflow.log_metric('accuracy', best_score)

    # logging training data
    train_df = x_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, 'training')

    # logging test data
    test_df = x_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, 'testing')

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(gridsearch.best_estimator_, "Random Forest-2")

    mlflow.set_tag('author', 'Noman Mazhar')

    print(best_params)
    print(best_score)