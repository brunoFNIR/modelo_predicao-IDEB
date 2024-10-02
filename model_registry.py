import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def mlflow_config():
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'brunoFNIR'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'f3d7f7d8e8e9a09476a105d2c051177ec8d2d45e'
    mlflow.set_tracking_uri('https://dagshub.com/brunoFNIR/ideb_prediction.mlflow')

    mlflow.tensorflow.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)

def read_data():
    url = 'raw.githubusercontent.com'
    username = 'brunoFNIR'
    repository = 'modelo_predicao-IDEB'
    file_name = 'br_inep_ideb_brasil.csv'
    data = pd.read_csv(f'https://{url}/{username}/{repository}/main/{file_name}')

    data = data.drop('projecao', axis=1)

    data = data.drop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26],
                 axis=0)

    X = data[['taxa_aprovacao', 'indicador_rendimento', 'nota_saeb_matematica', 'nota_saeb_lingua_portuguesa',
            'nota_saeb_media_padronizada']]
    y = data['ideb']

    return X, y

mlflow_config()
with mlflow.start_run():
    X, y = read_data()
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        input_example=X_train,
        registered_model_name="ideb_predict_linear-regression")