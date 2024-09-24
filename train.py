import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import mlflow
import os

def read_data():
    df = pd.read_csv('br_inep_ideb_brasil.csv')
    df = df.drop('projecao', axis=1)

    df = df.drop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26], axis=0)

    X = df[['taxa_aprovacao', 'indicador_rendimento', 'nota_saeb_matematica', 'nota_saeb_lingua_portuguesa',
              'nota_saeb_media_padronizada']]
    y = df['ideb']

    return X, y

def process_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def create_model(X):
    model = LinearRegression()

    return model

def config_mlflow():
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'brunoFNIR'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c64e879cbe5a3f32f5d7d5a392678e2c9fe10e85'
    mlflow.set_tracking_uri('https://dagshub.com/brunoFNIR/ideb_prediction.mlflow')

    mlflow.tensorflow.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)
def train_model(model, X_train, y_train, is_train=True):
    with mlflow.start_run(run_name='ideb_predict') as run:
        model.fit(X_train, y_train)
    if is_train:
        run_uri = f'runs:/{run.info.run_id}'
        mlflow.register_model(run_uri, 'desenvolvimento_educacao-basica')

if __name__ == "__main__":
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X)
    config_mlflow()
    train_model(model, X_train, y_train)