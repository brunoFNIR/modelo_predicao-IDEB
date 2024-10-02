import os
import mlflow
from fastapi import FastAPI

app = FastAPI(title='IDEB Predict - API',
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get API health"
                  },
                  {
                    "name": "Prediction",
                    "description": "IDEB Prediction"
                  }
              ])

def mlflow_config():
    print("reading model")
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '600'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'brunoFNIR'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'f3d7f7d8e8e9a09476a105d2c051177ec8d2d45e'
    mlflow.set_tracking_uri('https://dagshub.com/brunoFNIR/ideb_prediction.mlflow')

def load_model():
    print("setting mlflow")
    mlflow_config()
    print("creating client")
    client = mlflow.MlflowClient(tracking_uri='https://dagshub.com/brunoFNIR/ideb_prediction.mlflow')
    print("getting registered model")
    registered_model = client.get_registered_model('ideb_predict_linear-regression')
    print("reading model...")
    run_id = registered_model.latest_versions[-1].run_id
    loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/model')

    print("loaded.")
    return loaded_model

@app.get(path='/',
         tags=["Health"])
def api_health():
    return {"status":"healthy"}

@app.post(path='/predict',
          tags=["Prediction"])
def predict():
    load_model()
    return {"prediction": "test"}