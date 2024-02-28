import os
import mlflow
import pandas as pd
import torch
from sklearn.datasets import load_iris
if __name__ == "__main__":
    # setting
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://127.0.0.1:9000"
    os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5001"
    os.environ['AWS_ACCESS_KEY_ID'] = "minio"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "qwer1234"

    model_name = "iris_pytorch"
    run_id = "59e8a439df2a41b893fba07d128515b7"
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}")
    # data
    X, y = load_iris(return_X_y=True, as_frame=True)
    mapper = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    y = pd.get_dummies(y.apply(lambda x: mapper[x])).astype('int')
    DEVICE = torch.device('mps')
    X_train = torch.tensor(X.iloc[:20].values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(X_train)
    print(pred)
