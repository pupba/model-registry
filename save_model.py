import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# parser
import os
from argparse import ArgumentParser

# mlflow
import mlflow


def loadData() -> tuple:
    X, y = load_iris(return_X_y=True, as_frame=True)
    mapper = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    y = pd.get_dummies(y.apply(lambda x: mapper[x])).astype('int')
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


class DNN(nn.Module):
    def __init__(self, inp, outp):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(inp, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, outp)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def model_make(input_size: int, output_size: int):
    DEVICE = torch.device('mps')
    model = DNN(input_size, output_size)
    model.to(DEVICE)  # use mps
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_f = nn.CrossEntropyLoss()
    return model, opt, loss_f, DEVICE


def get_accuracy(pred_arr, original_arr):
    pred_arr = pred_arr.cpu().detach().numpy()
    original_arr = original_arr.cpu().detach().numpy()
    final_pred = []
    final_origin = []
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
        final_origin.append(np.argmax(original_arr[i]))
    count = 0

    for i in range(len(original_arr)):
        if final_pred[i] == final_origin[i]:
            count += 1
    return round(count/len(final_pred), 4)


def fit(X_train, Y_train, X_test, Y_test, model, opt, loss_f, DEVICE, epochs=10):
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
    Y_train = torch.tensor(Y_train.values, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
    Y_test = torch.tensor(Y_test.values, dtype=torch.float32).to(DEVICE)
    loss_ = []
    acc_ = []
    for epoch in range(1, epochs+1):
        opt.zero_grad()
        # forward feed
        y_pred = model(X_train)
        # loss
        loss = loss_f(y_pred, Y_train)
        print(f"epoch{epoch} {'-'*20} loss: {loss:.4f}")
        loss_.append(round(loss.cpu().detach().numpy().item(), 4))

        # backward
        loss.backward()
        # update weight
        opt.step()
        with torch.no_grad():
            acc = get_accuracy(y_pred, Y_train)
            acc_.append(acc)
    return model, loss_, acc_


if __name__ == "__main__":
    # mlflow set env
    # 값들은 따로 실제 사용시 secret 만들어서 저장하기...
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://127.0.0.1:9000"
    os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5001"
    os.environ['AWS_ACCESS_KEY_ID'] = "minio"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "qwer1234"
    # parser
    parser = ArgumentParser()
    parser.add_argument("--model-name", dest="model_name",
                        type=str, default="Pytorch_DNN_Iris")
    args = parser.parse_args()
    mlflow.set_experiment("new-exp")
    # data
    X_train, X_test, Y_train, Y_test = loadData()
    model, opt, loss_f, DEVICE = model_make(X_train.shape[1], Y_train.shape[1])

    signature = mlflow.models.infer_signature(
        model_input=X_train, model_output=Y_train)
    input_sample = X_train.iloc[:10]

    mlflow.pytorch.autolog()
    # mlflow
    with mlflow.start_run(run_name="iris_pytorch_Linear"):
        model, loss, acc = fit(X_train, Y_train, X_test, Y_test,
                               model, opt, loss_f, DEVICE, epochs=100)
        for i in range(len(loss)):
            mlflow.log_metric("train_loss", loss[i], i+1)
            mlflow.log_metric("train_acc", acc[i], i+1)
        mlflow.pytorch.log_model(
            model, args.model_name, signature=signature,
            input_example=input_sample
        )
