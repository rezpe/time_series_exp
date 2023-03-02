## Imports
import data_prep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Torch initialize 
import torch 
import os
import random
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
import pyro.distributions as dist

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

## Input data
seq_length=168
horizon=13
seasonal_removal=True

stations = ['28079017', '28079011', '28079048', '28079060', '28079036',
       '28079027', '28079056', '28079008', '28079038', '28079049',
       '28079055', '28079040', '28079059', '28079004', '28079018',
       '28079024', '28079058', '28079057', '28079039', '28079054',
       '28079016', '28079050', '28079035', '28079047']

fields=['SPA.NO2']

all_pairs = []
for station in stations:
    for field in fields:
        all_pairs.append([station,field])

# Station field pair
cluster_1 = all_pairs

conf=json.load(open("conf.conf"))
folder=conf["data_folder"]
X_train,y_train = get_train_data(folder,stations_fields,seasonal_removal,seq_length,horizon)
X_test, y_test, v_recover = get_test_data(folder,stations,seasonal_removal,seq_length,horizon)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values)

for key in X_test.keys():
    X_test[key]=torch.FloatTensor(X_test[key].values)
    y_test[key]=torch.FloatTensor(y_test[key].values)

from quantile_regression.tcn import *
config = { "batch_size":2028,
"epochs":11,
"ksize":12,
"lr":0.001001318,
"n_channels":250,
"regressor_size":999}

batch_size = config["batch_size"]
lr = config["lr"]
epochs = config["epochs"]
n_channels= config["n_channels"]
ksize = config["ksize"]
regressor_size= config["regressor_size"]

# Setup data
X_train_gf = X_train.reshape(len(X_train),1,len(X_train[0]))
train = TensorDataset(X_train_gf, y_train)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,num_workers=8)
from pytorch_lightning.callbacks import EarlyStopping
device = 0

trainer = pl.Trainer(max_epochs=epochs,
                      accelerator="gpu",
                      devices=[device])


tcn=TCN(n_channels=n_channels,k_size=ksize,regressor_size=regressor_size,lr=lr)
trainer.fit(model=tcn, train_dataloaders=train_loader)

from quantile_regression import metrics

errors = []

for station in X_test.keys():
    print(station)
    X_tests = X_test[station]
    y_tests = y_test[station]
    v_recovers = v_recover[station]

    X_test_gf = X_tests.reshape(len(X_tests),1,len(X_tests[0]))
    test = TensorDataset(X_test_gf, y_tests)
    test_loader = DataLoader(test, batch_size=256)

    allp={}
    for quantile in torch.arange(1,100,1):
        quantile = int(quantile)
        allp[quantile]=[]
    for p in trainer.predict(tcn,test_loader):
        for quantile in torch.arange(1,100,1):
            q_pred = p.icdf(quantile/100).reshape(-1).cpu().numpy()
            quantile = int(quantile)
            allp[quantile]=np.concatenate([allp[quantile],q_pred])
    pred_df = pd.DataFrame(allp)

    real_pred_df = pd.DataFrame()
    for col in pred_df:
        real_pred_df[col]= np.expm1(pred_df[col] + v_recovers["trend_norm"].values + v_recovers["seasonal"].values)
    real_y_test = np.expm1(y_tests.numpy().reshape(-1) + v_recovers["trend_norm"].values + v_recovers["seasonal"].values)

    tcn_metrics = metrics.evaluate(real_pred_df.values,real_y_test)
    tcn_metrics["station"]=station
    errors.append(tcn_metrics)

pd.DataFrame(errors)


