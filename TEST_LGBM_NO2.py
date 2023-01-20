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

## Airtable logging
from airtable import airtable
from datetime import datetime
import json
at = airtable.Airtable('appXOFgDU3lgoOYQI', open("airtable").read())

table_name="tblZQ3UBMflriV9Oi"

def at_log(name,metric,model,data,typet):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    at.create(table_name, dict(name=name,metric=json.dumps(metric),time=time,model=str(model),data=data,typet=typet))

## Input data
seq_length=168
horizon=13
seasonal_removal=True

stations = ['28079017', '28079011', '28079048', '28079060', '28079036',
       '28079027', '28079056', '28079008', '28079038', '28079049',
       '28079055', '28079040', '28079059', '28079004', '28079018',
       '28079024', '28079058', '28079057', '28079039', '28079054',
       '28079016', '28079050', '28079035', '28079047']

stations=stations[:1]

fields=['SPA.NO2','AEMET.BLH', 'AEMET.SP', 'AEMET.T2M', 'AEMET.TP',
        'AEMET.WS', 'MACC.NO2', 'MACC.O3', 'MACC.PM10',
       'MACC.PM25',  'SPA.O3', 'SPA.PM10', 'SPA.PM25']

fields=fields[:1]

X_train,y_train, X_test, y_test, v_recover = data_prep.get_data(stations,fields,seasonal_removal,seq_length,horizon)

from sklearn.model_selection import train_test_split
import quantile_regression
from quantile_regression.lgba import TotalLGBQuantile
from quantile_regression import metrics


config = {The best config from the execution results}

n_estimators = config["n_estimators"]
max_depth = config["max_depth"]

lgbmodel = TotalLGBQuantile(n_estimators=config["n_estimators"],max_depth=config["max_depth"])
lgbmodel.fit(X_train,y_train)


errors = {}

for station in test_data.keys():
    X_test = test_data[station]
    y_test = test_data[station]
    v_recover = test_data[station]

    predictions = lgbmodel.predict(X_test)

    predictions.values with vrecover
    error = metrics.evaluate(reconstructed,y_test.values.reshape(-1))

    plot(reconstructed, y_test)



