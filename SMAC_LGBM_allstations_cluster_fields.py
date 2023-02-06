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

#stations=stations[:1]

fields=['SPA.NO2','AEMET.BLH', 'AEMET.SP', 'AEMET.T2M', 'AEMET.TP',
        'AEMET.WS', 'MACC.NO2', 'MACC.O3', 'MACC.PM10',
       'MACC.PM25',  'SPA.O3', 'SPA.PM10', 'SPA.PM25']

fields=fields[:1]

stations_fields = pd.read_csv("features_cluster/keep_ts.csv").values

X_train,y_train = data_prep.get_train_data(stations_fields,seasonal_removal,seq_length,horizon)

from sklearn.model_selection import train_test_split
import quantile_regression
from quantile_regression.lgba import TotalLGBQuantile
from quantile_regression import metrics

def train_model(config):
    
    n_estimators = config["n_estimators"]
    max_depth = config["max_depth"]
    
    print(config)

    X_traint, X_train_val, y_traint, y_train_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    lgbmodel = TotalLGBQuantile(n_estimators=config["n_estimators"],max_depth=config["max_depth"])
    lgbmodel.fit(X_traint,y_traint)
    
    predictions = lgbmodel.predict(X_train_val)
    error = metrics.evaluate(predictions.values,y_train_val.values.reshape(-1))
    
    at_log(f"lgbm_smac_s{len(stations)}_f{len(fields)}:{datetime.now().strftime('%Y_%m_%d')}",
       error,
       str(lgbmodel),
       str(stations)+""+str(fields),
      "test")
    
    return error["crps"]

# Tests run
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

# Define your hyperparameters
configspace = ConfigurationSpace()
configspace.add_hyperparameter(UniformIntegerHyperparameter("n_estimators", 1000, 3000))
configspace.add_hyperparameter(UniformIntegerHyperparameter("max_depth", 10, 15))

# Provide meta data for the optimization
scenario = Scenario({
    "run_obj": "quality",  
    "runcount-limit": 150,  
    "cs": configspace,
    "abort_on_first_run_crash": False
})

smac = SMAC4BB(scenario=scenario, tae_runner=train_model)
best_found_config = smac.optimize()

print("Best Configuration:")
print(best_found_config)

rh = smac.get_runhistory()
values=[]
for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in rh.data.items():
   values.append((cost, time, status, starttime, endtime, additional_info))
df=pd.DataFrame(values)
df.columns=("cost", "time", "status", "starttime", "endtime", "additional_info")
df=df.sort_values("starttime")

configs = pd.DataFrame([dict(cfg) for cfg in smac.runhistory.get_all_configs()])

total = pd.concat([df,configs],axis=1)

total.to_csv(f"results/lgbm_s_{len(stations)}_f_{len(fields)}_t_{datetime.now().strftime('%Y_%m_%d')}.csv")