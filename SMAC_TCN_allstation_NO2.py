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

stations_fields = []
for station in stations:
    for field in fields:
        stations_fields.append([station, field])

X_train,y_train = data_prep.get_train_data(stations_fields,seasonal_removal,seq_length,horizon)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values)
## Model

class TCN(pl.LightningModule):
    def __init__(self,n_channels=128,k_size=3,regressor_size=200,lr=0.01):
        super().__init__()
        
        self.n_channels, self.k_size, self.regressor_size, self.lr = n_channels, k_size, regressor_size, lr

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, 
                      out_channels=self.n_channels, 
                      kernel_size=self.k_size, 
                      padding=(int(self.k_size/2))),
            nn.MaxPool1d(2),
            nn.ReLU(),
        )

        self.regressor_mu = nn.Sequential(
            nn.LayerNorm(self.n_channels*int(seq_length/2)),
            nn.Linear(self.n_channels*int(seq_length/2),self.regressor_size),
            nn.ReLU(),
            nn.LayerNorm(self.regressor_size),
            nn.Linear(self.regressor_size,1)
        )
        
        self.regressor_s = nn.Sequential(
            nn.LayerNorm(self.n_channels*int(seq_length/2)),
            nn.Linear(self.n_channels*int(seq_length/2),self.regressor_size),
            nn.ReLU(),
            nn.LayerNorm(self.regressor_size),
            nn.Linear(self.regressor_size,1),
            nn.Softplus(),  # enforces positivity
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,self.n_channels*x.shape[2])
        mu = self.regressor_mu(x)
        s = self.regressor_s(x)
        base_dist = dist.Normal(mu, s)
        return base_dist
    
    def training_step(self, batch, batch_idx):
        x, y = batch  
        base_dist = self(x)
        loss = -base_dist.log_prob(y).mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch  
        base_dist = self(x)
        loss = -base_dist.log_prob(y).mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch  
        base_dist = self(x)
        loss = -base_dist.log_prob(y).mean()
        # We would have to modify this for better aggregation
        # https://forums.pytorchlightning.ai/t/understanding-logging-and-validation-step-validation-epoch-end/291/2
        self.log("test_loss", loss)
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.lr/100)
        return optimizer

## Smac tests
from quantile_regression import metrics

def train_model(config):
    
    print(config)
    
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    n_channels= config["n_channels"]
    ksize = config["ksize"]
    regressor_size= config["regressor_size"]

    # Setup data
    X_train_gf = X_train.reshape(len(X_train),1,len(X_train[0]))
    train = TensorDataset(X_train_gf, y_train)
    train, valid = random_split(train,[int(len(train)*0.95), len(train)-int(len(train)*0.95)])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,num_workers=8)
    valid_loader = DataLoader(valid, batch_size=256)

    from pytorch_lightning.callbacks import EarlyStopping
    # 16GB (0 and 3) + 32GB (1 and 2)
    device = 1

    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="gpu",
                         devices=[device],
                         #max_time="00:00:20:00",
                         #callbacks=[EarlyStopping("val_loss",
                         #                         verbose=True, 
                         #                         patience=1,
                         #                         mode="min")]
                         )
    print(device)
    tcn=TCN(n_channels=n_channels,k_size=ksize,regressor_size=regressor_size,lr=lr)
    trainer.fit(model=tcn, train_dataloaders=train_loader,val_dataloaders=valid_loader)

    error = trainer.test(tcn,valid_loader)
    
    at_log(f"tcn_smac:{datetime.now().strftime('%Y_%m_%d')} seasonal: {seasonal_removal}",
       [{"nll":error}],
       str(tcn),
       str(stations)+""+str(fields),
      "test")

    return error[0]["test_loss"]

# Tests run
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

# Define your hyperparameters
configspace = ConfigurationSpace()
configspace.add_hyperparameter(UniformIntegerHyperparameter("batch_size", 30, 4000))
configspace.add_hyperparameter(UniformIntegerHyperparameter("epochs", 5, 40))
configspace.add_hyperparameter(UniformIntegerHyperparameter("n_channels", 8, 256))
configspace.add_hyperparameter(UniformIntegerHyperparameter("ksize", 3, 20))
configspace.add_hyperparameter(UniformIntegerHyperparameter("regressor_size", 50, 1000))
configspace.add_hyperparameter(UniformFloatHyperparameter("lr", 0.001, 0.04, default_value=0.01, log=True))

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

total.to_csv(f"results/tcn_s_{len(stations)}_f_{len(fields)}_t_{datetime.now().strftime('%Y_%m_%d')}.csv")