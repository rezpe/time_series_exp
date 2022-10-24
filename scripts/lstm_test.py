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

X_train = torch.FloatTensor(X_train.values)
for key in X_test.keys():
    X_test[key]=torch.FloatTensor(X_test[key].values)
y_train = torch.FloatTensor(y_train.values)
for key in y_test.keys():
    y_test[key]=torch.FloatTensor(y_test[key].values)

## Model
import pytorch_lightning as pl
import pyro.distributions as dist

class LSTM(pl.LightningModule):
    def __init__(self,hidden_dim=32, n_layers=4, regressor_size=200, lr=0.01):
        super().__init__()
        
        self.lr = lr
        self.regressor_size = regressor_size
        
        self.input_size = 1
        self.output_size=1
        self.hidden_dim, self.n_layers = hidden_dim, n_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Regressor
        self.regressor_mu = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim,self.regressor_size),
            nn.ReLU(),
            nn.LayerNorm(self.regressor_size),
            nn.Linear(self.regressor_size,1)
        )
        
        self.regressor_s = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim,self.regressor_size),
            nn.ReLU(),
            nn.LayerNorm(self.regressor_size),
            nn.Linear(self.regressor_size,1),
            nn.Softplus(),  # enforces positivity
        )
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        
        out, (h,c) = self.lstm(x)
        out = out[:, -1, :]
        mu = self.regressor_mu(out)
        s = self.regressor_s(out)
        base_dist = dist.Normal(mu, s)
        return base_dist

        return out
    
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

# SMAC
from quantile_regression import metrics

def train_model(config):
    
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    epochs = config["epochs"]
    lr= config["lr"]
    n_layers= config["n_layers"]

    # Setup data
    X_train_gf = X_train.reshape([len(X_train), -1, 1])
    train = TensorDataset(X_train_gf, y_train)
    train, valid = random_split(train,[int(len(train)*0.95), len(train)-int(len(train)*0.95)])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=256)

    from pytorch_lightning.callbacks import EarlyStopping

    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="gpu",
                         #max_time="00:00:20:00",
                         callbacks=[EarlyStopping("val_loss",
                                                  verbose=True, 
                                                  patience=1,
                                                  mode="min")])

    lstm=LSTM(hidden_dim=hidden_dim, n_layers=n_layers,lr=lr)
    trainer.fit(model=lstm, train_dataloaders=train_loader,val_dataloaders=valid_loader)

    error = trainer.test(lstm,valid_loader)
    
    at_log(f"lstm_smac:{datetime.datetime.now().strftime('%Y_%m_%d')} seasonal: {seasonal_removal}",
       [{"nll":error}],
       str(lgbmodel),
       str(stations)+""+str(fields),
      "test")

    return error

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter,UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

# Define your hyperparameters
configspace = ConfigurationSpace()
configspace.add_hyperparameter(UniformIntegerHyperparameter("batch_size", 30, 4000))
configspace.add_hyperparameter(UniformIntegerHyperparameter("epochs", 2, 20))
configspace.add_hyperparameter(UniformIntegerHyperparameter("hidden_dim", 8, 150))
configspace.add_hyperparameter(UniformFloatHyperparameter("lr", 0.0001, 0.1, default_value=0.001, log=True))
configspace.add_hyperparameter(UniformIntegerHyperparameter("n_layers", 1, 2))

# Provide meta data for the optimization
scenario = Scenario({
    "run_obj": "quality",  
    "runcount-limit": 50,  
    "cs": configspace,
    "abort_on_first_run_crash":False
})

smac = SMAC4BB(scenario=scenario, tae_runner=train_model)
best_found_config = smac.optimize()

print("Best Configuration:")
print(best_found_config)

cfg_traj = [ traj.incumbent for traj in smac.trajectory]
plt.plot([smac.runhistory.get_cost(cfg) if cfg in cfg_traj else None for cfg in smac.runhistory.get_all_configs()], 'bo')
plt.plot([smac.runhistory.get_cost(cfg)  for cfg in  smac.runhistory.get_all_configs()])
plt.title('SMAC scores')
plt.legend(['Tested configuration', 'Selected incumbent'])
plt.xlabel('Number of iterations')
plt.ylim([0,100])
plt.ylabel('Score')
plt.save("results.png")