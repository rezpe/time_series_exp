import torch 
import os
import random
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
import pyro.distributions as dist

seq_length=168
horizon=13
seasonal_removal=True

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
