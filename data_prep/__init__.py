import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import multiprocessing

def process_station(pair):
    folder,station_field,seasonal_removal,horizon,seq_length = pair
    station,field = station_field

    # Reading input file
    df = pd.read_csv(f"{folder}/{station}.csv",sep=";")
    # Limiting Date
    df = df[df["DATE"]<"2020-01-01"]
    
    print(f"{station}: {field}")

    tdf = df[["DATE",field]].copy()

    #Stop is there are null values
    if (tdf[field].isnull().sum()>0):
        return

    # Log transform
    tdf[field]=np.log1p(tdf[field])

    if seasonal_removal:
        #STL Decomp
        stl = seasonal_decompose(tdf[field], model="additive",period=24)
        tdf[field]=tdf[field]-stl.seasonal
        # We keep the trend at prediction time:
        # norm_field = field-trend_pred (horizon shifted)
        tdf["trend_norm"]=stl.trend.shift(horizon)
        tdf["seasonal"]=stl.seasonal
    else:
        tdf["trend_norm"]=0
        tdf["seasonal"]=0

    # We remove the trend at prediction time for lagged values
    for h in np.arange(0,horizon+seq_length+1):
        if h>horizon:
            temp = pd.DataFrame()
            temp[f"value - {h}"]=(tdf[field].shift(h)-tdf["trend_norm"]).copy()
            tdf=pd.concat([tdf,temp],axis=1)

    # We remove the trend at prediction time 
    # It must be done AFTER creating the lagged values
    tdf[field]=tdf[field]-tdf["trend_norm"]
    tdf=tdf.dropna()

    cols = tdf.columns[tdf.columns.str.contains(f"value -")]
    X = tdf[cols].copy()
    y = tdf[[field]].copy()
    y.columns=["values"]

    TRAIN_SPLIT = tdf[tdf["DATE"]>"2018"].index.values[0]

    return X[X.index<=TRAIN_SPLIT].copy(),y[X.index<=TRAIN_SPLIT].copy()
    
    
def process_test_station(pair):
    folder,station,seasonal_removal,horizon,seq_length = pair
    field = 'SPA.NO2'
    # Reading input file
    df = pd.read_csv(f"{folder}/{station}.csv",sep=";")
    
    # Limiting Date
    df = df[df["DATE"]<"2020-01-01"]
    
    print(f"Test: {station}: {field}")

    tdf = df[["DATE",field]].copy()

    #Stop is there are null values
    if (tdf[field].isnull().sum()>0):
        return

    # Log transform
    tdf[field]=np.log1p(tdf[field])

    if seasonal_removal:
        #STL Decomp
        stl = seasonal_decompose(tdf[field], model="additive",period=24)
        tdf[field]=tdf[field]-stl.seasonal
        # We keep the trend at prediction time:
        # norm_field = field-trend_pred (horizon shifted)
        tdf["trend_norm"]=stl.trend.shift(horizon)
        tdf["seasonal"]=stl.seasonal
    else:
        tdf["trend_norm"]=0
        tdf["seasonal"]=0

    # We remove the trend at prediction time for lagged values
    for h in np.arange(0,horizon+seq_length+1):
        if h>horizon:
            temp = pd.DataFrame()
            temp[f"value - {h}"]=(tdf[field].shift(h)-tdf["trend_norm"]).copy()
            tdf=pd.concat([tdf,temp],axis=1)

    # We remove the trend at prediction time 
    # It must be done AFTER creating the lagged values
    tdf[field]=tdf[field]-tdf["trend_norm"]
    tdf=tdf.dropna()

    cols = tdf.columns[tdf.columns.str.contains(f"value -")]
    X = tdf[cols].copy()
    y = tdf[[field]].copy()
    y.columns=["values"]

    test_SPLIT = tdf[tdf["DATE"]>"2018"].index.values[0]

    return [station,X[X.index>test_SPLIT].copy()],\
            [station,y[X.index>test_SPLIT].copy()],\
            [station,tdf[X.index>test_SPLIT][["trend_norm","seasonal"]].copy()]

def get_train_data(folder,cluster1,seasonal_removal,seq_length,horizon):
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(process_station, [[folder,station_field,seasonal_removal,horizon,seq_length] for station_field in cluster1])
    
    X_train = [res[0] for res in result]
    y_train = [res[1] for res in result]

    X_train=pd.concat(X_train)
    y_train=pd.concat(y_train)

    return X_train,y_train

def get_test_data(folder,stations,seasonal_removal,seq_length,horizon):  
   
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(process_test_station, [[folder,station,seasonal_removal,horizon,seq_length] for station in stations])

    X_test = dict([res[0] for res in result])
    y_test = dict([res[1] for res in result])
    v_recover = dict([res[2] for res in result])
        
    return X_test, y_test, v_recover