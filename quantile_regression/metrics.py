import numpy as np
import pandas as pd
from tqdm import tqdm

## Medida del CRPS
def heavyside(prediction,actual):
    return prediction >= actual

def cdf_dif(prediction,actual):
    quantiles = np.arange(1,100)/100.0
    t=pd.Series(prediction)
    dif=t-t.shift(1)
    dif=dif.dropna()
    fs = sum(dif*((quantiles-heavyside(prediction,actual))[1:]**2))
    # If the actual is outside the range of the prediction, 
    # we need to account for that areas outside the range 
    if actual > prediction[-1]:
        fs += (actual-prediction[-1]) * 1
    if actual < prediction[0]:
        fs += (prediction[0]-actual) * 1
    return fs

def CRPS(predictions, actuals):
    difs_mean = [cdf_dif(predictions[i],actuals[i]) for i in range(len(actuals))]
    return np.mean(difs_mean)

def evaluate(predictions,target):

    res={}
    
    # Calculate the CRPS
    res["crps"]=CRPS(predictions,target)
    
    ## Calculate as well measures for the quantile 50
    total_df = pd.DataFrame(predictions)
    quantiles = np.arange(1,100)/100.0 
    total_df.columns=np.array(quantiles).astype(str)
    #RMSE       
    res["rmse"]=np.sqrt(np.mean((target-total_df["0.5"])**2))
    #MAE    
    res["mae"]=np.mean(np.abs(target-total_df["0.5"] ) )
    #Bias 
    res["bias"]=np.mean(target-total_df["0.5"])
    #Corr
    res["corr"]=np.corrcoef(target,total_df["0.5"])[0][1]
    
    return res