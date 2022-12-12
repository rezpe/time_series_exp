import lightgbm as lgb
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from scipy import stats

class TotalLGBQuantile():
    
    def __init__(self,n_estimators,max_depth):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.quantiles=[0.022750131948179195,0.15865525393145707,0.5,0.8413447460685429,0.9772498680518208]
        self.estimators = []
        
    def __str__(self):
        return f"{self.n_estimators}_{self.max_depth}"
        
    def fit(self,X_train,y_train):
        print("Distributed training !")
        for q in tqdm(self.quantiles):
            print(f"Quantile: {q}")
            reg = lgb.LGBMRegressor(n_estimators=self.n_estimators,
                                    objective= 'quantile',
                                    loss="quantile",
                                    alpha=q,
                                    random_state=2020,
                                   max_depth=self.max_depth,
                                   n_jobs=30)
                                
            reg.fit(X_train, y_train)
            self.estimators.append(reg)
        print("Done")
        
    def predict(self,X):
        predictions_gbr = {}
        print("predicting")
        for i,reg in tqdm(enumerate(self.estimators)):
            predictions_gbr[i]=reg.predict(X)
            
        total_df=pd.DataFrame(predictions_gbr)
        
        pred = pd.DataFrame()

        # Mean
        pred["mean"]=total_df[2]

        # Std Variation
        temp = (total_df-total_df[2])
        temp[0]=np.abs(temp[0]/2)
        temp[1]=np.abs(temp[1])
        temp[3]=np.abs(temp[2])
        temp[4]=np.abs(temp[4]/2)
        pred["std"]=np.std(temp[[0,1,3,4]],axis=1)

        mi_norm = stats.norm(pred["mean"],pred["std"])
        
        final_pred = pd.DataFrame()
        for quantile in np.arange(1,100,1):
            final_pred[quantile]=mi_norm.ppf(quantile/100)
        
        return final_pred