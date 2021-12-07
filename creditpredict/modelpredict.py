## Developed By Group 8-MLOps
## Used Credit Prediction Model

import pandas as pd
import numpy as np
import joblib
import warnings
from io import BytesIO
import requests
from creditpredict import CreditPredictionModel as CreditPredictionModel
warnings.filterwarnings('ignore')

model_path = 'https://github.com/rajap20/MLOpsProject/blob/main/creditpredict/creditmodel.pkl?raw=true'
        
class CreditPredictor():              
        
    def __init__(self):
        model_file = BytesIO(requests.get(model_path).content)
        self.model = joblib.load(model_file)
        
        
    def predict(self,
                 area_code  =  "3075.0",
                 asset_cost  =  87000.0,
                 state  =  "AP",
                 loan_amt  =  71000.0,
                 resid_type  =  "O",
                 emi  =  0.124252,
                 net_irr  =  1.321153,
                 net_salary  =  0.158169,
                 proc_fee  =  0.093758,
                 roi  =  0.793282,
                 age  =  0.306122,
                 tenure  =  0.470588):

        """
        Predicts the price of an used car given it's attributes.
  
        Parameters:
        area_code (str): "Area Code"
        state (str): "State" ("AP", "AS", "BR", "CG", "DL", "UP", "HA", "GJ", "HP", "CH", "JH",
               "JK", "KA", "KL", "MH", "MP", "OR", "TN", "PY", "PB", "RJ", "WB",
               "UC", "TR", "MN")
        tenure (float) : (default=0.470588	, label="Tenure")
        roi (float) : (default=-0.793282, label="Rate of Interest")
        emi (float) : (default=0.124252, label="EMI")
        proc_fee (float) :(default=-0.093758, label="Processing Fee")
        asset_cost (float) : (default=87000.0, label="Asset cost")
        loan_amt (float) : (default=71000.0, label="Loan Amount")
        resid_type (str) : (["O", "R", "L"],  default="O", label = "Residence Type (O: Owned, R: Rented, L: Leased)")
        age (float) : (default=0.306122, label="Age")
        net_salary (float) : (default=	0.158169, label="Net Salary")
        net_irr (float) : (default=-1.321153, label="Net Internal Rate of return")
                   
        Returns:
            float: The output shows the probability of default of the candidate
            """
        
        credit_data = {}
        
        credit_data['area_code'] = area_code
        credit_data['asset_cost'] = asset_cost
        credit_data['state'] =  state
        credit_data['loan_amt'] =  loan_amt
        credit_data['resid_type'] =  resid_type
        credit_data['emi'] =  emi
        credit_data['net_irr'] =  net_irr
        credit_data['net_salary'] =  net_salary
        credit_data['proc_fee'] =  proc_fee
        credit_data['roi'] =  roi
        credit_data['age'] =  age
        credit_data['tenure'] =  tenure

                
        df = pd.DataFrame(credit_data, index = [0])
        
        return np.round(self.model.pipeline.predict(df)[0], 2)

if __name__ == "__main__":
   from creditpredict import CreditPredictionModel as CreditPredictionModel
