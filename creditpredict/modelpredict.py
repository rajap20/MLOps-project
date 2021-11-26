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

model_path = 'https://github.com/rajap20/MLOpsProject/blob/main/creditmodel.pkl?raw=true'
        
class CreditPredictor():              
        
    def __init__(self):
        model_file = BytesIO(requests.get(model_path).content)
        self.model = joblib.load(model_file)
        
    def predict(self,
                V3 = 'ACH',
                V4 = '3075.0',
                V5 = '530041',
                V6 = 'AP',
                V7 = 'DEALER',
                V8 = 'SC',
                V9 = 0.470588235,
                V10 = -0.793281608,
                V11 = 0.124252057,
                V12 = -0.093757816,
                V13 = 87000,
                V14 = 71000,
                V15 = 'M',
                V16 = 'OTHERS',
                V17 = 'SAL',
                V18 = 'O',
                V19 = 0.306122449,
                V20 = -0.906781205,
                V21 = 0.158169082,
                V22 = -1.321152847):
        
        """
        Predicts the price of an used car given it's attributes.
  
        Parameters:
         V3 (str): "Payment Type" ("ACH", "ADM", "DRE", "DAS", "ENCH", "PDC")
         V4 (str): "Area Code"
         V5 (str): "Pin code"
        V6 (str): "State" ("AP", "AS", "BR", "CG", "DL", "UP", "HA", "GJ", "HP", "CH", "JH",
               "JK", "KA", "KL", "MH", "MP", "OR", "TN", "PY", "PB", "RJ", "WB",
               "UC", "TR", "MN")
        V7 (str) : "Dealer Type", ("DEALER", "ASC"), default="DEALER"
        V8 (str) : ["SC", "MO", "MC", "EB"], default="SC", label = "Product Code (SC: Scooter, MO: Moped, MC: Motorcycle, EB: Electric Bike)"
        V9 (float) : (default=0.470588	, label="Tenure")
        V10 (float) : (default=-0.793282, label="Rate of Interest")
        V11 (float) : (default=0.124252, label="EMI")
        V12 (float) :(default=-0.093758, label="Processing Fee")
        V13 (float) : (default=87000.0, label="Asset cost")
        V14 (float) : (default=71000.0, label="Loan Amount")
        V15 (str) : (["M", "F"], default="M", label = "Gender")
        V16 (str) : (["OTHERS", "PG", "SSC", "HSC", "UG"], default="OTHERS", label = "Qualification")
        V17 (str) : (["SAL", "SEP", "AGR", "STU", "OTH", "PEN", "HOW"], default="SAL", label = "Employment type (SAL: Salaried, AGR: Agriculture, HOW: Housewife, OTH: Others, STU: Student, SEP: Self-employed, PEN: Pension)")
        V18 (str) : (["O", "R", "L"],  default="O", label = "Residence Type (O: Owned, R: Rented, L: Leased)")
        V19 (float) : (default=0.306122, label="Age")
        V20 (float) : (default=-0.906781, label="CIBIL Score")
        V21 (float) : (default=	0.158169, label="Net Salary")
        V22 (float) : (default=-1.321153, label="Net Internal Rate of return")
                   
        Returns:
            float: The output shows the probability of default of the candidate
            """
        
        credit_data = {}
        
        credit_data['V3'] = V3
        credit_data['V4'] = V4
        credit_data['V5'] = V5
        credit_data['V6'] = V6
        credit_data['V7'] = V7
        credit_data['V8'] = V8
        credit_data['V9'] = V9
        credit_data['V10'] = V10        
        credit_data['V11'] = V11
        credit_data['V12'] = V12
        credit_data['V13'] = V13
        credit_data['V14'] = V14
        credit_data['V15'] = V15
        credit_data['V16'] = V16
        credit_data['V17'] = V17
        credit_data['V18'] = V18
        credit_data['V19'] = V19
        credit_data['V20'] = V20
        credit_data['V21'] = V21
        credit_data['V22'] = V22 
                
        df = pd.DataFrame(credit_data, index = [0])
        
        return np.round(self.model.pipeline.predict(df)[0], 2)

if __name__ == "__main__":
   from creditpredict import CreditPredictionModel as CreditPredictionModel
