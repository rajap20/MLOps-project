## Developed By Group 8-MLOps
## Used Credit Prediction Model

import pandas as pd
import numpy as np
import joblib
import warnings
from io import BytesIO
import requests
from usedcar import CarPredictionModel as CarPredictionModel
warnings.filterwarnings('ignore')

model_path = 'https://github.com/rajap20/MLOpsProject/blob/main/creditmodel.pkl?raw=true'
        
class CreditPredictor():              
        
    def __init__(self):
        model_file = BytesIO(requests.get(model_path).content)
        self.model = joblib.load(model_file)
        
    def predict(self,
                V3 = 'ACH',
                V4 = 3075,
                V5 = 530041,
                V6 = 'AP',
                V7 = 'DEALER',
                V8 = 'SC',
                v9 = 0.470588235,
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
            km_driven (float): Kilometer driven by the car (odometer reading) in 1000 kms. E.g. 5.5 indicates 5500 km. driven.
            fuel_type (str): 'Petrol' or 'Diesel': Default is 'Petrol'
            age: (int): Number of years since car is bought.
            transmission (str): 'Manual' or 'Automatic': Default is 'Manual'
            owner (str): 'First' or 'Second' or 'Third'. Default is 'First'
            seats (int): Number of seats. Default is 4.
            make (str): Currently it supports only 'maruti' or 'hyundai'. Default is 'maruti'.
            model (str): 'alto' or 'swift' or 'desire' or 'zen'. Default is 'swift'
            mileage (float): Mileage of the car in km per liter. Default is 10.0
            engine (float): Engine capacity of the car in cc. Default is 800.0 cc.
            power (float): Power of the car in bhp. Default is 85.0 bhp. 
            location (str): Which location the car is available for the sell. 'Bangalore' or 'Hyderabad' or 'Mumbai' or 'Chennai'. Default is 'Bangalore'
            
          
        Returns:
            float: The expected sale price of the car in INR Lakhs. For example, 8.5 means the car is expected to be sold at INR 8.5 lakhs.
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
   from MLOpsProject import CreditPredictionModel as CreditPredictionModel
