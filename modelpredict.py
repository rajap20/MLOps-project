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

model_path = 'https://github.com/rajap20/MLOps-project/blob/main/creditmodel.pkl?raw=true'
        
class CreditPredictor():              
        
    def __init__(self):
        model_file = BytesIO(requests.get(model_path).content)
        self.model = joblib.load(model_file)
        
    def predict(self,
                V3 = 
                V4 = 
                V5 = 
                V6
                V7
                V8
                V10
                V11
                V12
                V13
                V14
                V15
                V16
                V17
                V18
                V19
                V20
                V21
                V22
 
                km_driven = 1.0, 
                fuel_type = 'Petrol',
                age = 1,
                transmission = 'Manual',
                owner = 'First',
                seats = 4,
                make = 'maruti',
                model = 'swift',
                mileage = 10.0,
                engine = 800.0,
                power = 85.0,
                location = 'Bangalore'):

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
        
        car_data = {}
        
        car_data['KM_Driven'] = km_driven
        car_data['Fuel_Type'] = fuel_type
        car_data['age'] = age
        car_data['Transmission'] = transmission
        car_data['Owner_Type'] = owner
        car_data['Seats'] = seats
        car_data['make'] = make
        car_data['model'] = model        
        car_data['mileage_new'] = mileage
        car_data['engine_new'] = engine
        car_data['model'] = model
        car_data['power_new'] = power
        car_data['Location'] = location 
                
        df = pd.DataFrame(car_data, index = [0])
        
        return np.round(self.model.pipeline.predict(df)[0], 2)

if __name__ == "__main__":
   from usedcar import CarPredictionModel as CarPredictionModel 
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
