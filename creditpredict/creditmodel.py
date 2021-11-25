
import pandas as pd
import numpy as np
import joblib
import warnings
from io import BytesIO
import requests
warnings.filterwarnings('ignore')

class CreditPredictionModel():

    def __init__(self, pipeline, all_features, cat_features, num_features, accuracy):
      self.pipeline = pipeline
      self.all_features = all_features
      self.cat_features = cat_features
      self.num_features = num_features
      self.rmse = accuracy
