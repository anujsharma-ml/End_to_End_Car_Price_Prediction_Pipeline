import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Iqrclipper(BaseEstimator,TransformerMixin):
   def __init__(self,factor=1.5):
    self.factor = factor
    self.lower_bound_ = None
    self.upper_bound_ = None

   def fit(self,X,y=None):
      X_df = pd.DataFrame(X)
      q1 = X_df.quantile(0.25)
      q3 = X_df.quantile(0.75)
      iqr = q3-q1

      self.lower_bound_ = q1 - self.factor * iqr
      self.upper_bound_ = q3 + self.factor * iqr
      return self

   def transform(self,X):
      X_df = pd.DataFrame(X).copy()
      return X_df.clip(lower=self.lower_bound_, upper=self.upper_bound_, axis=1)