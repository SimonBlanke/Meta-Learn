'''
MIT License
Copyright (c) [2018] [Simon Franz Albert Blanke]
Email: simonblanke528481@gmail.com
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import time
import datetime
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.externals import joblib

from .label_encoder_dict import label_encoder_dict


class MetaRegressor(object):

  def __init__(self):
    self.path = './meta_learn/data/sklearn.neighbors.KNeighborsClassifier'
    self.meta_regressor = None

    self.model_name = self._get_model_name()


  def train_meta_regressor(self):
    X_train, y_train = self._get_meta_knowledge()

    X_train = self._label_enconding(X_train)
    #print(X_train)
    self._train_regressor(X_train, y_train)
    self._store_model()


  def _get_model_name(self):
    model_name = self.path.split('/')[-1]
    return model_name


  def _get_hyperpara(self):
    return label_encoder_dict[self.model_name]


  def _label_enconding(self, X_train):
    hyperpara_dict = self._get_hyperpara()

    for hyperpara_key in hyperpara_dict:
      X_train = X_train.replace({str(hyperpara_key): hyperpara_dict[hyperpara_key]})

    return X_train


  def _get_meta_knowledge(self):
    data = pd.read_csv(self.path)
    
    column_names = data.columns
    score_name = [name for name in column_names if 'mean_test_score' in name]
    
    X_train = data.drop(score_name, axis=1)
    y_train = data[score_name]

    return X_train, y_train


  def _train_regressor(self, X_train, y_train):
    if self.meta_regressor == None:
      self.meta_regressor = GradientBoostingRegressor(n_estimators=10)
      self.meta_regressor.fit(X_train, y_train)
    

  def _store_model(self):
    filename = './meta_learn/data/'+str(self.model_name)
    joblib.dump(self.meta_regressor, filename)








