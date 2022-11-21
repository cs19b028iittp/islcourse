from sklearn.datasets import make_blobs, make_circles, load_digits
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score


###### PART 1 ######

def get_data_blobs(n_points=100):
  X, y =  make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  return X,y

def get_data_circles(n_points=100):
  X, y = make_circles(n_samples=n_points, shuffle=True,  factor=0.8, random_state=0)
  return X,y

def get_data_mnist():
  digits = load_digits()
  X=digits.data
  y=digits.target
  return X,y

def build_kmeans(X=None,k=10):
  km = KMeans(n_clusters=k, random_state=0)
  km.fit(X)
  return km

def assign_kmeans(km=None,X=None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  h=homogeneity_score(ypred_1,ypred_2)
  c=completeness_score(ypred_1,ypred_2)
  v=v_measure_score(ypred_1,ypred_2)
  return h,c,v


###### PART 2 ######

def build_lr_model(X=None, y=None):
  lr_model = LogisticRegression(random_state=0)
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  rf_model = RandomForestClassifier(max_depth=5, random_state=0)
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  acc, prec, rec, f1, auc = 0,0,0,0,0
  y_pred = model1.predict(X)
  acc = accuracy_score(y,y_pred)
  prec=precision_score(y, ypred,average='macro')
  rec=recall_score(y, ypred,average='macro')
  f1=f1_score(y, ypred,average='macro')
  auc = roc_auc_score(y, model1.predict_proba(X), average='macro', multi_class='ovr')
  return acc, prec, rec, f1, auc
