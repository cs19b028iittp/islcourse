import torch
import numpy as np
from torch import nn
import torch.optim as optim
import sklearn
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def get_data_blobs(n_points=100):
  pass
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2, random_state=0)
  return X,y

def get_data_circles(n_points=100):
  pass
  X, y = make_circles(n_samples=n_points, noise=0.1, factor=0.2, random_state=1)
  return X,y

def get_data_mnist():
  pass
  X, y = load_digits(return_X_y=True)
  return X,y

def build_kmeans(X=None,k=10):
  pass
  km = KMeans(n_clusters=k, random_state=0,init='k-means++')
  km = km.fit(X)
  return km

def assign_kmeans(km=None,X=None):
  pass
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  h,c,v = 0,0,0 
  h = sklearn.metrics.homogeneity_score(ypred_1, ypred_2)
  c = sklearn.metrics.completeness_score(ypred_1, ypred_2)
  v = sklearn.metrics.v_measure_score(ypred_1, ypred_2)
  return h,c,v

###### PART 2 ######

def build_lr_model(X=None, y=None):
  lr_model = LogisticRegression(random_state=0, solver='liblinear', fit_intercept=False)
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  rf_model = RandomForestClassifier(random_state=400)
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  acc, prec, rec, f1, auc = 0,0,0,0,0
  y_pred = model1.predict(X)
  acc = accuracy_score(y,y_pred)
  prec=precision_score(y, y_pred,average='macro')
  rec=recall_score(y, y_pred,average='macro')
  f1=f1_score(y, y_pred,average='macro')
  auc = roc_auc_score(y, model1.predict_proba(X), average='macro', multi_class='ovr')
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  lr_param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
  return lr_param_grid

def get_paramgrid_rf():
  rf_param_grid = {
        'max_depth': [1, 10, None],
        'n_estimators': [1, 10, 100],
        'criterion': ['gini', 'entropy']
    }
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  
  grid_search_cv = GridSearchCV(model1, param_grid, cv=cv)
  grid_search_cv.fit(X, y)
  params = grid_search_cv.best_params_
  acc, prec, rec, f1, auc = 0, 0, 0, 0, 0
  if 'criterion' in params.keys():
    rfc1 = RandomForestClassifier(random_state=42, n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'])
    rfc1.fit(X,y)
    acc, prec, rec, f1, auc = get_metrics(rfc1, X, y)
  else:
    lg1 = LogisticRegression(C=params['C'], penalty=params['penalty'],solver = "liblinear")
    lg1.fit(X, y)
    acc, prec, rec, f1, auc = get_metrics(lg1, X, y)
  
  top1_scores = []
  for k in metrics:
      if k == 'accuracy':
          top1_scores.append(acc)
      elif k == 'recall':
          top1_scores.append(rec)
      elif k == 'roc_auc':
          top1_scores.append(auc)
      elif k == 'precision':
          top1_scores.append(prec)
      else:
          top1_scores.append(f1)
        

  return top1_scores        
###### PART 3 ######

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    
    self.fc_encoder = nn.Linear(inp_dim,hid_dim) 
    self.fc_decoder = nn.Linear(hid_dim,inp_dim) 
    self.fc_classifier = nn.Linear(hid_dim,num_classes) 
    
    self.relu = nn.ReLU() #write your code - relu object
    self.softmax = nn.Softmax() #write your code - softmax object
    
  def forward(self,x):
    x = torch.flatten(x) # write your code - flatten x
    x = torch.nn.functional.normalize(x, p=2.0, dim=0)
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  def loss_fn(self,x,yground,y_pred,xencdec):
    lc1 = -(torch.nn.functional.one_hot(yground,num_classes=y_pred.shape[-1])*torch.log(y_pred)) # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    lc1=torch.mean(lc1)
    lc2 = torch.mean((x - xencdec)**2)
    return lc1+lc2
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  X,y = load_digits(return_X_y=True)
  X_tensor=torch.tensor(X)
  y_tensor=torch.tensor(y)
  return X_tensor,y_tensor

def get_loss_on_single_point(mynn,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  return lossval

def train_combined_encdec_predictor(mynn,X,y, epochs=11):
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    for j in range(X.shape[0]):
      try:
        optimizer.zero_grad()
        ypred, Xencdec = mynn(X[j])
        lval = mynn.loss_fn(X[j],y,ypred,Xencdec)
        lval.backward()
        optimizer.step()
      except:
        pass
  return mynn


