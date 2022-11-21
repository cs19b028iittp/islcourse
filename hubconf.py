# kali
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
# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2, random_state=0)
  return X,y

def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_samples=n_points, noise=0.1, factor=0.2, random_state=1)
  # write your code ...
  return X,y

def get_data_mnist():
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = load_digits(return_X_y=True)
  return X,y

def build_kmeans(X=None,k=10):
  pass
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  # this is the KMeans object
  km = KMeans(n_clusters=k, random_state=0,init='k-means++')
  km = km.fit(X)
  return km

def assign_kmeans(km=None,X=None):
  pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values
  h = sklearn.metrics.homogeneity_score(ypred_1, ypred_2)
  c = sklearn.metrics.completeness_score(ypred_1, ypred_2)
  v = sklearn.metrics.v_measure_score(ypred_1, ypred_2)
  return h,c,v

###### PART 2 ######

def build_lr_model(X=None, y=None):
  pass
  lr_model = LogisticRegression(random_state=0,solver='liblinear').fit(X, y)
  # write your code...
  # Build logistic regression, refer to sklearn
  return lr_model

def build_rf_model(X=None, y=None):
  pass
  rf_model = RandomForestClassifier(max_depth=2, random_state=0).fit(X,y)
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  pass
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  y_pred = model1.predict(X)
  y_pred_prob = model1.predict_proba(X)
  acc=accuracy_score(y,y_pred)
  prec=precision_score(y,y_pred,average='macro')
  rec=recall_score(y,y_pred,average='macro')
  f1=f1_score(y,y_pred,average='macro')
  auc=roc_auc_score(y,y_pred_prob,average='macro',multi_class='ovr')
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = {
        'max_depth': [1, 10, None],
        'n_estimators': [1, 10, 100],
        'criterion': ['gini', 'entropy']
    }
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
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
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
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
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
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
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
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


