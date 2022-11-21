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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


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
  lr_param_grid = {'penalty':['l1','l2'],
      "solver" : ["liblinear"]}
  return lr_param_grid

def get_paramgrid_rf():
  rf_param_grid = {"n_estimators": [1, 10, 100], "criterion": ["gini", "entropy"], "max_depth": [1, 10, None]}
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
 # grid_search_cv = GridSearchCV(model1, param_grid, cv=cv, scoring=metrics, refit=False)
 # grid_search_cv.fit(X, y)

  #cv_results = cross_validate(model1, X, y, cv=cv, scoring=metrics)
  
  top1_scores = []#[0.0 for i in range(len(metrics))]
  
  for scoring in metrics:
    grid_search_cv = GridSearchCV(model,param_grid, cv=cv, scoring=scoring)
    grid_search_cv.fit(X,y)
    top1_scores.append(grid_search_cv.best_score_)

  return top1_scores
  #for i in range(len(metrics)):
   # try:
    #  top1_scores[i]=grid_search_cv.scorer_[metrics[i]](grid_search_cv,X,y)
     # print('******##$$%%%')
    #except:
     # pass
  #return top1_scores

  
  ###### PART 3 ######

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    
    self.fc_encoder = nn.Linear(inp_dim,hid_dim) # write your code inp_dim to hid_dim mapper
    self.fc_decoder = nn.Linear(hid_dim,inp_dim) # write your code hid_dim to inp_dim mapper
    self.fc_classifier = nn.Linear(hid_dim,num_classes) # write your code to map hid_dim to num_classes
    
    self.relu = nn.ReLU #write your code - relu object
    self.softmax = nn.Softmax #write your code - softmax object
    
  def forward(self,x):
    x = torch.flatten(x) # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu()(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax()(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    yground = F.one_hot(yground,num_classes=y_pred.shape[0])
    lc1 = -torch.mean(torch.mul(yground,torch.log(y_pred+0.001))) # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  npX,npy = get_data_mnist()
  X, y = torch.from_numpy(npX),torch.from_numpy(npy)
  # write your code
  return X,y

def get_loss_on_single_point(mynn,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  print(lossval)
  # the lossval should have grad_fn attribute set
  return lossval

def train_combined_encdec_predictor(mynn,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    optimizer.zero_grad()
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)
    lval.backward()
    optimizer.step()
    
  return mynn

      
 
