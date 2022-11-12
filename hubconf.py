import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import numpy as np
from numpy.lib.function_base import average
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class cs19b032NN(nn.Module):
  def __init__(self, config, C, H, W):
    super(cs19b032NN, self).__init__()
    self.conv2ds = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel, stride, 1) for in_channels, out_channels, kernel, stride, padding in config ])
    self.m = nn.Softmax(dim=1)
    for in_channels, out_channels, kernel, stride, padding in config:
      H = int((H + 2 - kernel[0]) / stride) + 1
      W = int((W + 2 - kernel[1]) / stride) + 1
      C = out_channels
    self.fc = nn.Linear(H*W*C, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    for conv2d in self.conv2ds:
     x = conv2d(x)
    x = self.relu(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    output = self.m(x)
    return output


def loss_fn(y_pred, y_ground):
  v = -(y_ground * torch.log(y_pred + 0.0001))
  v = torch.sum(v)
  return v


# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  for X,y in train_data_loader:
    N, C, H, W =  X.shape
    break;
  config = [[C, C+2, (4,4), 1, 'same']]
  model = cs19b032NN(config, C, H, W).to(device)
  size = len(train_data_loader.dataset)
  optimizer = torch.optim.SGD(model.parameters(), lr=10e-5)
  model.train()
  for epoch in range(1,n_epochs+1):
    print("Epoch ", epoch);
    for batch, (X, y) in enumerate(train_data_loader):
      X, y = X.to(device), y.to(device)

      # Compute prediction error
      pred = model(X)
      y_1h = F.one_hot(y, num_classes= 10)
      loss = loss_fn(pred, y_1h)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
          loss, current = loss.item(), batch * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
          
  print ('Returning model... (rollnumber: cs19b032)')
  
  return model


# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  for X,y in train_data_loader:
    N, C, H, W =  X.shape
    break;
  config = [[C, C+2, (4,4), 1, 'same']]
  model = cs19b032NN(config, C, H, W).to(device)
  size = len(train_data_loader.dataset)
  optimizer = torch.optim.SGD(model.parameters(), lr)
  model.train()
  for epoch in range(1,n_epochs+1):
    print("Epoch ", epoch);
    for batch, (X, y) in enumerate(train_data_loader):
      X, y = X.to(device), y.to(device)

      # Compute prediction error
      pred = model(X)
      y_1h = F.one_hot(y, num_classes= 10)
      loss = loss_fn(pred, y_1h)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
          loss, current = loss.item(), batch * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
          
  print ('Returning model... (rollnumber: cs19b032)')
  
  return model


# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  
  size = len(test_data_loader.dataset)
  num_batches = len(test_data_loader)
  model1.eval()

  actual = []   
  predicted = [] 
  test_loss, correct = 0, 0
  with torch.no_grad():
      for X, y in test_data_loader:
           X, y = X.to(device), y.to(device)
           y1 = model1(X)
           actual.append(y)
           predicted.append(y1.argmax(1))
           y_h = F.one_hot(y, num_classes= 10)
           test_loss += loss_fn(y1, y_h).item()
           correct += (y1.argmax(1) == y).type(torch.float).sum().item() 
        
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

  predicted = [val.item() for sublist in predicted for val in sublist]
  actual = [val.item() for sublist in actual for val in sublist]


  accuracy_val = correct
  precision_val = precision_score(actual, predicted, average='macro')
  recall_val = recall_score(actual, predicted, average='macro')
  f1score_val= f1_score(actual, predicted, average='macro')

  # print(f"Accuracy : {(accuracy_val):>0.4f}"),
  # print(f"Precision: {(precision_val):>0.4f}")
  # print(f"Recall   : {(recall_val):>0.4f}")
  # print(f"F1 scores: {(f1score_val):>0.4f}")
  
  return accuracy_val, precision_val, recall_val, f1score_val
