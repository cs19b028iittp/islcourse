import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

device = "cuda" if torch.cuda.is_available() else "cpu"

class cs19b028(nn.Module):
  def __init__(self):
        super(cs19b028, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
  def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits
    
def get_model(train_data_loader, n_epochs):
  model = cs19b028().to(device)
  
  return model

def get_model_advanced(train_data_loader, n_epochs, lr, config):
  model = get_model();
  
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
  size = len(train_data_loader.dataset)
  model.train()
  for batch, (X, y) in enumerate(train_data_loader):
      X, y = X.to(device), y.to(device)

      # Compute prediction error
      pred = model(X)
      loss = loss_fn(pred, y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
          loss, current = loss.item(), batch * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
          
      return model
    
