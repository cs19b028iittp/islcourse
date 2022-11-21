from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  return X,y

def get_data_circles(n_points=100):
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_samples=n_points, random_state=0, factor=0.7)
  return X,y

def get_data_mnist():
  pass
  # write your code here
  # Refer to sklearn data sets
  X,y = None
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  km = KMeans(n_clusters=k, random_state=0).fit(X)
  return km

def assign_kmeans(km=None,X=None):
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values
  h,c,v = homogeneity_completeness_v_measure(ypred_1, ypred_2, beta=1.0)
  return h,c,v


