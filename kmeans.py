  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#X = np.array([[1,1],
#     [2,1],
#     [1,2],
#     [4,3],
#     [5,4],])
startpts=np.array([[2, 10], 
                   [5,8],
                   [1,2]], np.float64)
X = np.array([[2,10],
     [2,5],
     [8,4],
     [5,8],
     [7,5],
     [6,4],
     [1,2],
     [4,9],])
print(X)
#plt.scatter(X[:,0],X[:,1], label='True Position')
kmeans = KMeans(n_clusters=3,init=startpts,n_init=1)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
