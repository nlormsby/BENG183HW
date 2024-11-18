import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage

# data1 = 10*np.random.random_sample((30,2)) #Random sample of data for this example only

data1 = np.loadtxt('dataset1.txt', skiprows=1)

data2 = np.loadtxt('dataset2.txt', skiprows=1)

fig, axes = plt.subplots(2,4,figsize=(16,4)) #making a plot with 2 subplots
fig.tight_layout()

#linkage function and fccluster are scipy.cluster.hierarchy functions
links1 = linkage(data1, method="single")
axes[0,0].scatter(data1[:,0], data1[:,1], c=fcluster(links1,t=2,criterion="maxclust")) #plotting the scatterplot and coloring the points with the cluster data
axes[0,0].set_title("Single")

links2 = linkage(data1, method="average")
axes[0,1].scatter(data1[:,0], data1[:,1], c=fcluster(links2,t=2,criterion="maxclust")) #plotting the scatterplot and coloring the points with the cluster data
axes[0,1].set_title("Average")

links3 = linkage(data1, method="complete")
axes[0,2].scatter(data1[:,0], data1[:,1], c=fcluster(links3,t=2,criterion="maxclust")) #plotting the scatterplot and coloring the points with the cluster data
axes[0,2].set_title("Complete")

# kmeans is a scipy.cluster function
kmeans = KMeans(n_clusters=2,n_init=100,random_state=0).fit(data1)
axes[0,3].scatter(data1[:,0], data1[:,1], c=kmeans.predict(data1)) #plotting the scatterplot and coloring the points with the cluster data
axes[0,3].set_title("Kmeans")



links1 = linkage(data2, method="single")
axes[1,0].scatter(data2[:,0], data2[:,1], c=fcluster(links1,t=2,criterion="maxclust")) #plotting the scatterplot and coloring the points with the cluster data
axes[1,0].set_title("Single")

links2 = linkage(data2, method="average")
axes[1,1].scatter(data2[:,0], data2[:,1], c=fcluster(links2,t=2,criterion="maxclust")) #plotting the scatterplot and coloring the points with the cluster data
axes[1,1].set_title("Average")

links3 = linkage(data2, method="complete")
axes[1,2].scatter(data2[:,0], data2[:,1], c=fcluster(links3,t=2,criterion="maxclust")) #plotting the scatterplot and coloring the points with the cluster data
axes[1,2].set_title("Complete")

# kmeans is a scipy.cluster function
kmeans = KMeans(n_clusters=2,n_init=100,random_state=0).fit(data2)
axes[1,3].scatter(data2[:,0], data2[:,1], c=kmeans.predict(data2)) #plotting the scatterplot and coloring the points with the cluster data
axes[1,3].set_title("Kmeans")

plt.show()