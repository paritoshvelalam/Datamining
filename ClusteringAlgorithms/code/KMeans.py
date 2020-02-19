import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA

#np.random.seed(1234)
filepath=input("Enter file name: ")
k = int(input("Enter number of clusters: "))
init = eval(input("Enter gene id of cluster Centroids: "))
maxiter = int(input("Enter Maximum Iterations: "))
# filepath = "D:\\Buffalocourses\\secondsem\\dataminingandbioinformatics\\project2\\cho.txt"
# k = 5
# init = []
# maxiter = 100

filedata = np.loadtxt(filepath, delimiter="\t")
data = filedata[:,2:]
truelabels = filedata[:,1]
truelabels = np.reshape(truelabels, (len(truelabels),1))


## Initialize Centroids
if(len(init)==k):
    init=[i-1 for i in init]
    centroids= data[init,:]
else:
    print("Using random centroids.")
    perm=np.random.permutation(len(data))
    centroids= data[perm[0:k]]
    

##Run K-Means algorithm
iterations = 0
oldCentroids = np.zeros(shape = centroids.shape)
datawithclusterID = np.concatenate((np.zeros(shape = (data.shape[0],1)), data), axis = 1)
clusterID = np.zeros(shape = (data.shape[0], 1))
while((np.linalg.norm(oldCentroids-centroids)!=0) and (iterations<maxiter)):
    oldCentroids = np.copy(centroids)
    for i in range(len(datawithclusterID)):
        dist = np.linalg.norm(datawithclusterID[i,1:] - centroids, ord=2, axis=1)
        datawithclusterID[i][0] = np.argmin(dist)
        clusterID[i][0] = datawithclusterID[i][0]
    for i in np.unique(datawithclusterID[:,0]):
        centroids[int(i)] = datawithclusterID[datawithclusterID[:,0] == int(i)].mean(0)[1:]
    iterations += 1
    
    
## metrics
groundTruth = (truelabels.T == truelabels).astype(int)
clustering = (clusterID.T == clusterID).astype(int)
table = (groundTruth == clustering)
randIndex = table.sum()/table.size
table1 =  groundTruth & clustering
table2 = groundTruth | clustering
JaccardIndex = table1.sum()/table2.sum()


print("Jaccard Coefficient: {}".format(JaccardIndex))
print("Rand Index: {}".format(randIndex))

labels = []
for i in range(len(clusterID)):
    labels.append(clusterID[i][0]+1)

if(data.shape[1]>2):
    pca = PCA(n_components=2)
    dataforpca = np.matrix(data)
    data_pca = pca.fit_transform(dataforpca)

    title =  "plot_title"
    plt.figure(figsize=(8,6))
    px=data_pca[:,0]
    py=data_pca[:,1]
else:
    px = data[:,0]
    py = data[:,1]
unique = list(set(labels))
colors = [plt.cm.jet(float(i)/(max(unique))) for i in unique]
for i, u in enumerate(unique):
    xi = [px[j] for j  in range(len(px)) if labels[j] == u]
    yi = [py[j] for j  in range(len(px)) if labels[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(int(u)))
plt.legend()
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("K-Means plot")
#plt.text(0.5,0.05,"Jaccard Coefficient: {}".format(JaccardIndex) +  "         Rand Index: {}".format(randIndex), ha = 'center')
plt.show()
