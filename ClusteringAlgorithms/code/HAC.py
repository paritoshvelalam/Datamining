import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.spatial.distance as sd
import sys
from pandas import Series
import sys
from sklearn.decomposition import PCA

filepath = input("Enter file name: ")

num_clusters = int(input("Enter number of clusters: "))

# filepath = "D:\\Buffalocourses\\secondsem\\dataminingandbioinformatics\\project2\\iyer.txt"

filedata = np.loadtxt(filepath, delimiter="\t")
data = filedata[:,2:]
truelabels = filedata[:,1]
truelabels = np.reshape(truelabels, (len(truelabels),1))
clusterID = np.zeros((data.shape[0], 1))


# Using Union Disjoint DS

parent = [-1]*len(data)

def find(x, parent):
    if (parent[x] != -1):
        parent[x] = find(parent[x], parent)
        return parent[x]
    else:
        return x
        

def union(x, y, parent):
    p1 = find(x, parent)
    p2 = find(y, parent)
    print("{}->{}".format(p2+1,p1+1))
    parent[p2] = p1
    return

dist_mat = np.zeros((data.shape[0],data.shape[0]))
# Calculating the actual distance matrix using Euclidean Distance as the metric
dist_mat = sd.cdist(data, data, metric = 'euclid')

num_iter = data.shape[0] - num_clusters
# num_iter = 1
for i in range(num_iter):
    min_dist = sys.maxsize
    min_x = 0
    min_y = 0
    for j in range(1, data.shape[0]):
        for k in range(j):
            parent_j = find(j, parent)
            parent_k = find(k, parent)
            if(parent_j != parent_k):
                if (dist_mat[parent_j][parent_k] < min_dist):
                    min_dist = dist_mat[parent_j][parent_k]
                    min_x = parent_j
                    min_y = parent_k
    
    #point min_y to min_x
    union(min_x, min_y, parent)
    
    #update distancematrix
    
    for j in range(0, data.shape[0]):
        parentcurrent = find(j, parent)
        if((parentcurrent != min_x)):
            dist_mat[parentcurrent][min_x] = min(dist_mat[parentcurrent][min_x], dist_mat[parentcurrent][min_y])
            dist_mat[min_x][parentcurrent] = dist_mat[parentcurrent][min_x]


ClusID_dict = {}
j = 0
for i in range(len(parent)):
    if (parent[i] == -1):
        ClusID_dict[i] = j
        j += 1

for i in range(data.shape[0]):
    clusterID[i][0] =  ClusID_dict[find(i, parent)]
    

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
plt.title("HAC plot")
#plt.text(0.5,0.05,"Jaccard Coefficient: {}".format(JaccardIndex) +  "         Rand Index: {}".format(randIndex), ha = 'center')
plt.show()
