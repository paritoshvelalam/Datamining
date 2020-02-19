import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA

input_file_Path  = input('Enter the Filename: ')
sigma = float(input('Enter the Sigma value: '))
k = int(input('Enter number of clusters: '))
init = eval(input('Enter gene Id of cluster Centroids: '))
filedata = np.loadtxt(input_file_Path, delimiter="\t")

 
data = filedata[:,2:]

def buildSimmilarityMatrix(dataIn):
    result = np.zeros((dataIn.shape[0],dataIn.shape[0]))
    nData = dataIn.shape[0]
    for i in range(0,nData):
        for j in range(0, nData):
            weight = RbfKernel(dataIn[i], dataIn[j], sigma)
            result[i][j] = weight
    return result

def buildDegreeMatrix(similarityMatrix):
    diag = np.array(similarityMatrix.sum(axis=1)).ravel()
    result = np.diag(diag)
    return result

def unnormalizedLaplacian(simMatrix, degMatrix):
    result = degMatrix - simMatrix
    return result



def RbfKernel(data1, data2, sigma):
    delta =np.matrix(abs(np.subtract(data1, data2)))
    squaredEuclidean = (np.square(delta).sum(axis=1))
    result = np.exp(-(squaredEuclidean)/(sigma**2))
    return result





simMat = buildSimmilarityMatrix(data)
degMat = buildDegreeMatrix(simMat)
lapMat = unnormalizedLaplacian(simMat, degMat)


vals, vecs = np.linalg.eig(lapMat)

# sort these based on the eigenvalues
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]
maxi = 0;
index = 0;
for i in range(1,len(vals.tolist())):
  if(vals[i]-vals[i-1] > maxi):
      maxi = vals[i]-vals[i-1]
      index = i

maxiter = 300
data = np.array(vecs[:,0:index+1])
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


data = filedata[:,2:]


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

df_pca = pd.DataFrame(dict(x=list(px), y=list(py), labels=labels))
print("sigma ", sigma)
print("Iterations: {}".format(iterations))
print("Jaccard Index: {}".format(JaccardIndex))
print("Rand Index: {}".format(randIndex))
print(df_pca['labels'].groupby(df_pca['labels']).describe()['count'])
unique = list(set(labels))
colors = [plt.cm.jet(float(i)/(max(unique))) for i in unique]
for i, u in enumerate(unique):
    xi = [px[j] for j  in range(len(px)) if labels[j] == u]
    yi = [py[j] for j  in range(len(px)) if labels[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(int(u)))
plt.legend()
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("K-Means plot of Spectral Clustering " + input_file_Path)
plt.show()



