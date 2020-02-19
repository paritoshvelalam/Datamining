
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA


#np.random.seed(123)
filepath=input("Enter file name: ")
k=int(input("Enter number of clusters: "))
init_mean = np.array(eval(input("Enter Mean Matrix: ")), dtype = float)
init_covar = np.array(eval(input("Enter Covariance Matrix: ")), dtype = float)
init_prior = np.array(eval(input("Enter Prior Matrix: ")), dtype = float).reshape(1,-1)
maxiter = int(input("Enter Maximum Iterations: "))
epsilon = float(input("Enter Convergence Threshold: "))
smoothingvalue = float(input("Enter Smoothing Value: "))
# filepath = "D:\\Buffalocourses\\secondsem\\dataminingandbioinformatics\\project2\\cho.txt"
# k = 5
# init = []
# maxiter = 100
# epsilon = 1e-9
# maxiter = 600

filedata = np.loadtxt(filepath, delimiter="\t")
data = filedata[:,2:]
truelabels = filedata[:,1]
truelabels = np.reshape(truelabels, (len(truelabels),1))

def KMeans(k, init, maxiter, data):
    ## Initialize Centroids
    print("Using KMeans Initialization")
    if(len(init)==k):
        init=[i-1 for i in init]
        centroids= data[init,:]
    else:
        print("Using random centroids.")
        perm=np.random.permutation(len(data))
        centroids= data[perm[0:k]]

    ##Run K-Means algorithm
    iterations = 0
    oldCentroids = np.zeros(shape = centroids.shape, dtype = float)
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
    
    ## Variance of data within clusters and prior
    prior = np.zeros(shape = (1, k), dtype = float)
    covar = np.zeros(shape = (k, data.shape[1], data.shape[1]), dtype = float)
    for i in range(k):
        clusterdata = datawithclusterID[datawithclusterID[:,0] == int(i)][:,1:]
        covar[i] = (1/clusterdata.shape[0])*clusterdata.T.dot(clusterdata)
        
        prior[0][i] = len(clusterdata)/len(data)
    
    return clusterID, centroids, covar, prior

if ((init_mean.shape[0] == k) and (init_mean.shape[1] == data.shape[1]) and (init_covar.shape[0] == k) and (init_covar.shape[1] == data.shape[1]) and (init_covar.shape[2] == data.shape[1]) and (init_prior.shape[1]==k)):
    clusterID = np.zeros(shape = (data.shape[0], 1))
    means = np.copy(init_mean)
    pie = np.copy(init_prior)
    covar = np.copy(init_covar)
else:
    GMM_init_clusterID, GMM_init_mean, GMM_init_covar, GMM_init_prior = KMeans(k, [], maxiter, data)
    clusterID = np.copy(GMM_init_clusterID)
    means = np.copy(GMM_init_mean)
    pie = np.copy(GMM_init_prior)
    covar = np.copy(GMM_init_covar)


resp = np.zeros(shape = (data.shape[0], k), dtype = float)
iterations = 0

log_likelihood = 0
while(iterations < maxiter):
    ## E Step
    #print("start")
    for i in range(k):
        prior = pie[0][i]
        likelihood = multivariate_normal.pdf(data, means[i], covar[i], allow_singular = True)
        resp[:,i] = (prior)*likelihood
    resp = resp/resp.sum(axis = 1).reshape(-1,1)
    #print("resp")
    #print(resp.shape)
    #print(resp[0])
    ## M Step
    Nk = resp.sum(axis = 0).reshape(1,-1)  #1xk
    #print("Nk")
    #print(Nk.shape)
    #print(Nk)
    pie = Nk/len(data)   #1xk
    #print(pie.shape)
    #print(pie)
    means = np.dot(resp.T, data) #kxf
    #print(means)
    means = means / Nk.reshape(-1,1)  #kxf
    #print("means")
    #print(means.shape)
    #print(means)
    Nk.reshape(1,-1)
    for i in range(k):
        diff = (data - means[i]).T
        num = np.dot((resp[:,i]*diff), diff.T)
        covar[i] = num/Nk[0][i]     #kxfxf

    ## Loglikelihood
    log_likehood_old = log_likelihood
    log_likelihood = 0
    #print(smoothingvalue)
    for i in range(k):
        for j in range(covar.shape[1]):
            covar[i][j][j] = covar[i][j][j] + smoothingvalue
    #print("covar")
    #print(covar.shape)
    #print(covar)
    for i in range(k):
        prior = pie[0][i]
        mean = means[i]
        cov = covar[i]
        cov_inverse = np.linalg.inv(cov)
        cnst = (-1/2)*((k*np.log(2*np.pi)) + np.log(np.linalg.det(cov)))
        for j in range(len(data)):
            diff = data[j] - mean
            exp = np.dot(diff.T, np.dot(cov_inverse, diff))
            log_likelihood += cnst - (1/2)*(exp) + np.log(prior)
    #print(log_likehood_old - log_likelihood)
    if ((abs(log_likehood_old - log_likelihood) <= epsilon)):
        break 
        
    iterations += 1
            
clusterID[:,0] = np.argmax(resp, axis = 1)


## metrics
groundTruth = (truelabels.T == truelabels).astype(int)
clustering = (clusterID.T == clusterID).astype(int)
table = (groundTruth == clustering)
randIndex = table.sum()/table.size
table1 =  groundTruth & clustering
table2 = groundTruth | clustering
JaccardIndex = table1.sum()/table2.sum()

print("Jaccard Index: {}".format(JaccardIndex))
print("Rand Index: {}".format(randIndex))

print("Prior: {}".format(pie))
print("Means: {}".format(means))
print("Covariance: {}".format(covar))
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
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [px[j] for j  in range(len(px)) if labels[j] == u]
    yi = [py[j] for j  in range(len(px)) if labels[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(int(u)))

plt.legend()
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("GMM plot")
plt.show()
