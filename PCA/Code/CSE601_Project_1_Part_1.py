import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

data_file = sys.argv[1]
print(data_file)
data_file_name=data_file
#data_file="pca_a.txt"
#data_file_name="pca_a.txt"
with open(data_file, 'r') as f: #open the file
    contents = f.readlines() 
l=len(contents[0].strip().split('\t'))
l2=[]
disease_array=[]

for line in  contents:
    l1=line.split('\t')[:l-1]
    l3=line.split('\t')[l-1:]
    l3=list(map(lambda x:x.strip(),l3))
    l1=list(map(float,l1))
    disease_array.append(l3)
    l2.append(l1)
    
    
data=np.array(l2)
# print(data)
disease_array1=np.array(disease_array)
#disease_array-contains all the unique diseases from input file
disease_array=np.unique(disease_array1)
#finding mean of the all columns
data_mean=np.mean(data,axis=0)
data_mean=data_mean.reshape(1,l-1)
#subracting mean from original data
new_data=np.subtract(data,data_mean)
#finding covariance matrix
data_cov=np.cov(new_data.T)
#finding eigen_value ,eigen_vector
eigen_value, eigen_vectors = LA.eig(data_cov)
new_dimension=eigen_vectors[:,0:2]
#transform data to lower dimension
trnsformed_data=np.dot(data,new_dimension)
disease_array_linspace=np.linspace(0,1,len(disease_array))
colours=dict(zip(disease_array,disease_array_linspace))

col = []
plt1 = plt.figure(1)
for i in range(len(disease_array)):
    col.append(plt.cm.jet(float(i)/len(disease_array)))    
disease_map = {}
c=0;
for disease in disease_array:
    disease_map[disease]=c
    c+=1
disease_num=[]
for disease in disease_array1:
    disease_num.append(disease_map[disease[0]])
# print(disease_num)
#plotting scatter plot
disease_list_unique=disease_array
for i,dis in enumerate(disease_array):
    data_x = []
    data_y = []
    for index,xi in enumerate(trnsformed_data[:,0]):
        if disease_num[index] == i:
            data_x.append(xi)
    for index,yi in enumerate(trnsformed_data[:,1]):
        if disease_num[index] == i:
            data_y.append(yi)
    plt.scatter(data_x, data_y, c=col[i], label=dis)
plt.title("PCA : "+ data_file_name + " scatter plot")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
#plt.show()

#SVD plot
U, s, V = LA.svd(data)
dim_red =2
pca = U[:,:dim_red]
plt2 = plt.figure(2)
## SCATTER PLOT
col = []
for i in range(len(disease_list_unique)):
    col.append(plt.cm.jet(float(i)/len(disease_list_unique)))
for i,dis in enumerate(disease_list_unique):
    x = []
    y = []
    for it,xi in enumerate(pca[:,0]):
        if disease_num[it] == i:
            x.append(xi)
    for it,yi in enumerate(pca[:,1]):
        if disease_num[it] == i:
            y.append(yi)
    plt.scatter(x, y, c=col[i], label=dis)
plt.title("SVD : "+data_file_name + " scatter plot")
plt.legend()
plt.xlabel("Component 1")
plt.ylabel("Component 2")
#plt.show()



tsne = TSNE(n_components=2,init="pca", n_iter=1000,learning_rate=100)
#tsne = TSNE(n_components=2)
pca = tsne.fit_transform(data)
    
plt3 = plt.figure(3)
## SCATTER PLOT
col = []
for i in range(len(disease_list_unique)):
    col.append(plt.cm.jet(float(i)/len(disease_list_unique)))
for i,dis in enumerate(disease_list_unique):
    x = []
    y = []
    for it,xi in enumerate(pca[:,0]):
        if disease_num[it] == i:
            x.append(xi)
    for it,yi in enumerate(pca[:,1]):
        if disease_num[it] == i:
            y.append(yi)
    plt.scatter(x, y, c=col[i], label=dis)
plt.title("TSNE : "+data_file_name + " scatter plot")
plt.legend()
plt.show()
