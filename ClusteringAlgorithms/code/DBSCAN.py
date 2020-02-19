import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def dbscan(data, eps, minpts):
    clusterid = 0
    Newlabel = np.zeros(data.shape[0])
    
    for pt in range(0, data.shape[0]):
        if Newlabel[pt] != 0: 
            continue
        neighbors1 =[]
        neighbors1 = neighbors(data, eps, pt)
        
        
        if len(neighbors1) < minpts:
            Newlabel[pt] = -1
            continue
        clusterid += 1
        Newlabel[pt] = clusterid 
        
        
        for point in neighbors1:
          if Newlabel[point] == -1:
             Newlabel[point] = clusterid 
          if Newlabel[point] == 0:
              Newlabel[point] = clusterid
              pointneighbor = neighbors(data, eps, point)
              if len(pointneighbor) >= minpts:
                  neighbors1 += pointneighbor


    unique, counts = np.unique(Newlabel, return_counts= True)
    unique = [int(i) for i in unique]
    print("Cluster: count = " + str(dict(zip(unique, counts))))
    return  Newlabel
        
        

def neighbors(data, eps, point):
    
    neighbors1 = []
    
    for pt in range(0, data.shape[0]):
        if np.linalg.norm(data[point] - data[pt]) <= eps:
            neighbors1.append(pt)
    
    return neighbors1
  
def plot_pca(classes_list, feature_matrix):
    # get the unique list of classes
    unique_classes_list = list(set(classes_list))
    if(unique_classes_list == 2):
      principle_components_matrix[:,0] = feature_matrix[:,0]
      principle_components_matrix[:,1] = feature_matrix[:,1]
    else:
      # obtain the principle components matrix
      pca_object = PCA(n_components=2, svd_solver='full')
      pca_object.fit(feature_matrix)
      principle_components_matrix = pca_object.transform(feature_matrix)

    # plot cluster_ids using the principle components as the coordinates and classes as labels
    colors = [plt.cm.jet(float(i) / max(unique_classes_list)) for i in unique_classes_list]
    for i, u in enumerate(unique_classes_list):

        xi = [p for (j,p) in enumerate(principle_components_matrix[:,0]) if classes_list[j] == u]
        yi = [p for (j,p) in enumerate(principle_components_matrix[:,1]) if classes_list[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(int(u)))

    plt.title(input_file.split(".")[0] + " scatter plot")
    plt.xlabel("Principle_component_1")
    plt.ylabel("Principle_component_2")
    plt.legend()
    plt.show()
                
    
input_file = '/content/iyer.txt'

input_file = input('Enter the Filename: ')
data = np.loadtxt(input_file, dtype='float')
data_x = data[:, 2:]
epsilon = float(input('Enter epsilon(radius) value: '))
minimumpoints = int(input('Enter the minimum number of points: '))

DBSCAN_Labels = dbscan(data_x, epsilon, minimumpoints)


groundTruth_cluster_id_list = data[:, 1]


clusterID = DBSCAN_Labels.reshape(-1,1)
truelabels = data[:, 1].reshape(-1,1)
groundTruth = (truelabels.T == truelabels).astype(int)
clustering = (clusterID.T == clusterID).astype(int)
table = (groundTruth == clustering)
randIndex = table.sum()/table.size
table1 =  groundTruth & clustering
table2 = groundTruth | clustering
JaccardIndex = table1.sum()/table2.sum()
print("JaccardIndex" ,JaccardIndex);
print("randIndex" ,randIndex);

plot_pca(DBSCAN_Labels,data_x)

