Density based (DBSCAN):
-Copy DBSCAN.py and the required dataset files into the same directory.
-Run the script as follows:
	python density-based-script.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter filename: <filename of dataset>
-You will be prompted to enter the value of epsilon as follows:
	Enter epsilon value: <floating point value>
-You will be prompted to enter the value of MinPts as follows:
	Enter MinPts value: <integer value>
-The algorithm runs until all the points are either put into a cluster or are marked as noise
-The Jaccard Coefficient is printed to the console.



GMM (Gaussian Mixture Model):
-Copy GMM.py and the required dataset files into the same directory.
-Run the script as follows:
	python GMM.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter filename: <filename of dataset>
-You will be prompted to enter the value of no of clusters as follows:
	Enter no of clusters: <integer value>
	Ex: 2
-You will be prompted to enter the value of mean Matrix as follows:
	Enter Mean Matrix: <float value>
	The mean matrix dimensions has to be (no of clusters)x(no of features) 	
	Ex:[[0,0],[1,1]]
-You will be prompted to enter the value of Covariance Matrix as follows:
	Enter Covariance Matrix: <float value>
	The Covariance matrix dimensions has to be (no of clusters)x(no of features)x(no of features) 
	Ex:[[[1,1],[1,1]],[[2,2],[2,2]]]	
-You will be prompted to enter the value of prior Matrix as follows:
	Enter Prior Matrix: <float value>
	Prior Matrix has to be enteres as list
	Ex: [0.5,0.5]
-You will be prompted to enter the Maximum Iterations as follows:
	Enter Maximum Iterations: <Integer value>
	Ex: 100
-You will be prompted to enter the Convergence Threshold value as follows:
	Enter Convergence Threshold: <float value>
	Ex:1e-9
-You will be prompted to enter the Smoothing Value as follows:
	Enter Smoothing Value: <float value>
	Ex:1e-9
-The algorithm runs until convergence Threshold is achieved or maximum iterations are reached
-The Jaccard and Rand Coefficient is printed to the console.

If the matrix dimensions don't match with the expected dimensions, random initialization using K-Means is used.
For Initialization using k-means you should enter [] for Mean Matrix, Covariance Matrix and Prior Matrix.



KMeans:
-Copy KMeans.py and the required dataset files into the same directory.
-Run the script as follows:
	python Kmeans.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter file name: <filename of dataset>
-You will be prompted to enter no of clusters as follows:
	Enter number of clusters: <Integer point value>
-You will be prompted to enter gene Id of cluster Centroids as follows:(If Random initialization give [] as input)
	Enter gene Id of cluster Centroids: <Integer value> 
	Ex: [1,2,3,4,5]
-You will be prompted to enter Maximum Iterations as follows:
	Enter Maximum Iterations: <Integer value>

-The algorithm runs until centroids doesn’t change or maximum iterations are reached
-The Jaccard Coefficient and Rand Index is printed to the console.



SpetralClustering:
-Copy SpectralClustering.py and the required dataset files into the same directory.
-Run the script as follows:
	python SpectralClustering.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter file name: <filename of dataset>
-You will be prompted to Enter the Sigma value as follows:
	Enter  the Sigma value: <Integer point value>
-You will be prompted to Enter number of cluster as follows:
	Enter number of clusters:: <Integer point value>
-You will be prompted to enter gene Id of cluster Centroids as follows:
	Enter gene Id of cluster Centroids: <integer value>
-The Jaccard Coefficient and Rand Index is printed to the console.




HAC(Hierarchical Agglomerative Clustering):
-Copy density-based-script.py and the required dataset files into the same directory.
-Run the script as follows:
	HAC.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter file name: <filename of dataset>
-You will be prompted to enter number of clusters as follows:
	Enter number of clusters: <integer value>

-The algorithm runs until number of clusters specified are formed
-The Jaccard Coefficient and Rand index is printed to the console. At each iteration, what clusters are getting merged are printed.








