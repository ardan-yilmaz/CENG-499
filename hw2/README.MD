# Ceng 499 - Special Topics: Introduction to Machine Learning - HW2

Three different algorithms, namely, KNN, K-means, and hierarchical agglomerative clustering (hac), have been implemented in different files whose details can be found in the relative 
section.

## Files
1. knn.py: Applies 10-fold cross-validation on the training dataset. Tries k= 1,...,180 and maximizing the accuracy finds the optimal k for the KNN algorithm using L1 or L2 distances, 
then calculates the accuracy of the test set on the optimized k. 

2. kmeans.py: Applies k means clustering algorithm. Tries k=1,...,10 with a given number of different initial configurations, and for each k the accuracy produced by the best initial 
configuration is recorded and plotted, which can be used for the elbow method. After determining the optimal k, it clusters the datasets provided and visualizes the formed clusters. 

3. hac.py: Applies hierarchical agglomerative clustering. Starting with clusters for each data, it merges until the given stop criterion (number of clusters) is met. It can perform 
single, complete, average, centroid-linkages, and plots the results. 

## Usage

### 1. knn.py:

The user can provide the distance metric as a command-line argument. If not provided, the default value is L2. It directly calculates the optimal k using the train_set.npy and 
train_labels.npy datasets under "hw2_material/knn/". Then tests for the test_set.npy and test_labels.npy under the same directory. It first shows the graph of k vs. accuracy, and prints 
the optimal number k, and the calculated accuracy for the test set on the screen, as follows.
```
optimal k:  13  with training accuracy:  0.8090909090909092
test accuracy with L1:  0.8388888888888889
```
The following are the bash commands to run knn.py.

The following uses L2 distance by default.
```bash
python knn.py
```
The following two use L1 and L2 distances respectively.
```bash
python knn.py L1
```
```bash
python knn.py L2
```

### 2. kmeans.py: 
It mainly has two modes. The first one tries different k values ranging from 1 to 10; and by using different initializations for each k, plots the k vs objective diagram where one 
can apply the elbow method to determine the optimum k. The second is to plot the clusters given the optimal k. Users can specify the mode and the optimal k. In fact, running this script 
with both modes consecutively, one can both determine the optimal k and plot the clusters. That is, with the first run in the elbow method, one can determine the number of clusters, and 
then running in the cluster mode, one can see the clusters formed.

The mode defaults to "elbow" whose details can be found below.

#### Run in elbow mode
This mode both plots the graph of k vs objective and prints the optimal cluster centers on the console indicated by green arrow-heads on the cluster plots.

Run for the first mode (to find the optimal k):
```bash
python kmeans.py -elbow <dataset_number> <number_of_init_configurations>
```
Parameters:

* dataset_number indicates which dataset under "hw2_material/kmeans/" path to load and run. It can be 1, 2, 3, or 4. The default value is 1, loading "hw2_material/kmeans/dataset1.npy".

* number_of_init_configurations is the number of initial configurations tried for each k. The default value is 10.

Example runs for kmeans:

The first three runs basically the same due to the default settings explained. Runs for dataset 1 with 10 different initial configurations tried for each k.
```bash
python kmeans.py
```
```bash
python kmeans.py -elbow 
```
```bash
python kmeans.py -elbow 1 10
```
The following one runs in the elbow method, ie, plots a graph of k vs objective function for dataset 2 where each k is run with 15 different initial configurations.

```bash
python kmeans.py -elbow 2 15
```

#### Run in cluster mode
Provided with a dataset, optimal number of clusters, and a number of configurations to try to find the cluster centers, it plots the clusters. Also prints the optimal cluster centers 
on the console.

```bash 
python kmeans.py -elbow <dataset_number> <optimal_k> <number_of_init_configurations>
```
Parameters:

* optimal_k: indicates the number of clusters to be formed.  

* dataset_number: indicates which dataset under "hw2_material/kmeans/" path to load and run. It can be 1, 2, 3, or 4. 

* number_of_init_configurations is the number of initial configurations tried for each k. 

Example runs:

The following plots the "hw2_material/kmeans/dataset1.npy" with 2 clusters whose central points have been found with 10 different initial configurations.

```bash 
python kmeans.py -cluster 1 2 10
```
The following plots 4 clusters formed using dataset 4.

```bash 
python kmeans.py -cluster 4 4 10
```



### 3. hac.py:

Given a dataset, stop condition for merging, and linkage criterion, it applies hierarchical agglomerative clustering.

```bash 
python hac.py <dataset_number> <stop> <criterion>
```
Parameters:

* dataset_number: indicates which dataset under "hw2_material/kmeans/" path to load and run. It can be 1, 2, 3, or 4.
* stop: The number of clusters formed in the end, ie, stop condition for merging the clusters.
* criterion: to evaluate the inter-cluster distances. It can be single_linkage, complete_linkage, average_linkage, or centroid_linkage.

Example runs:
The following runs for dataset 1 until 2 clusters remain using the complete linkage method.

```bash 
python hac.py 1 2 complete_linkage
```

## Additional Notes

* Some runs might take a while especially for k-means.
