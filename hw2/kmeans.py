import numpy as np
import matplotlib.pyplot as plt
import sys



def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    

    N, d = data.shape
    assigned_clusters = np.zeros(shape=(N, ))


    for p, point in enumerate(data):
        min_dist = 99999
        for i, center in enumerate(cluster_centers):
            distance = np.linalg.norm(point-center)
            if distance < min_dist:
                min_dist = distance
                assigned_clusters[p] = i


    return assigned_clusters





def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    k,d = cluster_centers.shape
    new_centers = np.zeros((k,d))

    for (cluster, center) in enumerate(cluster_centers):
        data_in_cluster = []
        for (assigned_cluster, data_points) in zip(assignments, data):

            if cluster == int(assigned_cluster): 
                data_in_cluster.append(data_points)

        data_in_cluster = np.asarray(data_in_cluster)


        if(data_in_cluster.shape[0]):
            new_centers[cluster] = np.mean(data_in_cluster, axis=0)
        else:
            new_centers[cluster] = cluster_centers[cluster]


    return new_centers





def init(data, k):
    random_indices = np.random.choice(data.shape[0], size=k, replace=False)
    random_rows = data[random_indices, :]

    return random_rows





def objective_func(data, assignments, cluster_centers):
    obj = 0
    for point, cluster in zip(data, assignments):
        obj += (np.linalg.norm(point-cluster_centers[int(cluster)]))**2
    return (obj/2)





def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    
    """    

    k = initial_cluster_centers.shape[0]
    #print("initial_cluster_centers.shape: ", initial_cluster_centers.shape)
    cluster_centers = initial_cluster_centers
    min_obj = 99999


    while(1):
        assignments = assign_clusters(data, cluster_centers)
        cluster_centers = calculate_cluster_centers(data, assignments, cluster_centers, k)
        obj = objective_func(data, assignments, cluster_centers)

        #loop until objective does not change
        if(min_obj - obj < 10 ** -5): break
        min_obj = obj



    return cluster_centers, min_obj

            

def elbow_method(data, number_of_config, k_range =-1):

    if k_range != -1:

        obj_values = []
        global_min_obj = 99999
        min_obj_per_k = 99999
        for config in range(0,number_of_config):
            #print("for conf: ", config)
            cluster_centers, obj = kmeans(data, init(data, k_range)) 
            #print("return from kmeans")
            if obj < min_obj_per_k:
                min_obj_per_k = obj
                optimal_cluster_centers_per_k = cluster_centers 
        obj_values.append(min_obj_per_k)
        optimal_cluster_centers = optimal_cluster_centers_per_k  
        
        return optimal_cluster_centers      




    obj_values = []
    global_min_obj = 99999

    for k in range(1,10):

        min_obj_per_k = 99999

        for config in range(0,number_of_config):

            cluster_centers, obj = kmeans(data, init(data, k)) 

            if obj < min_obj_per_k:
                min_obj_per_k = obj
                optimal_cluster_centers_per_k = cluster_centers 

        obj_values.append(min_obj_per_k)

        optimal_cluster_centers = optimal_cluster_centers_per_k

            


    plt.plot(range(1,10), obj_values, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Objective')
    plt.title('The Elbow Method using Objective')
    plt.show()  


    

def form_clusters(data, optimal_k, optimal_cluster_centers):
    #optimal_k
    #optimal_k = 4

    #print("optimal_cluster_centers: " ,optimal_cluster_centers)
    optimal_cluster_centers = optimal_cluster_centers[:optimal_k +1]
    final_assigned_clusters = assign_clusters(data, optimal_cluster_centers)

    centers_x = []
    centers_y = []
    for cluster in range(optimal_k ):
        points_x = []
        points_y = []
        for data_index, data_cluster in enumerate(final_assigned_clusters):
            if int(data_cluster) == cluster:
                points_x.append(data[int(data_index)][0])
                points_y.append(data[int(data_index)][1])

        plt.scatter(np.array(points_x), np.array(points_y))

        centers_x.append(optimal_cluster_centers[cluster][0])
        centers_y.append(optimal_cluster_centers[cluster][1])

    plt.scatter(np.array(centers_x), np.array(centers_y), marker='>', color="green", s=500)

    plt.show()



if __name__ == "__main__":

    dataset1 = np.load("hw2_material/kmeans/dataset1.npy")
    dataset2 = np.load("hw2_material/kmeans/dataset2.npy")
    dataset3 = np.load("hw2_material/kmeans/dataset3.npy")
    dataset4 = np.load("hw2_material/kmeans/dataset4.npy")

    dataset_num = None

    #default run elbow
    if len(sys.argv) == 1: 
        data = dataset1
        number_of_config = 10
        elbow_method(data, number_of_config)

    else:    
        if sys.argv[1] == "-elbow":
            if len(sys.argv) == 3:
                dataset_num = int(sys.argv[2])
                number_of_config = 10
            elif len(sys.argv) == 4:
                dataset_num = int(sys.argv[2])
                number_of_config = int(sys.argv[3])
            else: number_of_config = 10

            if dataset_num != None:
                if dataset_num == 1: data = dataset1
                elif dataset_num == 2: data = dataset2
                elif dataset_num == 3: data = dataset3
                elif dataset_num == 4: data = dataset4
            else: data = dataset1 

            #print("dataset_num :", dataset_num)
            #print("number_of_config: ", number_of_config)
            elbow_method(data, number_of_config)

        elif sys.argv[1] == "-cluster":
            dataset_num = int(sys.argv[2])
            optimal_k = int(sys.argv[3])
            number_of_config = int(sys.argv[4])
            if dataset_num == 1: data = dataset1
            elif dataset_num == 2: data = dataset2
            elif dataset_num == 3: data = dataset3
            elif dataset_num == 4: data = dataset4   

            optimal_cluster_centers = elbow_method(data, number_of_config, optimal_k) 
            form_clusters(data, optimal_k, optimal_cluster_centers)   
            print("optimal optimal_cluster_centers :", optimal_cluster_centers)     



    



  












