import numpy as np
import matplotlib.pyplot as plt
import sys



def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    min_dist = 99999
    for p1 in c1:
        for p2 in c2:
            curr_dist = np.linalg.norm(p1-p2)
            if curr_dist < min_dist: min_dist = curr_dist
    return min_dist



def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """

    max_dist = -1
    for p1 in c1:
        for p2 in c2:
            curr_dist = np.linalg.norm(p1-p2)
            if curr_dist > max_dist: max_dist = curr_dist
    return max_dist    


def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    N = c1.shape[0]
    M = c2.shape[0]
    total_dist = 0
    for p1 in c1:
        for p2 in c2:
            total_dist += np.linalg.norm(p1-p2)

    return total_dist/(N*M)



def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    center1 = np.mean(c1,axis=0)
    center2 = np.mean(c2,axis=0)
    return np.linalg.norm(center1-center2)





def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """

    clusters = [[x] for x in data.tolist()]
    #print("init clusters \n", clusters)
    num_clusters = len(clusters)   

    while(1):
        min_dist = 99999
        closest_index_i = 99999
        closest_index_j = 99999
        for i in range(0, num_clusters):            
            for j in range(i+1, num_clusters):               
     
                #find the closest pair
                d = criterion(np.array(clusters[i]),np.array(clusters[j]))
                if d < min_dist:
                    closest_index_i = i
                    closest_index_j = j 
                    min_dist = d

        #merge the closest pair
        merged_cluster = clusters[closest_index_i] + clusters[closest_index_j]
        #pop the old clusters
        clusters.pop(closest_index_i)
        if closest_index_j > closest_index_i: closest_index_j -= 1
        clusters.pop(closest_index_j)
        #append the merged one
        clusters.append(merged_cluster)

        #stop merging at the desired num of clusters        
        num_clusters -= 1
        if num_clusters == stop_length: break

    return [np.asarray(x) for x in clusters]


if __name__ == "__main__":

    #dataset1 = np.load("hw2_material/hac/dataset1.npy")
    #dataset2 = np.load("hw2_material/hac/dataset2.npy")
    #dataset3 = np.load("hw2_material/hac/dataset3.npy")
    #dataset4 = np.load("hw2_material/hac/dataset4.npy")

    data_num = int(sys.argv[1])
    stop_length = int(sys.argv[2])
    criterion = sys.argv[3]

    if data_num == 1: data = np.load("hw2_material/hac/dataset1.npy")
    elif data_num == 2: data = np.load("hw2_material/hac/dataset2.npy")
    elif data_num == 3: data = np.load("hw2_material/hac/dataset3.npy")
    elif data_num == 4: data = np.load("hw2_material/hac/dataset4.npy")



    #criteria = [single_linkage, complete_linkage, average_linkage, centroid_linkage]

    if criterion == "single_linkage": criterion = single_linkage
    elif criterion == "complete_linkage": criterion = complete_linkage
    elif criterion == "average_linkage": criterion = average_linkage
    elif criterion == "centroid_linkage": criterion = centroid_linkage



    clusters = hac(data, criterion, stop_length)

    for cluster in clusters:
        points_x = []
        points_y = []
        for point in cluster:
            points_x.append(point[0])
            points_y.append(point[1])
        plt.scatter(np.array(points_x), np.array(points_y))
    plt.show()


    




   




    