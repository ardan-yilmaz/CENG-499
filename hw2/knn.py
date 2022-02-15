import numpy as np
import matplotlib.pyplot as plt
import sys


def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    dists = []
    if distance_metric == "L2":        
        for train_instance in train_data:
            dists.append(np.linalg.norm(train_instance-test_instance))
        

    elif distance_metric == "L1":
        for train_instance in train_data:
            dists.append(sum(abs(val1-val2) for val1, val2 in zip(train_instance,test_instance)))

    return np.array(dists)



        



def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """

    sorted_indices = np.argsort(distances)
    votes = []
    for i in range(0,k):
        index = sorted_indices[i]
        votes.append(labels[index])

    max_votes = np.bincount(np.array(votes)).argmax()

    return max_votes






def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """

    test_preds = []
    N,d = test_data.shape

    for i in range(N):
        distances = calculate_distances(train_data, test_data[i], distance_metric)
        cluster = majority_voting(distances, train_labels, k)
        test_preds.append(cluster)

    correct = (test_preds == test_labels).sum()
    return correct/N







def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """

    N,d = whole_train_data.shape

    split_data = np.array_split(whole_train_data, k_fold)
    split_labels = np.array_split(whole_train_labels, k_fold)

    

    validation_data = split_data[validation_index]
    validation_labels = split_labels[validation_index]

    split_data.pop(validation_index)
    split_data = np.concatenate(split_data)

    split_labels.pop(validation_index)
    split_labels = np.concatenate(split_labels)


    return split_data, split_labels, np.array(validation_data), np.array(validation_labels)




def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    #val_indices = int(whole_train_labels.shape[0]/k_fold)
    acc = []
    for val_index in range(k_fold):
        train_data, train_labels, test_data, test_labels = split_train_and_validation(whole_train_data, whole_train_labels, val_index, k_fold)
        acc.append(knn(train_data, train_labels, test_data, test_labels, k, distance_metric))

    return np.mean(np.array(acc))



if __name__ == "__main__":
    train_set = np.load("hw2_material/knn/train_set.npy")
    train_labels = np.load("hw2_material/knn/train_labels.npy")
    test_set = np.load("hw2_material/knn/test_set.npy")
    test_labels = np.load("hw2_material/knn/test_labels.npy")

    k_fold = 10

    if len(sys.argv) == 1:
        distance_metric = 'L2'
        

    elif len(sys.argv) == 2:
        distance_metric = sys.argv[1]



    ##### TRAIN
    acc = []
    max_acc = -1
    opt_k = -1
    #find the optimum k for k-NN using 10-fold CV on train set
    for k in range(1,180):
        avg_acc = cross_validation(train_set, train_labels, k_fold, k, distance_metric)
        acc.append(avg_acc)
        if avg_acc > max_acc:
            max_acc = avg_acc
            opt_k = k
        



    print("optimal k: ", opt_k, " with training accuracy: ", max_acc)

    plt.plot(range(1,180), acc) # Drawing a line
    plt.xlabel('k for k-NN')
    plt.ylabel('Average accuracy')
    plt.show()

    ## TEST
    opt_k = 13
    acc = knn(train_set, train_labels, test_set, test_labels, opt_k, distance_metric)
    print("test accuracy with  ", distance_metric, " : ", acc)







