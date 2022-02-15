import math
import numpy as np

def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    ent = 0
    total = np.sum(np.array(bucket))

    for el in bucket:
        if el == 0: continue 
        else: p = el/total
        ent += (p * math.log(p,2))
    return -1*ent






def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    ES = entropy(parent_bucket)    

    total_num_instances = np.sum(np.array(parent_bucket))
    num_left = np.sum(np.array(left_bucket))
    left_entropy = entropy(left_bucket)

    num_right = np.sum(np.array(right_bucket))
    right_entropy = entropy(right_bucket) 

    IS = (num_left/total_num_instances)*left_entropy + (num_right/total_num_instances)*right_entropy

    return ES - IS





def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    gini = 0
    total_num_instances = np.sum(np.array(bucket))
    for el in bucket:
        gini += (el/total_num_instances)**2
    return 1 - gini



def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    left_gini = gini(left_bucket)
    right_gini = gini(right_bucket)

    total_left = np.sum(np.array(left_bucket))
    total_right = np.sum(np.array(right_bucket))
    total = total_left+total_right

    return left_gini*(total_left / total) + right_gini*(total_right / total)


def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    sorted_indices = data[:, attr_index].argsort()
    data_sorted = data[sorted_indices]
    sorted_labels = labels[sorted_indices] 

    
    retVal = []  

    N = (data.shape)[0]

    for i in range(N-1):
        split_val = (data_sorted[i][attr_index]+data_sorted[i+1][attr_index])/2
        #print(split_val)

        right_bucket = [0]*num_classes
        left_bucket = [0]*num_classes
        parent_bucket = [0]*num_classes

        for j, label in enumerate(sorted_labels):
            parent_bucket[label] += 1
            if j <= i:  left_bucket[label] += 1
            else: right_bucket[label] += 1


        if heuristic_name == "info_gain": 

            val = info_gain(np.asarray(parent_bucket), np.asarray(left_bucket), np.asarray(right_bucket))
            #print("info_gain: ", val)

        elif heuristic_name == "avg_gini_index": 
            val = avg_gini_index(np.asarray(left_bucket), np.asarray(right_bucket))
            #print("avg_gini_index: ", val)

        retVal.append([split_val, val])       
        
    return np.asarray(retVal)





def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """

    left_bucket = np.array(left_bucket)
    right_bucket = np.array(right_bucket)

    num_classes_init = left_bucket.shape[0]


    #num_classes = len(left_bucket)

    left_total = np.sum(left_bucket)
    right_total = np.sum(right_bucket)
    grand_total = right_total + left_total   

    col_totals = []
    for i in range(num_classes_init):
        col_totals.append(left_bucket[i] + right_bucket[i])

    expected_left = []
    expected_right = []
    num_classes = 0
    for i in range(num_classes_init):
        if(left_bucket[i] == 0 and right_bucket[i] == 0): continue

        exp_l = (left_total*col_totals[i])/grand_total
        sq_dif_l = (left_bucket[i] - exp_l)**2 
        expected_left.append(sq_dif_l/exp_l)

        exp_r = (right_total*col_totals[i])/grand_total
        sq_dif_r = (right_bucket[i] - exp_r)**2 
        expected_right.append(sq_dif_r/exp_r)  

        num_classes +=1      

    chi_squared = np.sum(np.array(expected_left)) + np.sum(np.array(expected_right))
    return chi_squared, (num_classes-1)







if __name__ == "main":
    print( "build tree w/ nd3 alg")
