import numpy as np
import math
import os
import string

def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    retVal = set()
    for s in data:
      #print("sentence: ",s)
      for w in s:
        #print("word: ", w)
        retVal.add(w)
    return retVal
    

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """

    cnt = 0
    retVal = dict()
    for label in train_labels:
      cnt += 1
      if label in retVal.keys():
        retVal[label] += 1
      else: retVal[label] = 1

    for label in retVal.keys():
      retVal[label] /= cnt

    return retVal


    
    
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """

    

    dict_per_class = dict() # {<class>: {<char>: <num_of_char>}}
    for data, label in zip(train_data, train_labels):
      
      elements, repeats = np.unique(np.asarray(data), return_counts=True)     

      if label not in dict_per_class.keys():
        inner_dict = dict() #{<char>: <num_of_char>} 
        for e, cnt in zip(elements, repeats):
          inner_dict[e] = cnt         
        dict_per_class[label] = inner_dict

      else: 
        for e, cnt in zip(elements, repeats):
          if e not in dict_per_class[label].keys(): dict_per_class[label][e] = cnt
          else: dict_per_class[label][e] += cnt


    classes= np.unique(train_labels)
    d = len(vocab)
    retVal = dict()
    for c in classes:
      num_total_words_in_class = sum(dict_per_class[c].values())
      denom = num_total_words_in_class +d
      per_class = dict()
      for word in vocab:
        if word not in dict_per_class[c].keys(): 
          per_class[word] = 1/denom
          continue 
        num = dict_per_class[c][word] + 1
        per_class[word] = (num/denom) 

      retVal[c] = per_class

    return retVal







    

def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """

    retVal = []    
    for data in test_data:
      per_class = []
      for _class in pi.keys():
        summ = 0
        for word in data:
          if word not in vocab: continue
          summ += math.log(theta[_class][word])
        per_class.append((summ + math.log(pi[_class]), _class))

      retVal.append(per_class)



    return retVal

