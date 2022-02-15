import os
from nb import *

def prepocess(data):
  #PREPROCESS STEPS ARE AS FOLLOWS:
  # remove company name, starting with @ char, the first word
  # remove spaces
  # remove punctiation
  # remove non-ascii chars, such as emojis
  # make lowecase 
  # remove stop words and words with digits
  # contain words with at least more than 1 chars 

  retVal = [] # resulting preprocessed list of words

  stop_words = {"i", "you", "we", "he", "she", "it", "they", "them", "me", "my", "her", "his", "their", "our", "its",
                "am" ,"is", "are", "was", "were", "to", "about", "in", "on", "at", "a", "an", "the", "but", "for", "be",
                "if"}

  for i, word in enumerate(data.split(" ")):
    if i == 0: continue #do not include the fist word
    to_add = word.strip().translate(str.maketrans('', '', string.punctuation)).encode("ascii", "ignore").decode().lower()
    if len(to_add)>1 and (to_add not in stop_words) and (not any(char.isdigit() for char in to_add)):
      retVal.append(to_add)
    #print("to_add: ", to_add)

  return retVal

if __name__ == '__main__':
  #paths to datasets
  train_data_path = os.path.join("nb_data", "train_set.txt")
  train_labels_path = os.path.join("nb_data", "train_labels.txt")
  test_data_path = os.path.join("nb_data", "test_set.txt")
  test_labels_path = os.path.join("nb_data", "test_labels.txt")



  ####################################################
  #TRAINING PROCESS: BUILD UP THE VOCAB, PI, AND THETA
  ####################################################
 
  # read data to a list of sentences, which are lists of words, applying preprocessing
  train_data_file = open(train_data_path, 'r', encoding='utf8')
  train_data = [ prepocess(data) for data in train_data_file.readlines()]  
  train_data_file.close()
  vocab = vocabulary(train_data)
  #print("vocab: ", len(vocab))
  #print("train_data: ", len(train_data))

  #read train labels
  train_labels_file = open(train_labels_path, 'r', encoding='utf8')
  train_labels = [label.strip() for label in train_labels_file.readlines()]
  train_labels_file.close()
  #print("train_labels: ", len(train_labels))
  pi = estimate_pi(train_labels)
  #print(pi)

  theta = estimate_theta(train_data, train_labels, vocab)
  #print(theta)

  
  ####################################################
  #################### TEST PROCESS ##################
  ####################################################  
  # read test data to a list of sentences, which are lists of words, applying preprocessing
  test_data_file = open(test_data_path, 'r', encoding='utf8')
  test_data = [ prepocess(data) for data in test_data_file.readlines()]  
  test_data_file.close()
  #read test labels
  test_labels_file = open(test_labels_path, 'r', encoding='utf8')
  test_labels = [label.strip() for label in test_labels_file.readlines()]
  test_labels_file.close()  

  test_results = test(theta, pi, vocab, test_data) 
  #print("return from test_results")
  correct = 0
  for i, result in enumerate(test_results):
      r = sorted(result, reverse=True)
      if test_labels[i] == r[0][1]:
          correct += 1

  #print("correct: " ,correct)

  total = len(test_labels)
  print(f"Accuracy: {correct/total:.3f}")