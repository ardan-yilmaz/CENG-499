# Ceng 499 - Special Topics: Introduction to Machine Learning - HW4


## Files
1. nb.py: Given functions are implemented.  
2. nb_classifier.py: Building up a vocabulary using the training set, the script classifies the test sentences using Naive Bayes algorithm, and prints the test accuracy on the console.
3. hmm.py: Given functions, ie, forward, and viterbi are implemented. 

## Usage

1. nb.py: Given functions are implemented. Importing this file, it can be run with a tester directly. An example run is given below.

```bash
python nb_simple_tester.py
``` 

2. nb_classifier.py: Running this script directly builds up a dictionary from the training set under relative path "/nb_data", ie, using the files "/nb_data/train_set.txt" and  
"/nb_data/train_labels.txt". Upon building the dictionary, it first loads the data and applies preprocessing whose details are given as comments in the preprocess function 
in this script. Then, using the algorithm defined in the homework text, it finalizes building the dictionary, by which it classifies the sentences in the test set which 
is under the same relative path as training data. The resulting accuracy on the test set is printed on the console. 

```bash
python nb_classifier.py
``` 


3. hmm.py: Viterbi, and Forward algorithms are implemented in the relative functions. Importing this file, it can be run with a tester directly. An example run is given below.
```bash
python hmm_simple_tester.py
``` 
