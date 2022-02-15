# Ceng 499 - Special Topics: Introduction to Machine Learning - HW3

There are two parts of this homework, namely, decision tree implementation and tests with SVMs.

## PART I: DECISION TREES
1. dt.py: Implements the given functions.

## PART II: SVM

### Files:
1. draw_svm.py: Contains the function to draw boundaries of SVMs.
2. svm_task1.py: Visualizes decision boundaries of each SVM classifier trained with C values 0.01, 0.1, 1, 10, 100 sequentially. (Need to close the pop-up image to continue with the plot trained with the next C value).
2. svm_task2.py: Visualizes decision boundaries of each SVM classifier trained with different kernel functions, namely, "linear", "poly", "rbf", "sigmoid", sequentially. (Need to close the pop-up image to continue with the plot trained with the next C value).
2. svm_task3.py: Has two modes, ie, it either performs grid search or tests with the best hyperparameter configuration. When run in grid search mode, it splits data to train/validation sets and prints the hyperparameter configuration and the accuracy result on validation sets. The grid search is on the following: kernels =["linear", "poly", "rbf", "sigmoid"], C = [0.01, 0.1, 1, 10, 100], gammas = [0.001, 0.01, 0.1, 1]. ​When run in test mode, it classifies the test set based on the predetermined optimal hyperparameters, and prints the accuracy on the console. 
2. svm_task4.py: Sequentially runs for all four the points stated in the homework text, and prints the confusion matrix, TP, FP, TN, FN, and accuracy results on the console for each.


### Usage
To run svm_task1.py
```bash
python svm_task1.py
```
To run svm_task2.py
```bash
python svm_task2.py
```
To run svm_task3.py in grid search mode 
```bash
python svm_task3.py grid_search
```
To run svm_task3.py in test mode 
```bash
python svm_task3.py test
```
To run svm_task4.py
```bash
python svm_task4.py
```



