from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef


#LOAD THE TEST AND TRAIN SETS
train_set = np.load("hw3_material/svm/task4/train_set.npy")
train_labels = np.load("hw3_material/svm/task4/train_labels.npy")
nsamples, nx, ny = train_set.shape
train_set = train_set.reshape((nsamples,nx*ny))

test_set = np.load("hw3_material/svm/task4/test_set.npy")
test_labels = np.load("hw3_material/svm/task4/test_labels.npy")
nsamples, nx, ny = test_set.shape
test_set = train_set.reshape((nsamples,nx*ny))

#check the class imbalance ratio:
#_, counts = np.unique(train_labels, return_counts=True)
#print("train_labels counts: ", counts)
#_, counts = np.unique(test_labels, return_counts=True)
#print("test labels: \n", counts)



def performance_metrics(test_labels, y_pred):

	c = confusion_matrix(test_labels, y_pred)
	tn, fp, fn, tp = c.ravel()
	acc = (tp + tn)/(tn+fp+ fn+ tp)
	recall = tp / (tp + fn)
	prec = tp/(tp+fp)
	f1 = 2*recall*prec / (recall+prec)
	mcc = matthews_corrcoef(test_labels, y_pred)

	print("confusion_matrix", )	
	print(c)
	print("tn, fp, fn, tp: ", tn, fp, fn, tp)	
	print("accuracy: ", acc)
	#print("recall: ", recall)
	#print("prec: ", prec)
	#print("f1 score: ", f1)
	#print("MCC: ", mcc)





#############################################
### FIRST ITEM ##############################
#############################################

# kernel = rbf  and C=1
clf = SVC(kernel="rbf", C=1)
clf.fit(train_set, train_labels)
y_pred = clf.predict(test_set)
print("FIRST ITEM:")
performance_metrics(test_labels, y_pred)
print()




#############################################	
### 2ND ITEM: OVERSAMPLING THE MINORITY CLASS
#############################################

#find the instances of minority class
train_set_cp = train_set
train_labels_cp = train_labels
for data, label in zip(train_set, train_labels):
	#if minority class
	if label == 0:
		#copy the under-represented class instances 19 times
		i= 18
		while i:
			i-=1	
			train_set_cp = np.vstack([train_set_cp, data])
			train_labels_cp = np.append(train_labels_cp, label)

#print("AFTER OVERSAMPLING:")
#_, counts = np.unique(train_labels_cp, return_counts=True)
#print("train_labels_cp counts: ", counts)
#print("train_set_cp: ", train_set_cp.shape)

clf = SVC(kernel="rbf", C=1)
clf.fit(train_set_cp, train_labels_cp)
y_pred = clf.predict(test_set)
print("SECOND ITEM")
performance_metrics(test_labels, y_pred)
print()

		

##############################################
### 3RD ITEM: UNDERSAMPLING THE MAJORITY CLASS
##############################################

train_set_cp = train_set
train_labels_cp = train_labels
# find  the indices where label = 1
result = np.where(train_labels == 1)
#delete elements in those indices
to_del = (result[0][0:900])
train_labels_cp = np.delete(train_labels, to_del)
#print("UNDERSAMPLED train_labels \n", train_labels)

train_set_cp = np.delete(train_set, to_del, axis=0)
#print("UNDERSAMPLED train_labels_cp \n", train_set)

clf = SVC(kernel="rbf", C=1)
clf.fit(train_set_cp, train_labels_cp)
y_pred = clf.predict(test_set)
print("THIRD ITEM")
performance_metrics(test_labels, y_pred)
print()


##########################################################
### 4TH  ITEM: using class weight parameter of sckit learn
##########################################################

clf = SVC(kernel="rbf", C=1, class_weight = "balanced")
clf.fit(train_set, train_labels)
y_pred = clf.predict(test_set)
print("FOURTH ITEM")
performance_metrics(test_labels, y_pred)

