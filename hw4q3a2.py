import numpy as np
import pandas as pd
import scipy as scp
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection as ms
import matplotlib.pyplot as plt
from sklearn import metrics

cellA = pd.read_csv("celltypeA.txt",index_col=0, header=0)
cellB = pd.read_csv("celltypeB.txt", index_col=0, header=0)
cellC = pd.read_csv("celltypeC.txt", index_col=0, header=0)



# generate random data
data1 = cellA
data2 = cellB

# combine the datasets and then create the labels for dataset 1 as positive and dataset 2 as negative
# you can reverse the positive and negative labels by switching the ones and zeros associated with each set
X1 = np.concatenate((data1, data2))
Y1 = np.concatenate((np.zeros(len(data1)), np.ones(len(data2)))) #data1 as positive(1) and data2 as negative(0)



# store the values or auroc and aupr to make calculation on
aurocToAverage = []
auprToAverage = []

# score which will be later used to make calculation for aupr and auroc
score = np.zeros(len(Y1))

# this function will divide out model into training and testing data, 5 splits
stratifiedKFold = sk.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# this function will create our logistic regression to train on
linModel = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)

# we will split our dataset into the splits here, the split function will break our dataset into
# the correct sizes and allow us to create 5 different training and test datasets to average over
for train_index, test_index in stratifiedKFold.split(X1, Y1):
    testX1 = X1[test_index]
    trainX1 = X1[train_index]
    trainY1 = Y1[train_index]
    testY1 = Y1[test_index]

    # train the linear model on our training data
    linModel.fit(trainX1, trainY1)

    # predict the correct labels for our test dataset
    predY = linModel.predict_proba(testX1)[:, 1]
    score[test_index] = predY.copy()

    # calculate the auroc and aupr and then add them to our lists to average
    aurocToAverage.append(metrics.roc_auc_score(testY1, predY))
    auprToAverage.append(metrics.average_precision_score(testY1, predY))

print("For cellA positive and cellB negative samples:")
print("AUROC:" + str((sum(aurocToAverage) / len(aurocToAverage))))
print("AUPR:" + str((sum(auprToAverage) / len(auprToAverage))))



#calculates the false positive rate and true positive rate from our known values and our predicted scores
lr_fpr, lr_tpr, _ = sk.metrics.roc_curve(Y1, score)
x = np.linspace(0, 10, 1000)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot(range(1))



#calculates the precision and recall from our known values and predicted scores
lr_precision, lr_recall, _ = sk.metrics.precision_recall_curve(Y1, score)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
plt.plot(range(1))

plt.title("ROC Precision-Recall and Curve Plot")
plt.xlabel("FPR/Recall")
plt.ylabel("TPR/Precision")
plt.show()