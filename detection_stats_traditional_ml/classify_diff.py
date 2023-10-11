"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

Script for training and testing the traditional-ml-based-classifiers with differences features
"""

import csv
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--train', default ='dataset/classifying_train.csv', help='path to train set')
# parser.add_argument('--test', default ='dataset/classifying_test.csv', help='path to test set')
# parser.add_argument('--train', default ='dataset/classifying_train_targeted.csv', help='path to train set')
# parser.add_argument('--test_1', default ='dataset/classifying_test_targeted.csv', help='path to test set')
# parser.add_argument('--test_2', default ='dataset/classifying_test_nontargeted.csv', help='path to test set')
parser.add_argument('--train', default ='dataset/classifying_train_nontargeted.csv', help='path to train set')
parser.add_argument('--test_1', default ='dataset/classifying_test_nontargeted.csv', help='path to test set')
parser.add_argument('--test_2', default ='dataset/classifying_test_targeted.csv', help='path to test set')
parser.add_argument('--output', type=str, default='data', help='output folder')

opt = parser.parse_args()
print(opt)

reader_train = csv.reader(open(opt.train, 'r'), delimiter=',')
train_set = np.array(list(reader_train)).astype('float')

np.random.shuffle(train_set)

(row, col) = train_set.shape

clf = SVC(gamma='auto', probability=True)
# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# clf = LinearDiscriminantAnalysis()
# clf = MLPClassifier()

# full
data = train_set[:, 0:-1]
label = train_set[:, -1]

# jpeg
# data = train_set[:, 0:5*16]
# label = train_set[:, -1]

# gaussian
# data = train_set[:, 5*16:5*16+5*4]
# label = train_set[:, -1]

# rotation
# data = train_set[:, 5*16+5*4:5*16+5*4+5*8]
# label = train_set[:, -1]

# scale
# data = train_set[:, 5*16+5*4+5*8:-1]
# label = train_set[:, -1]

# jpeg + scale
# data = np.concatenate((train_set[:, 0:5*16], train_set[:, 5*16+5*4+5*8:-1]), axis=1)
# label = train_set[:, -1]

clf.fit(data, label)

#-----------------------------------------------------------

# reader_test = csv.reader(open(opt.test, 'r'), delimiter=',')
# test_set = np.array(list(reader_test)).astype('float')

# full
# data = test_set[:, 0:-1]
# label = test_set[:, -1]

# jpeg
# data = test_set[:, 0:5*16]
# label = test_set[:, -1]

# gaussian
# data = test_set[:, 5*16:5*16+5*4]
# label = test_set[:, -1]

# rotation
# data = test_set[:, 5*16+5*4:5*16+5*4+5*8]
# label = test_set[:, -1]

# scale
# data = test_set[:, 5*16+5*4+5*8:-1]
# label = test_set[:, -1]

# jpeg + scale
# data = np.concatenate((test_set[:, 0:5*16], test_set[:, 5*16+5*4+5*8:-1]), axis=1)
# label = test_set[:, -1]

# pred = clf.predict(data)
# pred_prob = clf.predict_proba(data)

# acc = metrics.accuracy_score(label, pred)
# fpr, tpr, thresholds = roc_curve(label, pred_prob[:,1], pos_label=1)
# eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
# roc_auc = metrics.auc(fpr, tpr)

# print('Accuracy: %.2f, EER: %.2f' % (acc*100, eer*100))

# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

#-----------------------------------------------------------

reader_test = csv.reader(open(opt.test_1, 'r'), delimiter=',')
test_set = np.array(list(reader_test)).astype('float')

# full
data = test_set[:, 0:-1]
label = test_set[:, -1]

pred = clf.predict(data)
pred_prob = clf.predict_proba(data)

acc = metrics.accuracy_score(label, pred)
fpr, tpr, thresholds = roc_curve(label, pred_prob[:,1], pos_label=1)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
roc_auc = metrics.auc(fpr, tpr)

print('Accuracy: %.2f, EER: %.2f' % (acc*100, eer*100))

#####

reader_test = csv.reader(open(opt.test_2, 'r'), delimiter=',')
test_set = np.array(list(reader_test)).astype('float')

# full
data = test_set[:, 0:-1]
label = test_set[:, -1]

pred = clf.predict(data)
pred_prob = clf.predict_proba(data)

acc = metrics.accuracy_score(label, pred)
fpr, tpr, thresholds = roc_curve(label, pred_prob[:,1], pos_label=1)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
roc_auc = metrics.auc(fpr, tpr)

print('Accuracy: %.2f, EER: %.2f' % (acc*100, eer*100))
