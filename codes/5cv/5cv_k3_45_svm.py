from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import pickle
from sklearn.model_selection import KFold
from sklearn.svm import SVC



h1 = tb.open_file("Nm_seq_pos_45_k3.h5", 'r')
pos_all = h1.root.data

h2 = tb.open_file("Nm_seq_neg_45_k3.h5", 'r')
neg_all = h2.root.data


one_hot = 4
k_mer_len = 100   # one hot encoding is 4
length_seq = 45  # Sequence length
k_mer = 3
len_seq_all_ways = length_seq/k_mer


#vec_length = one_hot*length_seq # one hot encoding
vec_length = k_mer_len*len_seq_all_ways # dna2vec



pos_length_all = len(pos_all)//len_seq_all_ways  # dna2vec
pos_length = pos_length_all*len_seq_all_ways

neg_length_all = len(neg_all)//len_seq_all_ways  # dna2vec
neg_length = neg_length_all*len_seq_all_ways

Xpos = np.zeros((int(pos_length_all), int(vec_length)))
Xneg = np.zeros((int(neg_length_all), int(vec_length)))

pos_length_all = int(pos_length_all)
neg_length_all = int(neg_length_all)

for i in range(pos_length_all):
    Xpos1 = np.concatenate([pos_all[i*len_seq_all_ways:(i + 1)*len_seq_all_ways]])
    Xpos1 = Xpos1.ravel()
    Xpos[i][:] = Xpos1

for j in range(neg_length_all):
    Xneg1 = np.concatenate([neg_all[((j)*len_seq_all_ways):((j + 1)*len_seq_all_ways)]])
    Xneg1 = Xneg1.ravel()
    Xneg[j][:] = Xneg1


positive_labels = [1]*len(Xpos)     # Positive is one in one_hot encoding
negative_labels = [0]*len(Xneg)


no_sam_neg_cv = 5*len(Xpos)
X = np.concatenate([Xpos, Xneg[:no_sam_neg_cv]])
y = np.concatenate([positive_labels, negative_labels[:no_sam_neg_cv]], 0)

h1.close()
h2.close()

k_fold = KFold(n_splits=5, shuffle=True, random_state=12345)
predictor = SVC(kernel='rbf', C=1.0, probability=True, random_state=12345)

yreal = []
yproba = []


for i, (train_index, test_index) in enumerate(k_fold.split(X)):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    print('Fold %d' % (i + 1))
    print()
    print('Training model')
    predictor.fit(Xtrain, ytrain)
    pred_proba = predictor.predict_proba(Xtest)
    precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
    lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    plt.step(recall, precision, label=lab)

    yreal.append(ytest)
    yproba.append(pred_proba[:,1])

yreal = np.concatenate(yreal)
yproba = np.concatenate(yproba)

f = open('5cv_k3_PR_45_svm.pckl', 'wb')
pickle.dump([yreal, yproba], f)
f.close()

precision, recall, _ = precision_recall_curve(yreal, yproba)
average_precision = average_precision_score(yreal, yproba)

lab = 'Overall AUPR=%.4f' % (auc(recall, precision))

plt.step(recall, precision, label=lab, lw=2, color='black')
plt.title('Precision Recall of SVM for k3 45')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left', fontsize='small')
plt.tight_layout()
plt.show()