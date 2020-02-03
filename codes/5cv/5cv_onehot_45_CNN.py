import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
from itertools import cycle
from scipy import interp
import tables as tb
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import pickle


# Training info
batch_size = 128
epochs = 100

## Data Reading
po = tb.open_file("Nm_seq_pos_45_onehot.h5", 'r')
da_ta6 = po.root.data
ne = tb.open_file("Nm_seq_neg_45_onehot.h5", 'r')
da_ta7 = ne.root.data


################################################ Data Preprocessing
## input shape
img_cols = 4
img_rows = 45

length_of_pos = len(da_ta6)
length_of_neg = len(da_ta7)

## number of samples
no_of_pos_sample = len(da_ta6)//img_rows    
no_of_neg_sample = len(da_ta7)//img_rows   


# create positive and negative input arrays
x_pos = np.zeros((no_of_pos_sample, img_rows, img_cols))
x_neg = np.zeros((no_of_neg_sample, img_rows, img_cols))

for i in range(no_of_pos_sample):
    x_pos1 = np.concatenate([da_ta6[i*img_rows:(i + 1)*img_rows]])
    x_pos[i][:][:] = x_pos1

for j in range(no_of_neg_sample):
    x_neg1 = np.concatenate([da_ta7[j*img_rows:(j + 1)*img_rows]])
    x_neg[j][:][:] = x_neg1

## labels
positive_labels = [1]*len(x_pos)     # Positive is one in one_hot encoding
negative_labels = [0]*len(x_neg)

## five fold 
no_sam_neg_cv = 5*len(x_pos)
input_Xs = np.concatenate([x_pos, x_neg[:no_sam_neg_cv]])
y_s = np.concatenate([positive_labels, negative_labels[:no_sam_neg_cv]], 0)

po.close()
ne.close()


seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []

integer_encoded = y_s.reshape(len(y_), 1)
input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
input_Xs = input_Xs.astype('float32')


y_true_label = []
y_prop_all = []
i =0
f = plt.figure()

for train, test in kfold.split(input_Xs, y_s):   

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_s)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    num_classes = len(onehot_encoded[0])


    model = Sequential()
    ## *********** First layer Conv
    model.add(Conv2D(256, kernel_size=(8, 4), strides=(1, 1), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(1, 4))
    model.add(Flatten())



    ## ********* Classification layer
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adagrad',
                  metrics=['binary_accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts', monitor='val_loss', verbose=0, save_best_only=True,
                               save_weights_only=False, mode='min', period=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                       epsilon=1e-4, mode='min')
    history = model.fit(input_Xs[train], onehot_encoded[train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0, callbacks = [earlyStopping, mcp_save, reduce_lr_loss],
                        validation_data=(input_Xs[test], onehot_encoded[test]))
    y_prop = model.predict(input_Xs[test])

    y_true_label.append(y_s[test])
    y_prop_all.append(y_prop)
    i = i + 1
    print('Fold',i)
y_real = np.concatenate(y_true_label)
y_proba = np.concatenate(y_prop_all)

f = open('5cv_onehot_PR_45_CNN.pckl', 'wb')
pickle.dump([y_real, y_proba], f)
f.close()


precision, recall, _ = precision_recall_curve(y_real, y_proba)
average_precision = average_precision_score(y_real, y_proba)
lab = 'Overall AUPRC=%.4f' % (auc(recall, precision))
plt.step(recall, precision, label=lab, lw=2, color='black')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left', fontsize='small')
plt.tight_layout()
plt.show()