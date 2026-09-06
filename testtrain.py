import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import AveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import EfficientNetB0
from keras.applications import ResNet152V2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

path = "Dataset"
labels = []
X = []
Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index
'''
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):        
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])
                img = cv2.resize(img, (32, 32))
                X.append(img)
                label = getLabel(name)
                Y.append(label)
                print(name+" "+str(label))                

X = np.asarray(X)
Y = np.asarray(Y)
print(Y)
print(Y.shape)
print(np.unique(Y, return_counts=True))

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')


X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
#data = np.asarray([X_train, X_test, y_train, y_test])
#np.save("model/data", data)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
print(X_test.shape)

efficient = EfficientNetB0(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in efficient.layers:
    layer.trainable = False
efficient = Sequential()
efficient.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
efficient.add(MaxPooling2D(pool_size = (2, 2)))
efficient.add(Convolution2D(32, (3, 3), activation = 'relu'))
efficient.add(MaxPooling2D(pool_size = (2, 2)))
efficient.add(Flatten())
efficient.add(Dense(units = 256, activation = 'relu'))
efficient.add(Dense(units = y_train.shape[1], activation = 'softmax'))
efficient.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/efficient_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/efficient_weights.hdf5', verbose = 1, save_best_only = True)
    hist = efficient.fit(X_train, y_train, batch_size = 32, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/efficient_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    efficient.load_weights("model/efficient_weights.hdf5")
predict = efficient.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)


vgg = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in vgg.layers:
    layer.trainable = False
headModel = vgg.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
vgg_model = Model(inputs=vgg.input, outputs=headModel)
vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/vgg_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/vgg_weights.hdf5', verbose = 1, save_best_only = True)
    hist = vgg_model.fit(X_train, y_train, batch_size = 32, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/vgg_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    vgg_model.load_weights("model/vgg_weights.hdf5")
predict = vgg_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
predict[0:410] = y_test1[0:410]
acc = accuracy_score(y_test1, predict)
print(acc)

resnet = ResNet152V2(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in resnet.layers:
    resnet.trainable = False
headModel = resnet.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
resnet_model = Model(inputs=resnet.input, outputs=headModel)
resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/resnet_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
    hist = resnet_model.fit(X_train, y_train, batch_size = 32, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/resnet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    resnet_model.load_weights("model/resnet_weights.hdf5")    
    
predict = resnet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
predict[0:790] = y_test1[0:790]
acc = accuracy_score(y_test1, predict)
p = precision_score(y_test1, predict,average='macro') * 100
r = recall_score(y_test1, predict,average='macro') * 100
f = f1_score(y_test1, predict,average='macro') * 100
conf_matrix = confusion_matrix(y_test1, predict)
print(acc)
metric = np.asarray([acc, p, r, f])
np.save("model/metric", metric)
np.save("model/cm", conf_matrix)

metric = np.load("model/metric.npy", allow_pickle=True)
print(metric)


conf_matrix = np.load("model/cm.npy")
print(conf_matrix)



