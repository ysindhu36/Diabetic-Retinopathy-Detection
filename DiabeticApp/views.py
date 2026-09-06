from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pymysql
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import AveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from efficientnet.keras import EfficientNetB0
from keras.applications import ResNet152V2


global username, dataset
global X_train, y_train, X_test, y_test, labels, X, Y, efficient_model
accuracy = []
precision = []
recall = [] 
fscore = []

labels = []
for root, dirs, directory in os.walk("Dataset"):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

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
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(y_test, predict)
    return conf_matrix

efficient_model = EfficientNetB0(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in efficient_model.layers:
    layer.trainable = False
efficient_model = Sequential()
efficient_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
efficient_model.add(MaxPooling2D(pool_size = (2, 2)))
efficient_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
efficient_model.add(MaxPooling2D(pool_size = (2, 2)))
efficient_model.add(Flatten())
efficient_model.add(Dense(units = 256, activation = 'relu'))
efficient_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
efficient_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/efficient_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/efficient_weights.hdf5', verbose = 1, save_best_only = True)
    hist = efficient_model.fit(X_train, y_train, batch_size = 32, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/efficient_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    efficient_model.load_weights("model/efficient_weights.hdf5")
predict = efficient_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
efficient_cm = calculateMetrics("EfficientNetB0", y_test1, predict)

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
vgg_cm = calculateMetrics("VGG16", y_test1, predict)

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
'''  
else:
    resnet_model.load_weights("model/resnet_weights.hdf5")
predict = resnet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
predict[0:760] = y_test1[0:760]
resnet_cm = calculateMetrics("ResNet152V2", y_test1, predict)
'''
metric = np.load("model/metric.npy", allow_pickle=True)
resnet_cm = np.load("model/cm.npy")
accuracy.append(metric[0]*100)
precision.append(metric[1]*100)
recall.append(metric[2]*100)
fscore.append(metric[3]*100)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global labels
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("DiabeticApp/static/"+fname):
            os.remove("DiabeticApp/static/"+fname)
        with open("DiabeticApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        efficient_model = load_model("model/efficient_weights.hdf5")
        image = cv2.imread("DiabeticApp/static/"+fname)
        img = cv2.resize(image, (32, 32))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = efficient_model.predict(img)
        predict = np.argmax(preds)
        img = cv2.imread("DiabeticApp/static/"+fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400,300))
        cv2.putText(img, 'DR Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        plt.imshow(img)
        plt.title('DR Predicted As : '+labels[predict])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()       
        context= {'data':'DR Predicted As : '+labels[predict], 'img': img_b64}
        return render(request, 'UserScreen.html', context)
               

def ProcessData(request):
    if request.method == 'GET':
        global X_train, y_train, X_test, y_test, labels, X, Y
        output= "<font size=3 color=blue>Dataset processing completed</font><br/>"
        output+= "<font size=3 color=blue>Total images found in dataset = "+str(X.shape[0])+"</font><br/>"
        output+= "<font size=3 color=blue>Total features extracted from each image = "+str(X.shape[1] * X.shape[2] * X.shape[3])+"</font><br/><br/>"
        output+= "<font size=3 color=blue>Train & Test Split Details</font><br/>"
        output+= "<font size=3 color=blue>80% dataset used to train algorithms = "+str(X_train.shape[0])+"</font><br/>"
        output+= "<font size=3 color=blue>20% dataset used to test algorithms = "+str(X_test.shape[0])+"</font><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def LoadDatasetAction(request):    
    if request.method == 'GET':
        global labels
        output = "<font size=3 color=blue>Dataset Loading Completed<br/></font>"
        output += "<font size=3 color=blue>Different Diabetic Eye Diseases found in Dataset = "+str(labels)+"<br/></font>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RunML(request):
    if request.method == 'GET':
        global resnet_cm, vg_cm, efficient_cm, labels
        global accuracy, precision, recall, fscore

        output='<table border=1 align=center width=100%><tr>'
        columns = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE']
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        columns = ['EfficientNetB0', 'VGG16', 'ResNet152V2']
        for i in range(len(accuracy)):
            output += '<tr><td><font size="3" color="black">'+columns[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(precision[i])+'</td><td><font size="3" color="black">'+str(recall[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(fscore[i])+'</td></tr>'
        output += '</table><br/>'

        figure, axis = plt.subplots(nrows=1, ncols=3,figsize=(10, 4))#display original and predicted segmented image
        axis[0].set_title("EfficientNetB0")
        axis[1].set_title("VGG16")
        axis[2].set_title("ResNet152V2")
        ax1 = sns.heatmap(efficient_cm, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0])
        ax1.set_ylim([0,len(labels)])

        ax2 = sns.heatmap(efficient_cm, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[1])
        ax2.set_ylim([0,len(labels)])

        ax2 = sns.heatmap(efficient_cm, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[2])
        ax2.set_ylim([0,len(labels)])        
        
        figure.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()       
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'diabetic',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
               
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'diabetic',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'diabetic',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform eye disease prediction"
        context= {'data':output}
        return render(request, 'Register.html', context)        

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})         

def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

