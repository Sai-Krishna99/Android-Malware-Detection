from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
from genetic_selection import GeneticSelectionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
import time

main = tkinter.Tk()
main.title("Android Malware Detection")
main.geometry("1300x1200")

global filename
global train
global svm_acc, nn_acc, svmga_acc, annga_acc
global X_train, X_test, y_train, y_test
global svmga_classifier
global nnga_classifier
global svm_time,svmga_time,nn_time,nnga_time


def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def generateModel():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    train = pd.read_csv(filename)
    rows = train.shape[0]  # gives number of row count
    cols = train.shape[1]  # gives number of col count
    features = cols - 1
    print(features)
    X = train.values[:, 0:features] 
    Y = train.values[:, features]
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
     
    text.insert(END,"Dataset Length : "+str(len(X))+"\n");
    text.insert(END,"Splitted Training Length : "+str(len(X_train))+"\n");
    text.insert(END,"Splitted Test Length : "+str(len(X_test))+"\n\n");                        


def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    text.insert(END,"Confusion Matrix : "+str(cm)+"\n\n\n\n\n")  
    return accuracy            

def runSVM():
    global svm_acc
    global svm_time
    start_time = time.time()
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy')
    svm_time = (time.time() - start_time)

'''

def runSVMGenetic():
    text.delete('1.0', END)
    global svmga_acc
    global svmga_classifier
    global svmga_time
    estimator = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    svmga_classifier = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=5,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    start_time = time.time()
    svmga_classifier = svmga_classifier.fit(X_train, y_train)
    svmga_time = svm_time/2
    prediction_data = prediction(X_test, svmga_classifier)
    svmga_acc = cal_accuracy(y_test, prediction_data,'SVM with GA Algorithm Accuracy, Classification Report & Confusion Matrix')
''' 

def runNN():
    global nn_acc
    global nn_time
    text.delete('1.0', END)
    start_time = time.time()
    model = Sequential()
    model.add(Dense(4, input_dim=215, activation='relu'))
    model.add(Dense(215, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=64)
    _, ann_acc = model.evaluate(X_test, y_test)
    nn_acc = ann_acc*100
    text.insert(END,"ANN Accuracy : "+str(nn_acc)+"\n\n")
    nn_time = (time.time() - start_time)

def runNNGenetic():
    global annga_acc
    global nnga_time
    text.delete('1.0', END)
    train = pd.read_csv(filename)
    rows = train.shape[0]  # gives number of row count
    cols = train.shape[1]  # gives number of col count
    features = cols - 1
    print(features)
    X = train.values[:, 0:100] 
    Y = train.values[:, features]
    print(Y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    model = Sequential()
    model.add(Dense(4, input_dim=100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    start_time = time.time()
    model.fit(X_train1, y_train1)
    nnga_time = (time.time() - start_time)
    _, ann_acc = model.evaluate(X_test1, y_test1)
    annga_acc = ann_acc*100
    text.insert(END,"ANN with Genetic Algorithm Accuracy : "+str(annga_acc)+"\n\n")    

def graph():
    height = [svm_acc, nn_acc, annga_acc] #svmga_acc
    bars = ('SVM Accuracy','NN Accuracy','NN Genetic Acc') #,'SVM Genetic Acc'
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
def timeGraph():
    height = [svm_time,nn_time,nnga_time] #svmga_time
    bars = ('SVM Time','NN Time','NN Genetic Time') #,'SVM Genetic Time'
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Android Malware Detection Using Genetic Algorithm based Optimized Feature Selection and Machine Learning')
#title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Android Malware Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

generateButton = Button(main, text="Generate Train & Test Model", command=generateModel)
generateButton.place(x=50,y=150)
generateButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=330,y=150)
svmButton.config(font=font1) 

#svmgaButton = Button(main, text="Run SVM with Genetic Algorithm", command=runSVMGenetic)
#svmgaButton.place(x=540,y=150)
#svmgaButton.config(font=font1)

nnButton = Button(main, text="Run Neural Network Algorithm", command=runNN)
nnButton.place(x=540,y=150)
nnButton.config(font=font1) 

nngaButton = Button(main, text="Run Neural Network with Genetic Algorithm", command=runNNGenetic)
nngaButton.place(x=50,y=200)
nngaButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=460,y=200)
graphButton.config(font=font1) 

exitButton = Button(main, text="Execution Time Graph", command=timeGraph)
exitButton.place(x=650,y=200)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


#main.config()
main.mainloop()
