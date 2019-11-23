#!/usr/bin/env python
# coding: utf-8

# ## Comparaison des 3 types de modèles étudiés précédemment

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn import svm
from sklearn.model_selection import KFold
import time

mnist = fetch_openml('mnist_784')


# In[2]:


'''data = np.random.randint(70000, size=1000)
Xdata = mnist.data[data]
Ydata = mnist.target[data]'''

Xdata = mnist.data
Ydata = mnist.target


# In[ ]:


kf = KFold(n_splits=10,shuffle=True)
trainScores = []
testScores = []
timesL = []
timesP = []
i = 1
for train_index, test_index in kf.split(Xdata):
    print("loop ", i)
    #print("TRAIN:", train_index, "TEST:", test_index)
    xtrain, xtest = Xdata[train_index], Xdata[test_index]
    ytrain, ytest = Ydata[train_index], Ydata[test_index]
    start_time = time.time()
    # Création du classifieur
    clf = neighbors.KNeighborsClassifier(3, p=2, n_jobs=-1)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    time2 = time.time()
    timesL.append(time2 - start_time)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
    timesP.append(time.time() - time2)
    i = i + 1
'''print("acc list train : ", trainScores)
print("acc list test : ", testScores)
print("timesL list : ", timesL)
print("timesP list : ", timesP)'''
print("Model KNN, accuracy on training set : ", np.mean(trainScores), "accuracy on test set : ", np.mean(testScores))
print("Model KNN, training time : ", np.mean(timesL), ", prediction time : ", np.mean(timesP))


# In[ ]:


kf = KFold(n_splits=10,shuffle=True)
trainScores = []
testScores = []
timesL = []
timesP = []
i = 1
for train_index, test_index in kf.split(Xdata):
    print("loop ", i)
    #print("TRAIN:", train_index, "TEST:", test_index)
    xtrain, xtest = Xdata[train_index], Xdata[test_index]
    ytrain, ytest = Ydata[train_index], Ydata[test_index]
    start_time = time.time()
    # Création du classifieur
    i = 20
    nbByLayer = 60
    nbFinal = 10
    layers_param0 = (np.ones((i,), dtype=int)*nbByLayer).tolist()
    nbMinus = 0
    for j in range(len(layers_param0)):
        layers_param0[j] = int(layers_param0[j] - nbMinus)
        nbMinus = nbMinus + ((nbByLayer-nbFinal)/(i-1))
    layers_param = tuple(layers_param0)
    print("->", layers_param)
    start_time = time.time()
    #print(layers_param, "(", len(layers_param), " elements)")
    # Création du classifieur
    clf = neural_network.MLPClassifier(hidden_layer_sizes = layers_param)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    time2 = time.time()
    timesL.append(time2 - start_time)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
    timesP.append(time.time() - time2)
    i = i + 1
'''print("acc list train : ", trainScores)
print("acc list test : ", testScores)
print("timesL list : ", timesL)
print("timesP list : ", timesP)'''
print("Model ANN, accuracy on training set : ", np.mean(trainScores), "accuracy on test set : ", np.mean(testScores))
print("Model ANN, training time : ", np.mean(timesL), ", prediction time : ", np.mean(timesP))


# In[ ]:


kf = KFold(n_splits=10,shuffle=True)
trainScores = []
testScores = []
timesL = []
timesP = []
i = 1
for train_index, test_index in kf.split(Xdata):
    print("loop ", i)
    #print("TRAIN:", train_index, "TEST:", test_index)
    xtrain, xtest = Xdata[train_index], Xdata[test_index]
    ytrain, ytest = Ydata[train_index], Ydata[test_index]
    start_time = time.time()
    # Création du classifieur
    clf = svm.SVC(kernel='poly', gamma='auto', C=0.3)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    time2 = time.time()
    timesL.append(time2 - start_time)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
    timesP.append(time.time() - time2)
    i = i + 1
'''print("acc list train : ", trainScores)
print("acc list test : ", testScores)
print("timesL list : ", timesL)
print("timesP list : ", timesP)'''
print("Model SVC, accuracy on training set : ", np.mean(trainScores), "accuracy on test set : ", np.mean(testScores))
print("Model SVC, training time : ", np.mean(timesL), ", prediction time : ", np.mean(timesP))


# In[ ]:




