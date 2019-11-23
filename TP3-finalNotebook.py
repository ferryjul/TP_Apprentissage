#!/usr/bin/env python
# coding: utf-8

# # TP 3 Apprentissage Supervisé

# ## Prise en main du classifieur SVM-SVC et premières observations

# In[1]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784') # Open dataset


# In[2]:


from sklearn.model_selection import train_test_split
import numpy as np
# Pour des running times moins longs :
# Echantillon de données contenant 5000 instances aléatoirement choisies
data = np.random.randint(70000, size=7000)
fact = 0.7
xtrain, xtest, ytrain, ytest = train_test_split(mnist.data[data], mnist.target[data], train_size=fact)
print(len(ytrain), " instances in train set.")
print(len(ytest), " instances in test set.")


# In[4]:


from sklearn import svm

# Création du classifieur
clf = svm.SVC(gamma='auto', kernel='linear')
# Entrainement du classifier
clf.fit(xtrain, ytrain)
# Evaluation du classifieur sur le training set
trainingScore = clf.score(xtrain, ytrain)
print("SVC classifier accuracy on training set : ", trainingScore)
# Evaluation du classifieur sur le test set
testScore = clf.score(xtest, ytest)
print("SVC classifier accuracy on test set : ", testScore)


# ## Impact du kernel sur précision et running times

# In[8]:


from sklearn import svm
import time
from sklearn.metrics import accuracy_score

testScores = []
timesL = []
trainingScores = []
times = []
kernels = ['rbf','precomputed', 'poly', 'linear', 'sigmoid']
for kern in kernels:
    start_time = time.time()
    if(kern == 'precomputed'):
        # Création du classifieur
        kernel_train = np.dot(xtrain, xtrain.T)  # linear kernel
        clf = svm.SVC(gamma='auto', kernel='precomputed') # linear
        # Entrainement du classifier
        clf.fit(kernel_train, ytrain)
        time3 = time.time() - start_time
        # Evaluation du classifieur
        kernel_test = np.dot(xtest, xtrain.T)
        y_pred = clf.predict(kernel_test)
        testScore = accuracy_score(ytest, y_pred)
        y_train_pred = clf.predict(kernel_train)
        trainingScore = accuracy_score(ytrain, y_train_pred)
    else:
        # Création du classifieur
        clf = svm.SVC(gamma='auto', kernel=kern)
        # Entrainement du classifier
        clf.fit(xtrain, ytrain)
        time3 = time.time() - start_time
        # Evaluation du classifieur
        trainingScore = clf.score(xtrain, ytrain)
        testScore = clf.score(xtest, ytest)
    print("[kernel = ", kern, "] SVC classifier accuracy on training set : ", trainingScore)
    print("[kernel = ", kern, "] SVC classifier accuracy on test set : ", testScore)
    time2 = time.time() - start_time
    times.append(time2)
    timesL.append(time3)
    testScores.append(testScore)
    trainingScores.append(trainingScore)
    print("[kernel = ", kern, "] total running time : ", time2)
    print("[kernel = ", kern, "] total training time : ", time3)


# ## Impact de C (pour différents kernels) sur précision et running times

# In[ ]:


from sklearn import svm
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


kernels = ['rbf','precomputed', 'poly', 'linear', 'sigmoid']
for kern in kernels:
    testScores = []
    timesL = []
    trainingScores = []
    times = []
    tolParams = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for cP in tolParams:
        start_time = time.time()
        if(kern == 'precomputed'):
            # Création du classifieur
            kernel_train = np.dot(xtrain, xtrain.T)  # linear kernel
            clf = svm.SVC(C=cP, gamma='auto', kernel='precomputed') # linear
            # Entrainement du classifier
            clf.fit(kernel_train, ytrain)
            time3 = time.time() - start_time
            # Evaluation du classifieur
            kernel_test = np.dot(xtest, xtrain.T)
            y_pred = clf.predict(kernel_test)
            testScore = accuracy_score(ytest, y_pred)
            y_train_pred = clf.predict(kernel_train)
            trainingScore = accuracy_score(ytrain, y_train_pred)
        else:
            # Création du classifieur
            clf = svm.SVC(C=cP, gamma='auto', kernel=kern)
            # Entrainement du classifier
            clf.fit(xtrain, ytrain)
            time3 = time.time() - start_time
            # Evaluation du classifieur
            trainingScore = clf.score(xtrain, ytrain)
            testScore = clf.score(xtest, ytest)
        time2 = time.time() - start_time
        times.append(time2)
        timesL.append(time3)
        testScores.append(testScore)
        trainingScores.append(trainingScore)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.scatter(tolParams, times, color='g')
    ax1.plot(tolParams, trainingScores, label = "Training Set")
    ax1.plot(tolParams, testScores, label = "Test Set")
    plt.title("Kernel : %s" %(kern))
    plt.xlabel('c ("cost")')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Running time (s)',  color='g')
    ax1.legend(loc='lower left')
    plt.autoscale(tight=True)
    plt.show()


# ## Calcul d'une matrice de confusion

# In[5]:


from sklearn.metrics import confusion_matrix
clf = svm.SVC(gamma='auto', kernel='poly') # best kernel found previously
# Entrainement du classifier
clf.fit(xtrain, ytrain)
cm = confusion_matrix(ytest, clf.predict(xtest))
print(cm)
# By definition a confusion matrix C
# is such that C(i,j) is equal to the 
# number of observations known to be in 
# group i but predicted to be in group j.


# In[ ]:




