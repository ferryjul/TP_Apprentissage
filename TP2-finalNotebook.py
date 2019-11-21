#!/usr/bin/env python
# coding: utf-8

# # TP 2 Apprentissage Supervisé

# ## Prise en main du classifieur ANN et premières observations

# In[3]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784') # Open dataset


# In[4]:


from sklearn.model_selection import train_test_split
# 49,000 instances in training set, the remaining in test set
fact = 49000/len(mnist.data)
xtrain, xtest, ytrain, ytest = train_test_split(mnist.data, mnist.target, train_size=fact)


# In[5]:


from sklearn import neural_network

# Création du classifieur
clf = neural_network.MLPClassifier(hidden_layer_sizes = (50))
# Entrainement du classifier
clf.fit(xtrain, ytrain)
# Evaluation du classifieur sur le training set
trainingScore = clf.score(xtrain, ytrain)
print("ANN classifier accuracy on training set : ", trainingScore)
# Evaluation du classifieur sur le test set
testScore = clf.score(xtest, ytest)
print("ANN classifier accuracy on test set : ", testScore)


# In[6]:


# Affichage de la classe et de la prédiction de l'image 4
outprt = "Classe de l'image 4 du test set : " + str(ytest[3]) + "; classe prédite : " + str(clf.predict(xtest[3].reshape(1,-1))) + " (proba de classification estimée : " + str(clf.predict_proba(xtest[3].reshape(1,-1))[:,5]) + ")"
print(outprt)


# In[7]:


# Calcul de la précision avec le package metrics.precision_score de sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import recall_score
# Avec d'autres paramètres d'appel on pourrait calculer d'autres métriques (recall...etc)
print("precision score = ", precision_score(ytest, clf.predict(xtest),average='micro')) # Le score devrait être le même
print("recall score = ",recall_score(ytest, clf.predict(xtest),average='weighted'))
print("zero one loss score = ",zero_one_loss(ytest, clf.predict(xtest),normalize=False)) 


# ## Impact du nombre de couches (épaisseur constante) sur la précision

# In[20]:


# Code pour évaluation des modèles de profondeur différente (largeur fixe)

import numpy as np
import matplotlib.pyplot as plt 
import time

test_layers = [2,10,20,50,100]
trainScores = []
testScores = []
times = []
nbByLayer = 50
for i in test_layers:
    layers_param0 = (np.ones((i,), dtype=int)*nbByLayer).tolist()
    layers_param = tuple(layers_param0)
    start_time = time.time()
    #print(layers_param, "(", len(layers_param), " elements)")
    # Création du classifieur
    clf = neural_network.MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes = layers_param)
    # Entrainement du classifier
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    #print("[", i, " hidden layers] ANN classifier accuracy on training set : ", trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
    #print("[", i, " hidden layers] ANN classifier accuracy on test set : ", testScore)
    times.append(time.time() - start_time)


# ## Impact du nombre de couches (épaisseur variable) sur la précision

# In[ ]:


# Code pour évaluation des modèles de profondeur différente (largeur variable)
import numpy as np
import matplotlib.pyplot as plt 
import time

test_layers = [2,10,20,50,100]
trainScores = []
testScores = []
times = []
nbByLayer = 60
nbFinal = 10
for i in test_layers:
    layers_param0 = (np.ones((i,), dtype=int)*nbByLayer).tolist()
    nbMinus = 0
    for j in range(len(layers_param0)):
        layers_param0[j] = int(layers_param0[j] - nbMinus)
        nbMinus = nbMinus + ((nbByLayer-nbFinal)/(i-1))
    layers_param = tuple(layers_param0)
    #print("->", layers_param)
    start_time = time.time()
    #print(layers_param, "(", len(layers_param), " elements)")
    # Création du classifieur
    clf = neural_network.MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes = layers_param)
    # Entrainement du classifier
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    #print("[", i, " hidden layers] ANN classifier accuracy on training set : ", trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
    #print("[", i, " hidden layers] ANN classifier accuracy on test set : ", testScore)
    times.append(time.time() - start_time)


# ## Code pour construire un plot à partir d'un des scénarios précédents

# In[26]:


# Code pour plot sur l'impact de la profondeur des classifieurs 
# (Code unique, à appeler après l'un ou l'autre des codes 
# (largeur fixe ou variable))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.scatter(test_layers, times, color='g')
ax1.plot(test_layers, trainScores, label = "Training Set")
ax1.plot(test_layers, testScores, label = "Test Set")
plt.title("Accuracy on training and test set, varying #layers (one layer = %d neurons)" %(nbByLayer))
plt.xlabel('Number of layers in ANN')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Running time (s)',  color='g')
ax1.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# ## Comparaison des différents solveurs proposés

# In[6]:


# On se place dans le scénario le plus favorable trouvé précédemment : 
# On fait varier la profondeur du réseau, avec une largeur décroissante
# (de l'entrée vers la sortie)
# Et on teste pour les 3 solveurs proposés

import numpy as np
import matplotlib.pyplot as plt 
import time
from sklearn import neural_network

for solv in ['sgd', 'adam', 'lbfgs']:
    test_layers = [2,10,20,50,100]
    trainScores = []
    testScores = []
    times = []
    nbByLayer = 60
    nbFinal = 10
    for i in test_layers:
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
        clf = neural_network.MLPClassifier(solver=solv, alpha=0.0001, hidden_layer_sizes = layers_param)
        # Entrainement du classifier
        clf.fit(xtrain, ytrain)
        # Evaluation du classifieur sur le training set
        trainingScore = clf.score(xtrain, ytrain)
        trainScores.append(trainingScore)
        #print("[", i, " hidden layers] ANN classifier accuracy on training set : ", trainingScore)
        # Evaluation du classifieur sur le test set
        testScore = clf.score(xtest, ytest)
        testScores.append(testScore)
        #print("[", i, " hidden layers] ANN classifier accuracy on test set : ", testScore)
        times.append(time.time() - start_time)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.scatter(test_layers, times, color='g')
    ax1.plot(test_layers, trainScores, label = "Training Set")
    ax1.plot(test_layers, testScores, label = "Test Set")
    plt.title("Accuracy on training and test set, varying #layers (one layer = %d neurons)" %(nbByLayer))
    plt.xlabel('Number of layers in ANN')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Running time (s)',  color='g')
    ax1.legend(loc='lower left')
    plt.autoscale(tight=True)
    plt.show()


# ## Comparaison des différentes fonctions d'activation proposées

# In[7]:


# On se place dans le scénario le plus favorable trouvé précédemment : 
# On fait varier la profondeur du réseau, avec une largeur décroissante
# (de l'entrée vers la sortie)
# Et on teste pour les 4 fonctions d'activation proposées
# Le solveur est celui par défaut

import numpy as np
import matplotlib.pyplot as plt 
import time
from sklearn import neural_network

for solv in ['identity', 'logistic', 'tanh', 'relu']:
    test_layers = [2,10,20,50,100]
    trainScores = []
    testScores = []
    times = []
    nbByLayer = 60
    nbFinal = 10
    for i in test_layers:
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
        clf = neural_network.MLPClassifier(activation=solv, alpha=0.0001, hidden_layer_sizes = layers_param)
        # Entrainement du classifier
        clf.fit(xtrain, ytrain)
        # Evaluation du classifieur sur le training set
        trainingScore = clf.score(xtrain, ytrain)
        trainScores.append(trainingScore)
        #print("[", i, " hidden layers] ANN classifier accuracy on training set : ", trainingScore)
        # Evaluation du classifieur sur le test set
        testScore = clf.score(xtest, ytest)
        testScores.append(testScore)
        #print("[", i, " hidden layers] ANN classifier accuracy on test set : ", testScore)
        times.append(time.time() - start_time)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.scatter(test_layers, times, color='g')
    ax1.plot(test_layers, trainScores, label = "Training Set")
    ax1.plot(test_layers, testScores, label = "Test Set")
    plt.title("Activation function : %s" %(solv))
    plt.xlabel('Number of layers in ANN')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Running time (s)',  color='g')
    ax1.legend(loc='lower left')
    plt.autoscale(tight=True)
    plt.show()


# ## Comparaison de différentes valeurs de la régularisation L2

# In[17]:


# On se place dans le scénario le plus favorable trouvé précédemment : 
# Le solveur est adam, la profondeur est de 20 couches 
# avec une largeur décroissante linéairement allant de 60 à 10
# (de l'entrée vers la sortie)
# Le solveur est celui par défaut (adam)
# La fonction d'activation est celle par défaut (relu)

import numpy as np
import matplotlib.pyplot as plt 
import time
from sklearn import neural_network

alphas = [0.00001,0.0001,0.001,0.01]
trainScores = []
testScores = []
times = []
for alph in alphas:
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
    clf = neural_network.MLPClassifier(alpha=alph, hidden_layer_sizes = layers_param)
    # Entrainement du classifier
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    #print("[", i, " hidden layers] ANN classifier accuracy on training set : ", trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
    #print("[", i, " hidden layers] ANN classifier accuracy on test set : ", testScore)
    times.append(time.time() - start_time)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.scatter(alphas, times, color='g')
ax1.plot(alphas, trainScores, label = "Training Set")
ax1.plot(alphas, testScores, label = "Test Set")
plt.title("Accuracy on training and test set, varying alpha")
plt.xlabel('Alpha (L2 regularization)')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Running time (s)',  color='g')
ax1.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# In[ ]:




