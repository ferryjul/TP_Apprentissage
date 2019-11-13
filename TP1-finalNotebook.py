#!/usr/bin/env python
# coding: utf-8

# # TP 1 Apprentissage Supervisé

# ## I)  Jeux de données

# ### 1) Ouverture du jeu de données MNIST et analyse de sa structure

# In[1]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')


# In[3]:


print(mnist) # Jeu de données entier


# In[10]:


print(mnist.data) # Valeurs des attributs des instances


# In[9]:


print(mnist.target) # Classes associées aux instances


# In[8]:


len(mnist.data) # Nombre d'instances


# In[11]:


print (mnist.data.shape) # (Nombre d'instances * Nombre d'attributs (ici, les pixels)) (dimensions du tableau .data)


# In[12]:


print (mnist.target.shape) # (Nombre d'instances) (dimensions du tableau .target)


# In[25]:


mnist.data[0] # Valeurs des attributs (pixels) pour la première instance du dataset => 784 valeurs


# In[14]:


mnist.data[0][1] # Valeur du second pixel pour la première image (instance) du dataset


# In[36]:


mnist.data[:,1].shape # Valeurs des seconds pixels de toutes les images du dataset => 70000 valeurs


# In[33]:


mnist.data[:100] # Valeurs des pixels pour les 100 premières images du dataset => Tableau 100*784


# ### 2) Visualisation des données

# In[44]:


import matplotlib.pyplot as plt 
# Images va contenir les images du dataset mnist, redimensionnées (coupées)
images = mnist.data.reshape((-1, 28, 28))
# Créer l'image 
plt.imshow(images[0],cmap=plt.cm.gray_r, interpolation="nearest") 
# L'afficher à l'écran
plt.show()


# ### 3) Ouverture d'autres datasets

# In[53]:


from sklearn.datasets import load_wine 
wineDataSet = load_wine()


# In[56]:


print(wineDataSet.data.shape) # 178 instances, 13 attributs


# In[57]:


print(wineDataSet.feature_names) # Noms des attributs


# In[58]:


print(wineDataSet.target_names) # Nom des classes/labels


# ## II) La méthode des k-plus proches voisins

# In[2]:


from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import numpy as np


# In[44]:


# Echantillon de données contenant 5000 instances aléatoirement choisies
data = np.random.randint(70000, size=5000)


# In[4]:


# Split dataset between training (70%) and test (30%) sets
xtrain, xtest, ytrain, ytest = train_test_split(mnist.data[data], mnist.target[data], train_size=0.8)


# In[6]:


# Nombre de voisins que va considérer notre classifieur
n_neighbors = 10
# Création du classifieur
clf = neighbors.KNeighborsClassifier(n_neighbors)
# Entrainement du classifieur
clf.fit(xtrain, ytrain)
# Evaluation du classifieur sur le training set
trainingScore = clf.score(xtrain, ytrain)
print("kNN classifier accuracy on training set : ", trainingScore)
# Evaluation du classifieur sur le test set
testScore = clf.score(xtest, ytest)
print("kNN classifier accuracy on test set : ", testScore)


# Notre classifieur a une erreur non nulle sur l'ensemble d'apprentissage. C'est normal (sinon on aurait un phénomène d'overfitting (précision très haute sur le training set mais très mauvaise précision sur des exemples inconnus : on apprend du "bruit")

# In[ ]:


outprt = "Classe de l'image 4 du test set : " + str(ytest[3]) + "; classe prédite : " + str(clf.predict(xtest[3].reshape(1,-1))) + " (proba de classification estimée : " + str(clf.predict_proba(xtest[3].reshape(1,-1))[:,5]) + ")"
print(outprt)


# In[7]:


# Etude de l'impact du nombre de voisins, sans k-fold-cross-validation
trainScores = []
testScores = []
kValues = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for k in kValues:
    # Création du classifieur
    clf = neighbors.KNeighborsClassifier(k)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
    print("k = %d, training set accuracy = %lf, test set accuracy = %lf" %(k, trainingScore, testScore))
plt.scatter(kValues, trainScores, label = "Training Set")
plt.scatter(kValues, testScores, label = "Test Set")
plt.title("Accuracy on training and test set, varying k (the number of neighbours)")
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# In[16]:


# Etude de l'impact du nombre de voisins, avec k-fold-cross-validation
from sklearn.model_selection import cross_val_score
trainScores = []
testScores = []
kValues = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for k in kValues:
    # Création du classifieur
    clf = neighbors.KNeighborsClassifier(k)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = cross_val_score(clf, xtrain, ytrain, cv=10)
    trainScores.append(np.mean(trainingScore))
    # Evaluation du classifieur sur le test set
    testScore = cross_val_score(clf, xtest, ytest, cv=10)
    testScores.append(np.mean(testScore))
    print("k = %d, training set accuracy = %lf, test set accuracy = %lf" %(k, np.mean(trainingScore), np.mean(testScore)))
plt.scatter(kValues, trainScores, label = "Training Set")
plt.scatter(kValues, testScores, label = "Test Set")
plt.title("Accuracy on training and test set, varying k (the number of neighbours)")
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# In[19]:


# Recherche du meilleur split training/test set
from sklearn.model_selection import cross_val_score
trainScores = []
testScores = []
kValues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
n_neighbors = 3 # Among the best values found previously
for k in kValues:
    xtrain, xtest, ytrain, ytest = train_test_split(mnist.data[data], mnist.target[data], train_size=k)
    # Création du classifieur
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
plt.scatter(kValues, trainScores, label = "Training Set")
plt.scatter(kValues, testScores, label = "Test Set")
plt.title("Accuracy on training and test set, varying %train/test")
plt.xlabel('%training set')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# In[56]:


# Impact de la taille du training set sur l'accuracy
from sklearn.model_selection import cross_val_score
trainScores = []
testScores = []
kValues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
xtrainT, xtest, ytrainT, ytest = train_test_split(mnist.data[data], mnist.target[data], train_size=0.8)
n_neighbors = 3 # Among the best values found previously
for k in kValues:
    xtrain, nomatter1, ytrain, nomatter2 = train_test_split(xtrainT, ytrainT, train_size=k)
    # Création du classifieur
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
plt.scatter(kValues, trainScores, label = "Training Set")
plt.scatter(kValues, testScores, label = "Test Set")
plt.title("Accuracy on training and test set, varying %train/test")
plt.xlabel('Size of the training set (%max size)')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# In[25]:


# Essai de plusieurs métriques de distance
from sklearn.model_selection import cross_val_score
trainScores = []
testScores = []
kValues = [1,2,3] # Manhattan, euclidian  and norm3 distances
n_neighbors = 3 # Among the best values found previously
for k in kValues:
    xtrain, xtest, ytrain, ytest = train_test_split(mnist.data[data], mnist.target[data], train_size=0.7)
    # Création du classifieur
    clf = neighbors.KNeighborsClassifier(n_neighbors, p=k)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    trainScores.append(trainingScore)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    testScores.append(testScore)
plt.scatter(kValues, trainScores, label = "Training Set")
plt.scatter(kValues, testScores, label = "Test Set")
plt.title("Accuracy on training and test set, varying %train/test")
plt.xlabel('distance metric')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# In[29]:

# Impact of different n_jobs values on the model training & evaluation running times
import time
rTimes = []
rETimes = []
nS = [1,2,3,4,-1]
data = np.random.randint(70000, size=500)
for n in nS:
    xtrain, xtest, ytrain, ytest = train_test_split(mnist.data[data], mnist.target[data], train_size=0.7)
    start_time = time.time()
    clf = neighbors.KNeighborsClassifier(n_neighbors, p=k, n_jobs=n)
    # Entrainement du classifieur
    clf.fit(xtrain, ytrain)
    pTime = time.time() - start_time
    rETimes.append(pTime)
    # Evaluation du classifieur sur le training set
    trainingScore = clf.score(xtrain, ytrain)
    # Evaluation du classifieur sur le test set
    testScore = clf.score(xtest, ytest)
    rTimes.append(time.time() - start_time)
plt.scatter(nS, rTimes, label = "Total")
plt.scatter(nS, rETimes, label = "Train")
plt.title("Ruuning times")
plt.xlabel('n_jobs')
plt.ylabel('time')
plt.legend(loc='lower left')
plt.autoscale(tight=True)
plt.show()


# Les kNN offrent ici de bonnes performances mais l'augmentation du nombre d'instances entraine une rapide augmentation des temps de calcul. En effet, la complexité algorithmique de la recherche des k plus proches voisins de chaque instance parmi n instances est plus que linéaire, ce qui entraine un mauvais passage à l'échelle. En outre, cette méthode a l'inconvénient d'être très sensible au bruit.
