#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:59:43 2024

@author: Malaurie Bernard et Salomé Marty-Laurent 
"""

import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
import time
import sklearn
from sklearn import cluster 
from sklearn import metrics 


def extract_two_firsts_columns(data):
    path = './'
    databrut = arff . loadarff ( open ( path + data , 'r' ) )
    datanp = [[x[0],x[1]] for x in databrut[0]]

    
    f0 = [x[0] for x in datanp] # tous les elements de la premiere colonne
    f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne
    return datanp,f0,f1




def find_best_k(datanp):
    L = []
    for k in range(2,11):
        model = cluster.KMeans(n_clusters=k,init='k-means++')
        model.fit(datanp)
        labels = model.labels_
        L.append(sklearn.metrics.silhouette_score(datanp,labels))
    
    return L.index(max(L))+2
        

def cluster_best_k(datanp):
    k=find_best_k(datanp)
    model = cluster.KMeans(n_clusters=k,init='k-means++')
    model.fit(datanp)

def affichage_init (f0,f1):
    plt.scatter(f0,f1,s=8)
    plt.title("Donnees initiales")
    plt.show()

def affichage_cluster (f0,f1,labels):
        plt.scatter(f0,f1,c=labels,s = 8)
        plt.title(" Donnees apres clustering Kmeans ")
        plt.show()

########################################   PARTIE 2. 2   #################################
#*********************************
#On extrait les données pour les clusteriser
datanp,f0,f1 = extract_two_firsts_columns("xclara.arff")
affichage_init(f0,f1)

print( " Appel KMeans pour une valeur fixee de k " )
tps1 = time.time()

k=find_best_k(datanp)
print("La valeur de k trouvée est",k)

##On clusterise avec le meilleur k
model = cluster.KMeans(n_clusters=k,init='k-means++')
model.fit(datanp)
tps2=time.time()
labels = model.labels_
iteration =model.n_iter_
affichage_cluster(f0,f1,labels)
print( "nb clusters = " ,k , " , nb iter = " , iteration , " , runtime = ", round(( tps2 - tps1 ) * 1000 ,2 ) ," ms " )


#*********************************
datanp,f0,f1 = extract_two_firsts_columns("twodiamonds.arff")
affichage_init (f0,f1)

# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#
print( " Appel KMeans pour une valeur fixee de k " )
tps1 = time.time()

k=find_best_k(datanp)
print("La valeur de k trouvée est",k)

##On clusterise avec le meilleur k
model = cluster.KMeans(n_clusters=k,init='k-means++')
model.fit(datanp)
tps2=time.time()
labels = model.labels_
iteration =model.n_iter_
affichage_cluster (f0,f1,labels)
print( "nb clusters = " ,k , " , nb iter = " , iteration , " , runtime = ", round(( tps2 - tps1 ) * 1000 ,2 ) ," ms " )



#*********************************
datanp,f0,f1 = extract_two_firsts_columns("pmf.arff")
affichage_init (f0,f1)

print( " Appel KMeans pour k allant de 2 à 10" )
tps1 = time.time()

k=find_best_k(datanp)
print("La valeur de k trouvée est",k)

##On clusterise avec le meilleur k
model = cluster.KMeans(n_clusters=k,init='k-means++')
model.fit(datanp)
tps2=time.time()
labels = model.labels_
iteration =model.n_iter_
affichage_cluster (f0,f1,labels)
print( "nb clusters = " ,k , " , nb iter = " , iteration , " , runtime = ", round(( tps2 - tps1 ) * 1000 ,2 ) ," ms " )


# Avec la méthode visuelle de définition des clusters et par la méthode des métriques on retouve les mêmes 
# résultats

########################################   PARTIE 2. 3   #################################

## Jeux de données pour lesquels l'algorithme K-means a des difficultés 

#*********************************
#On extrait les données pour les clusteriser
datanp,f0,f1 = extract_two_firsts_columns("smile1.arff")
affichage_init(f0,f1)

print( " Appel KMeans pour une valeur fixee de k " )
tps1 = time.time()

k=find_best_k(datanp)
print("La valeur de k trouvée est",k)

##On clusterise avec le meilleur k
model = cluster.KMeans(n_clusters=k,init='k-means++')
model.fit(datanp)
tps2=time.time()
labels = model.labels_
iteration =model.n_iter_
affichage_cluster(f0,f1,labels)
print( "nb clusters = " ,k , " , nb iter = " , iteration , " , runtime = ", round(( tps2 - tps1 ) * 1000 ,2 ) ," ms " )

#*********************************
#On extrait les données pour les clusteriser
datanp,f0,f1 = extract_two_firsts_columns("rings.arff")
affichage_init(f0,f1)

print( " Appel KMeans pour une valeur fixee de k " )
tps1 = time.time()

k=find_best_k(datanp)
print("La valeur de k trouvée est",k)

##On clusterise avec le meilleur k
model = cluster.KMeans(n_clusters=k,init='k-means++')
model.fit(datanp)
tps2=time.time()
labels = model.labels_
iteration =model.n_iter_
affichage_cluster(f0,f1,labels)
print( "nb clusters = " ,k , " , nb iter = " , iteration , " , runtime = ", round(( tps2 - tps1 ) * 1000 ,2 ) ," ms " )

#*********************************
#On extrait les données pour les clusteriser
datanp,f0,f1 = extract_two_firsts_columns("golfball.arff")
affichage_init(f0,f1)

print( " Appel KMeans pour une valeur fixee de k " )
tps1 = time.time()

k=find_best_k(datanp)
print("La valeur de k trouvée est",k)

##On clusterise avec le meilleur k
model = cluster.KMeans(n_clusters=k,init='k-means++')
model.fit(datanp)
tps2=time.time()
labels = model.labels_
iteration =model.n_iter_
affichage_cluster(f0,f1,labels)
print( "nb clusters = " ,k , " , nb iter = " , iteration , " , runtime = ", round(( tps2 - tps1 ) * 1000 ,2 ) ," ms " )






