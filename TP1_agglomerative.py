#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:52:28 2024

@author: Malaurie Bernard et Salomé Marty--Laurent 
"""

import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
import time
import sklearn
from sklearn import cluster 
from sklearn import metrics 
import scipy.cluster.hierarchy as shc



def extract_two_firsts_columns(data):
    path = './'
    databrut = arff . loadarff ( open ( path + data , 'r' ) )
    datanp = [[x[0],x[1]] for x in databrut[0]]
    print(datanp[0])
    
    f0 = [x[0] for x in datanp] # tous les elements de la premiere colonne
    f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne
    return datanp,f0,f1

def affichage_init (data,f0,f1):
    plt.scatter(f0,f1,s=8)
    title = "Données initiales : " + data
    plt.title(title)
    plt.show()


def affichage_dendrogrammes(data, datanp, linkage):
    linked_mat=shc.linkage(datanp,linkage)
    plt.figure(figsize=(12,12))
    shc.dendrogram(linked_mat,orientation='top',distance_sort='descending',show_leaf_counts=False)
    title = "Dendrogramme du dataset " + data + "\nAvec pour otpion de linkage : " + linkage
    plt.title(title)
    plt.show()

def clusterisation(datanp, distance_threshold ,linkage) :
    model = cluster.AgglomerativeClustering(distance_threshold=distance_threshold,linkage=linkage,n_clusters=None)
    model = model.fit(datanp)
    return model, model.n_clusters_, model.n_leaves_

def affichage_clusterisation(data, f0,f1,labels,k, leaves, distance_threshold, linkage, runtime):
    plt.scatter(f0, f1,c = labels, s = 8 )
    title = "Resultat du clustering de la dataset : " + data + "\nAvec pour otpion de linkage : " + linkage + "\nEt un seuil de distance : " + str(distance_threshold) + "\nOn a " + str(k) + " clusters" +"\nEt un temps d'éxecution de : " + str(runtime) + "ms"
    plt.title(title)
    plt.show()
    

    
def affichage_dataset_clusterised(data, distance_threshold, linkage, initial, dendrogramme) :
    """La vairiable initial est un booléen qui indique si on veut afficher les données initiales, et dendrogramme est un booléen qui indique si on veut afficher le dendrogramme."""
    #On extrait les données pour les clusteriser
    datanp,f0,f1 = extract_two_firsts_columns(data)
    if (initial == 1):
        affichage_init(data,f0,f1)
    #Dendogrammes
    if (dendrogramme == 1):
        affichage_dendrogrammes(data, datanp, linkage)
    #Clusterisation 
    tps1 = time.time()
    model, k, leaves = clusterisation(datanp, distance_threshold, linkage)
    tps2 = time.time()
    runtime = round (( tps2 - tps1 ) * 1000 , 2 )
    affichage_clusterisation(data, f0, f1, model.labels_, k, leaves, distance_threshold, linkage, runtime)


def best_distance (data, linkage) :
    datanp,f0,f1 = extract_two_firsts_columns(data)
    L = []
    #INitialisation de K_values
    model, k, leaves = clusterisation(datanp, 0, linkage)
    K_values = [k]
    for distance in range (1,11):
        model, k, leaves = clusterisation(datanp, distance, linkage)
        K_values.append(k)
        if(K_values[distance] != K_values[distance-1] and K_values[distance] > 1): #To find threshold distance
            L.append(sklearn.metrics.silhouette_score(datanp,model.labels_)) 
    
    
    if not L:
        L = [0]
        
    return L.index(max(L))+1
        
def test_linkage (data, distance_threshold) :
    Link= ["single","average", "complete", "ward"]
    datanp,f0,f1 = extract_two_firsts_columns(data)
    L = []
    for linkage in Link:
        model, k, leaves = clusterisation(datanp, distance_threshold, linkage)
        affichage_dataset_clusterised(data, distance_threshold, linkage, 0,0)
        L.append(sklearn.metrics.silhouette_score(datanp,model.labels_))
        
    return Link[L.index(max(L))]
        
    
########################################   PARTIE 3  #################################
#3.2-Datasets pour lesquels la méthode fonctionne plutot bien
#*******************DATASET 1******************************
#affichage_dataset_clusterised("pmf.arff", 10, "complete", 1, 1)

#*******************DATASET 2*********************************
#affichage_dataset_clusterised("triangle1.arff", 5, "single", 1, 1)

#*******************DATASET 3 *********************************
#affichage_dataset_clusterised("smile1.arff", 0.1, "single", 1, 1)



#*******************VARIATION DE DISTANCES *********************************
print("La meilleure distance pour le dataset triangle1.arff est de:", best_distance("triangle1.arff", "single"))

#*******************VARIATION DE LINKAGE *********************************
print("Le meilleur paramètre de linkage pour le dataset triangle1.arff est: ", test_linkage ("triangle1.arff", 5))



#3.3-Datasets pour lesquels la méthode fonctionne pas bien du tout
#*******************DATASET 1******************************
affichage_dataset_clusterised("twodiamonds.arff", 10, "complete", 0, 1)

#*******************DATASET 2*********************************
affichage_dataset_clusterised("donutcurves.arff", best_distance("donutcurves.arff", "single"), "single", 0, 1)

#*******************DATASET 3 *********************************
affichage_dataset_clusterised("impossible.arff", 2, "single", 0, 1)



