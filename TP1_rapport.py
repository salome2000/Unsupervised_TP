#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:11:47 2024

@author: Malaurie Bernard et Salomé Marty Laurent 
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
    with open(path + data, 'r') as fichier:
        lignes = fichier.readlines()

        # Extract two columns from each line
        datanp = [line.split() for line in lignes]

    # Extract values from the merged columns
    f0 = [float(x[0]) for x in datanp]  # all elements from the first column
    f1 = [float(x[1]) for x in datanp]  # all elements from the second column

    datanp = [[float(val) for val in line.split()] for line in lignes]
    return datanp, f0, f1



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

def affichage_init (data,f0,f1):
    plt.scatter(f0,f1,s=8)
    title = "Données initiales " + data
    plt.title(title)
    plt.show()

def affichage_cluster (f0,f1,labels,k,data,runtime, datanp):
    plt.scatter(f0,f1,c=labels,s = 8)
    title = "Algorithme k-Means" + "\nTaille du dataset: " + str(len(datanp)) +"\nClusterisation du dataset: " + data + "\nLa valeur de k trouvée est " + str(k) +"\nTemps d'execution: " + str(runtime) + " ms"
    plt.title(title)
    plt.show()
    
    
def affichage_dataset_clusterised_knn(data, initial, clusters):
    """La vairiable initial est un booléen qui indique si on veut afficher les données initiales, et dendrogramme est un booléen qui indique si on veut afficher le dendrogramme."""
    #On extrait les données pour les clusteriser
    datanp,f0,f1 = extract_two_firsts_columns(data)
    if (initial == 1):
        affichage_init(data,f0,f1)
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
    runtime =  round(( tps2 - tps1 ) * 1000 ,2 )
    if(clusters == 1):
        affichage_cluster(f0,f1,labels,k,data,runtime,datanp)

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

def affichage_clusterisation(data, f0,f1,labels,k, leaves, distance_threshold, linkage, runtime, datanp):
    plt.scatter(f0, f1,c = labels, s = 8 )
    title = "Algorithme agglomératif" + "\nTaille du dataset: " + str(len(datanp)) +"\nResultat du clustering de la dataset : " + data + "\nAvec pour otpion de linkage : " + linkage + "\nEt un seuil de distance : " + str(distance_threshold) + "\nOn a " + str(k) + " clusters" +"\nEt un temps d'éxecution de : " + str(runtime) + "ms"
    plt.title(title)
    plt.show()

def affichage_silhouette_score(L,nb_iteration):
    plt.scatter(nb_iteration, L)
    title = "Algorithme agglomératif" + "\n Silhouette score"
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
    affichage_clusterisation(data, f0, f1, model.labels_, k, leaves, distance_threshold, linkage, runtime,datanp)


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
    nb_iteration = []
    for linkage in Link:
        model, k, leaves = clusterisation(datanp, distance_threshold, linkage)
        affichage_dataset_clusterised(data, distance_threshold, linkage, 0,0)
        if(distance_threshold >1):
            L.append(sklearn.metrics.silhouette_score(datanp,model.labels_))
            nb_iteration.append(linkage)
        
     
    if not L:
        L = [1]
      
    affichage_silhouette_score(L,nb_iteration)
        
    return Link[L.index(max(L))]
        


############### MAIN ################################


####################Calcul avec algo KNN##############################
#affichage_dataset_clusterised_knn("x1.txt", 1, 1)

#affichage_dataset_clusterised_knn("x2.txt", 1, 1)

#affichage_dataset_clusterised_knn("x3.txt", 1, 1)

#affichage_dataset_clusterised_knn("x4.txt", 1, 1)

#affichage_dataset_clusterised_knn("zz1.txt", 1, 1)

#affichage_dataset_clusterised_knn("zz2.txt", 1, 1)

####################Calcul avec algo agglomérative ##############################
#*******************VARIATION DE DISTANCES *********************************
#print("La meilleure distance pour le dataset x4.txt est de:", best_distance("x4.txt", "single"))

#*******************VARIATION DE LINKAGE *********************************
print("Le meilleur paramètre de linkage pour le dataset zz1.txt est: ", test_linkage ("zz1.txt", 6))

print("Le meilleur paramètre de linkage pour le dataset zz2.txt est: ", test_linkage ("zz2.txt", 1))

affichage_dataset_clusterised("x1.txt", best_distance("x1.txt","single"),"single", 0, 0)

affichage_dataset_clusterised("x2.txt", best_distance("x2.txt","single"),"single", 0, 0)
affichage_dataset_clusterised("x3.txt", best_distance("x3.txt","single"), "single", 0, 0)
affichage_dataset_clusterised("x4.txt", best_distance("x4.txt","single"), "single", 0, 0)
affichage_dataset_clusterised("zz1.txt", best_distance("zz1.txt","single"), "single", 0, 0)
affichage_dataset_clusterised("zz2.txt", best_distance("zz2.txt","single"), "single", 0, 1)


