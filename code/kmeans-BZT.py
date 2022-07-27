from dis import dis
from math import dist
import numpy as np
import os
import csv
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
import time

# =============================================================================
# Program:        K-means clustering algorithm
# Author:         Adriana Ladera
# Description:    This is an unsupervised learning program 
#                 that utilizes the k-means clustering algorithm
#                 to have the computer detect features in data without
#                 human input other than a k-value, which dictates
#                 how many clusters the algorithm should expect to find
#                 features for. It accepts data as an M x N array, where M is 
#                 the number of samples and N is the dimension, i.e. variables
#                 of each sample.
# K-MEANS CLUSTERING DOCUMENTATION:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# PRINCIPLE COMPONENT ANALYSIS DOCUMENTATION:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# =============================================================================

# =============================================================================
#  FUNCTIONS                    
# =============================================================================

def read_file(filename):
    # with open(filename, newline = '') as file:
    #     df = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
    df = pd.read_csv(filename, index_col=0)
    lookup = pd.DataFrame()
    lookup["temperature"] = df["temperature"]
    lookup["cf_pattern"] = df["cf_pattern"]
    df = df.drop('temperature',  axis=1)
    df = df.drop('cf_pattern',  axis=1)
    return lookup, df

# finds number of principle components and saves cumulative explained variance
def get_pc(data, lookup, solver, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if solver == "arpack":
        n_comp = len(data)-1
    else:
        n_comp = len(data)
    pc_anal = PCA(n_components = n_comp, svd_solver=solver)
    pc_anal.fit_transform(data)
    x, y = [], []
    num_components = 0
    for i, cum in zip(range(1, n_comp+1), np.cumsum(pc_anal.explained_variance_ratio_ * 100)):
        x.append(i)
        y.append(cum)
    for i, cum in zip(range(1, n_comp+1), np.cumsum(pc_anal.explained_variance_ratio_ * 100)):
        if cum >= 99:
            num_components = i
            print("principle components: {}".format(num_components))
            break

    lookup["PCA-x"] = x
    lookup["PCA-y"] = y

    return lookup, num_components

# transforms the data using principle component analysis (PCA) and plotting PCA data
def kmeans(data, lookup, solver, n_comp, k_range, n_iter, n_tol, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    shape = data.shape
    print("data shape before PCA (samples, variables): ({}, {})".format(shape[0], shape[1]))
    pca = PCA(n_components = n_comp, svd_solver=solver)
    data_pca = pca.fit_transform(data)
    print("data shape after PCA (samples, variables):{}".format(data_pca.shape))

    # making predictions with k-means
    centroid_list, distortions, predictions = [], [], []
    silhouettes = []
    inertia = []
    choosing_k = pd.DataFrame()

    start_t = time.time()
    for k in k_range:
        km = KMeans(
        n_clusters = k, init = 'random',
        n_init = 10, max_iter = n_iter, 
        tol = n_tol, random_state = 0)
        
        pred = km.fit_predict(data_pca) #algorithm predictions using k-means
        lookup["k{}, pred".format(k)] = pred
        predictions.append(pred)

        np.unique(pred) # cluster identifiers, an arbitrary int label
            
        centroids = km.cluster_centers_ # centroid is the center of a cluster
        centroid_list.append(centroids)
        
        labels = km.labels_
        silhouettes.append(silhouette_score(data_pca, labels, metric = 'euclidean'))

        inertia.append(km.inertia_)

        distortions.append(sum(np.min(cdist(data_pca, km.cluster_centers_, 'euclidean'), axis = 1)) / data_pca.shape[0])

        centroids_x = [0] * len(data_pca)
        centroids_y = [0] * len(data_pca)
        for a, b, i in zip(centroids[:,0] , centroids[:,1], range(len(centroids[:,0]))):
            centroids_x[i] = a
            centroids_y[i] = b

        lookup["k{}, x".format(k)] = data_pca[:, 0]
        lookup["k{}, y".format(k)] = data_pca[:, 1]
        lookup["k{}, centx".format(k)] = centroids_x
        lookup["k{}, centy".format(k)] = centroids_y

        end_t = time.time()
        time_k = end_t - start_t
        minnies = int(time_k / 60)
        secs = time_k % 60
        print("\nTime elapsed for k = {}:".format(k), minnies, "min, {:.5f} sec\n".format(k, secs))

    choosing_k["k"] = k_range
    choosing_k["distortions"] = distortions
    choosing_k["inertia"] = inertia
    choosing_k["silhouettes"] = silhouettes

    lookup.to_csv("{}kmeans_results.csv".format(save_dir))
    choosing_k.to_csv("{}k_value_selection.csv".format(save_dir))
    
# =============================================================================
# MAIN                    
# =============================================================================

# list of concentrations, change according to temperature folders, electric field folders, etc.
concentrations = ["0.05", "0.15", "0.25"]
start_t = time.time()
for conc in concentrations:
    swag = "data_tables/BZT_C-{}_10K-450K_data.csv".format(conc)
    n_iter = 1600 # the maximum iterations per a single run of the algorithm
    n_tol = 1e-10 # tolerance limit (error) 
    k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10] # k = 2 to k = 10
    solver = "full" # solver can be "full", "randomized", "auto", or "arpack"

    save = os.getcwd() + '/' + conc + '/' #change depending on temp range
    if not os.path.exists(save):
        os.makedirs(save)

    weenie_hut_jr = read_file(swag)
    lookup = weenie_hut_jr[0]
    data = weenie_hut_jr[1]

    pca_results = get_pc(data, lookup, solver, save)
    table_pca = pca_results[0]
    n_comp = pca_results[1]
    kmeans(data, table_pca, solver, n_comp, k_range, n_iter, n_tol, save)
end_t = time.time()

time_iter = end_t - start_t
time_iter = time_iter % (24 * 3600)
hour = time_iter // 3600
time_iter %= 3600
minutes = time_iter // 60
time_iter %= 60

print("\nTotal time elapsed: {} hrs, {} min, {} sec\n".format( hour, minutes, time_iter))