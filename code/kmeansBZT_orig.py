from dis import dis
from math import dist
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

#####################################################################
# writes data arrays to files
#####################################################################
def writefile(filename, arr, flag, save_dir):
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    weewoo_file = open(save_dir + filename, "w")
    print("Writing file", filename)
    for r in range(len(arr)):
        if flag:
            for c in range(len(arr[r])):
                weewoo_file.write("{:30}".format(arr[r][c]))
        else:
            weewoo_file.write("{:10}".format(arr[r]))
        weewoo_file.write("\n")
    weewoo_file.close()

#####################################################################
# reads a transposed 2D matrix of data and transposes it back
# into its original M x N form
#####################################################################
def read_file(filename):
    df = pd.read_csv(filename)
    if "data" in filename:
        df = df.drop('temperature',  axis=1)
        df = df.drop('cf_pattern',  axis=1)
    return df

#####################################################################
# finds number of principle components and saves cumulative 
# explained variance
#####################################################################
def get_pc(data, solver, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if solver == "arpack":
        n_comp = len(data)-1
    else:
        n_comp = len(data)
    pc_anal = PCA(n_components = n_comp, svd_solver=solver)
    pc_anal.fit_transform(data)
    pca_variance = []
    num_components = 0
    for i, cum in zip(range(1, n_comp+1), np.cumsum(pc_anal.explained_variance_ratio_ * 100)):
        pca_variance.append([i, cum])
    for i, cum in zip(range(1, n_comp+1), np.cumsum(pc_anal.explained_variance_ratio_ * 100)):
        if cum >= 99:
            num_components = i
            print("principle components: {}".format(num_components))
            break

    return pca_variance, num_components

#####################################################################
# transforms the data using principle component analysis (PCA) 
# and plotting PCA data
#####################################################################
def kmeans(data, lookup, solver, n_comp, k_range, n_iter, n_tol, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("data shape before PCA (samples, variables): ({}, {})".format(len(data), len(data[0])))
    pca = PCA(n_components = n_comp, svd_solver=solver)
    data_pca = pca.fit_transform(data)
    print("data shape after PCA (samples, variables):{}".format(data_pca.shape))

    # making predictions with k-means
    identifiers, centroid_list, distortions, predictions = [], [], [], []
    silhouettes = []
    predictions = []
    inertia = []
   
    start_t = time.time()
    for k in k_range:
        km = KMeans(
        n_clusters = k, init = 'random',
        n_init = 10, max_iter = n_iter, 
        tol = n_tol, random_state = 0)
        
        pred = km.fit_predict(data_pca) #algorithm predictions using k-means
        lookup["k = {}".format(k)] = pred
        predictions.append(pred)

        cluster = np.unique(pred) # cluster identifiers, an arbitrary int label
        identifiers.append(cluster)
            
        centroids = km.cluster_centers_ # centroid is the center of a cluster
        centroid_list.append(centroids)
        
        labels = km.labels_
        silhouettes.append(silhouette_score(data_pca, labels, metric = 'euclidean'))

        inertia.append(km.inertia_)

        clust_dat = open("{}cluster_data_k{}.txt".format(save_dir, k), "w")

        # p = predictions, a = cluster plot x, b = cluster plot y
        for p, a, b in zip(pred, data_pca[:, 0] , data_pca[:, 1]):
            clust_dat.write("{}\t{}\t{}\n".format(p, a, b))
        clust_dat.close()

        # cluster plot coordinates for centroids
        centroid_file = open("{}centroids_k{}.txt".format(save_dir, k), "w")
        for a, b in zip(centroids[:,0] , centroids[:,1]):
            centroid_file.write("{}\t{}\n".format(a, b))
        centroid_file.close()
                
        distortions.append(sum(np.min(cdist(data_pca, km.cluster_centers_, 'euclidean'), axis = 1)) / data_pca.shape[0])

        end_t = time.time()
        time_k = end_t - start_t
        minnies = int(time_k / 60)
        secs = time_k % 60
        print("\nTime elapsed for k = {}:".format(k), minnies, "min, {:.5f} sec\n".format(k, secs))

    data_frame = []
    data_frame.append(k_range)
    data_frame.append(distortions)
    data_frame.append(inertia)
    data_frame.append(silhouettes)

    return predictions, data_frame
    
# =============================================================================
# MAIN                    
# =============================================================================
solver = "full" # solver can be "full", "randomized", "auto", or "arpack"
swag = "datasets/BZT_C-0.0_10K-450K_data.csv"
yeehaw = "datasets/BZT_C-0.00_10K-450K_lookup.csv"
n_iter = 1600 # the maximum iterations per a single run of the algorithm
n_tol = 1e-10 # tolerance limit (error) 
k_range = [2, 3, 4, 5, 6, 7, 8, 10] # k = 2 to k = 10
save = os.getcwd() + '/results/' #change depending on temp range
solver = "full" # solver can be "full", "randomized", "auto", or "arpack"

if not os.path.exists(swag):
    sys.exit("The path \"{}\" does not exist. Please make sure you have downloaded the necessary\n"
    " directories from the google drive before proceeding.")
save = os.getcwd() + '/0.05/' #change depending on temp range
if not os.path.exists(save):
    os.makedirs(save)

total_time = 0
start_t = time.time()
data = read_file(swag)
lookup = read_file(yeehaw)

pca_variance = get_pc(data, solver, save)[0]
num_components = get_pc(data, solver, save)[1]
filename = "PCA-iter" + str(n_iter) + "-tol" + str(n_tol) + ".txt"
kmeans_data = kmeans(data, solver, num_components, k_range, n_iter, n_tol, save)
end_t = time.time()

time_iter = end_t - start_t
total_time += time_iter

time_iter = time_iter % (24 * 3600)
hour = time_iter // 3600
time_iter %= 3600
minutes = time_iter // 60
time_iter %= 60

print("\nTime elapsed for Ba(Ti0.95,Zr0.05)O3 - {} hrs, {} min, {} sec\n".format(hour, minutes, time_iter))
   
# writing PCA data and predictions to files
writefile(filename, pca_variance, 1, save)
for i, k in zip(range(len(kmeans_data[0])), range(len(k_range))):
    filename = "k" + str(k_range[k]) + "-iter" + str(n_iter) + "-tol" + str(n_tol) + ".txt"
    writefile(filename, kmeans_data[0][i], 0, save)

#transposing data
kmeans_transposed = np.transpose(kmeans_data[1])

writefile("optimal_k_methods.txt", kmeans_transposed, 1, save)