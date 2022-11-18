# /Users/adrianaladera/Desktop/analysis/ML_BZT_doodoo/manuscript_figs_data/plot_med_clusters
# ===============================================================================
# by Adriana Ladera                                                             
#                                                                               
# Plots a supercell according to the 8 permutations of x,y,z components given   
# that the components are either positive or negative. If one of the components 
# has a 0 value, then the arrow will be colored black. Gives the supercell of   
# the median temperature of each cluster for a given K.           
# ===============================================================================

import matplotlib.pyplot as plt
import os
import pandas as pd
import time

# where the K-means-clustering-BZT directory is stored (not including the "-means-clustering-BZT/" directory)
root = "<YOUR DIRECTORY HERE>" 
main_dir = "{}K-means-clustering-BZT/".format(root)
dipole_dir = "{}source_files/"
 
# dictionary of concentrations and their corresponding k values that you want to visualize 
# normally determined by the elbow plots, in which the optimal K is visualized as well as K+-1
# k-range [2, 10]
valid_k = { "0.15": [3, 4, 5]} 

start = time.time()
for conc in valid_k.keys():
    results_table = pd.read_csv("{}{}/kmeans_results.csv".format(main_dir, conc))
    k_values = pd.read_csv("{}{}/k_value_selection.csv".format(main_dir, conc))
    destination = "{}plot_med_clusters/{}/".format(main_dir, conc)
    if not os.path.exists(destination):
        os.makedirs(destination)
    if os.path.isdir(main_dir + conc):
        for k in valid_k[conc]:

            clusters = {}
            clust_keys = {}
            new_key = k - 1
            c_range = range(k)

            for c in c_range:
                pelota = []
                for el_pepe in range(len(results_table["k{}, pred".format(k)])):
                    if results_table["k{}, pred".format(k)][el_pepe] == c:
                        # old_key = c
                        # if old_key not in clust_keys:
                            # clust_keys[old_key] = new_key 
                            # new_key -= 1
                        pelota.append([results_table["temperature"][el_pepe], results_table["cf_pattern"][el_pepe]])
                clusters["c{}".format(c)] = pelota
            
            # find median temperature and cf pattern of each cluster
            for clust, key in zip(clusters, clusters.keys()):
                pelota.sort(key=lambda x:x[0])
                median_list = clusters[clust]
                median_list.sort(key=lambda x:x[0])
                med_key = median_list[len(median_list)//2]
                
                print(k, med_key[0], med_key[1]) # k value, temperature, pattern

                fig = plt.figure(figsize=(5,4))
                ax = fig.gca(projection='3d')

                read_file = open("{}{}/StripeX-T{}_CF{}.txt".format(dipole_dir, conc, med_key[0], med_key[1]), "r")
                lines = read_file.readlines()
                    
                # processing lines in old stripe file
                for l in lines:
                    flag = 0
                    if l.split()[1] == "arrow": # if line contains component information
                        n = len(l.split())
                        row = []
                        temp = []
                        for i in range(n):
                            ass = l.split()[i]
                            weenie = ass.split(',')
                            if len(weenie) == 2 and weenie[1] != '':
                                row.append(weenie[0])
                                row.append(weenie[1])
                            else:
                                row.append(weenie[0])
                        x_comp = float(row[10]) - float(row[6]) 
                        y_comp = float(row[11]) - float(row[7])
                        z_comp = float(row[12]) - float(row[8]) 
                        temp.append(float(row[6]))
                        temp.append(float(row[7]))
                        temp.append(float(row[8]))
                        temp.append(x_comp)
                        temp.append(y_comp)
                        temp.append(z_comp)
                                
                        # assigning color value to different x y and z components
                        if x_comp > 0 and y_comp > 0 and z_comp > 0: # x y z = + + +
                            temp.append("#FF1493") #pink
                        if x_comp > 0 and y_comp > 0 and z_comp < 0: # x y z = + + -
                            temp.append("#CC0000") #red
                        if x_comp > 0 and y_comp < 0 and z_comp < 0: # x y z = + - -
                            temp.append("#FF7F50") #orange
                        if x_comp > 0 and y_comp < 0 and z_comp > 0: # x y z = + - +
                            temp.append("#FFD700") #yellow   
                        if x_comp < 0 and y_comp > 0 and z_comp > 0: # x y z = - + +
                            temp.append("#008000") #green
                        if x_comp < 0 and y_comp > 0 and z_comp < 0: # x y z = - + -
                            temp.append("#2ACAEA") #blue  
                        if x_comp < 0 and y_comp < 0 and z_comp < 0: # x y z = - - -
                            temp.append("#0000FF") #dark blue
                        if x_comp < 0 and y_comp < 0 and z_comp > 0: # x y z = - - +
                            temp.append("#8A2BE2") #purple
                        else: #one of the components has a 0 value
                            temp.append("black")
                        
                        # plots the respective vector
                        ax.quiver([temp[0]], [temp[1]], [temp[2]], [temp[3]], [temp[4]], [temp[5]], colors=[temp[6]], length=1, linewidth=0.5)                   
            
                #change limits according to supercell size
                ax.set_xlim3d([0, 30])
                ax.set_xlabel('X (nm)')
                
                ax.set_ylim3d([0, 30])
                ax.set_ylabel('Y (nm)')
                
                ax.set_zlim3d([0, 30])
                ax.set_zlabel('Z (nm)')
                ax.set_title('k{}, T = {}K, CF = {}'.format(k, med_key[0], med_key[1]))
                
                ax.view_init(elev=35, azim=-46)
                #ax.set_axis_off()         

                fontname = 'Times New Roman'

                # changing font size and font style
                for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
                    label.set_fontsize(11)
                    label.set_fontname(fontname)
                ax.xaxis.label.set_fontname(fontname)
                ax.yaxis.label.set_fontname(fontname)
                ax.zaxis.label.set_fontname(fontname)
                ax.title.set_fontname(fontname)
                ax.title.set_fontsize(17)
                ax.xaxis.label.set_fontsize(12)
                ax.yaxis.label.set_fontsize(12)
                ax.zaxis.label.set_fontsize(12)
                ax.title.set_fontweight('bold')
            
                plt.savefig("{}/k{}_T{}_CF{}.jpeg".format(destination, k, med_key[0], med_key[1]), dpi=600)

end = time.time()

time_iter = end - start
time_iter = time_iter % (24 * 3600)
hour = time_iter // 3600
time_iter %= 3600
minutes = time_iter // 60
time_iter %= 60

print("\nTotal time elapsed: {} hrs, {} min, {} sec\n".format( hour, minutes, time_iter))