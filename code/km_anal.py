# /Users/adrianaladera/Desktop/analysis/ML_BZT_doodoo/manuscript_figs_data/figgies
from re import S
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import axis
import numpy as np
import matplotlib.font_manager as foom
import os

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
    except ValueError:
        return False
    return True  

concentrations = ["0.00"]
root = "/Users/adrianaladera/Desktop/analysis/ML_BZT_doodoo/manuscript_figs_data/" #where the K-means-clustering-BZT directory is stored
main_dir = "{}K-means-clustering-BZT/".format(root)
plt.rcParams["axes.linewidth"] = 2.50

for conc in concentrations:
    data_table = pd.read_csv("{}{}/kmeans_results.csv".format(main_dir, conc))
    k_values = pd.read_csv("{}{}/k_value_selection.csv".format(main_dir, conc))

    ############################### PRINCIPLE COMPONENTS ###############################  
    fig, axs = plt.subplots(1)

    plt.tick_params(axis='both', which='both', labelsize=14) #tick labels

    plt.xlabel("number of principle components", fontname='Times New Roman', fontsize=12)
    plt.ylabel("cumulative % of variation explained", fontname='Times New Roman', fontsize=12)
    plt.xticks(fontname='Times New Roman', fontsize=12)
    plt.yticks(fontname='Times New Roman', fontsize=12)
            
    components = data_table["PCA-x"]
    cum_sum = data_table["PCA-y"]

    plt.plot(components, cum_sum, color="green", linestyle="solid", linewidth=3)
    plt.savefig("{}{}/PCA_plot.jpeg".format(main_dir, conc), dpi=600)

    ############################# DISTORTIONS #############################
    fit1x,fit1y, fit2x, fit2y = [], [], [], []
    fit15x, fit15y = [],[]
    if conc != "0.25":
        for s in range(5):
            fit1x.append(k_values["k"][s])
            fit1y.append(k_values["distortions"][s])
        for s in range(5):
            fit2x.append(k_values["k"][s+3])
            fit2y.append(k_values["distortions"][s+3])
    if conc == "0.15":
        for s in range(3):
            fit15x.append(k_values["k"][s+1])
            fit15y.append(k_values["distortions"][s+1])
    else:
        for s in range(3):
            fit1x.append(k_values["k"][s])
            fit1y.append(k_values["distortions"][s])
        for s in range(7):
            fit2x.append(k_values["k"][s+1])
            fit2y.append(k_values["distortions"][s+1])

    plt.plot(k_values["k"], k_values["distortions"], color='#FF1493', marker='o')
    coef = np.polyfit(fit2x,fit2y,1)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(fit2x, poly1d_fn(fit2x), '--k')
    if conc == "0.15":
        coef = np.polyfit(fit15x,fit15y,1)
        poly1d_fn = np.poly1d(coef) 
        plt.plot(fit15x, poly1d_fn(fit15x), '--k')
    font = foom.FontProperties(family='Times New Roman', size=14)
    plt.title("x = {}".format(conc))
    plt.savefig("{}{}/distortions_elbow_X-{}.jpeg".format(main_dir, conc, conc), dpi=600)

    ############################# INERTIA #############################    
    plt.clf()
    fit1x,fit1y, fit2x, fit2y = [], [], [], []
    if conc == "0.05":
        for s in range(5):
            fit1x.append(k_values["k"][s])
            fit1y.append(k_values["inertia"][s])
        for s in range(5):
            fit2x.append(k_values["k"][s+3])
            fit2y.append(k_values["inertia"][s+3])
    else:
        for s in range(4):
            fit1x.append(k_values["k"][s])
            fit1y.append(k_values["inertia"][s])
        for s in range(6):
            fit2x.append(k_values["k"][s+2])
            fit2y.append(k_values["inertia"][s+2])
                
    plt.plot(k_values["k"], k_values["inertia"], color='#8A2BE2', marker='o')
    coef = np.polyfit(fit2x,fit2y,1)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(fit2x, poly1d_fn(fit2x), '--k')
    font = foom.FontProperties(family='Times New Roman', size=14)
    plt.title("x = {}".format(conc))
    plt.savefig("{}{}/inertia_elbow_X-{}.jpeg".format(main_dir, conc, conc), dpi=600)
        
    # ############################# PHASES(T) ############################# 
    for k in range (2, 10):
        plt.clf()
        plt.tick_params(axis='both', which='both', labelsize=14) #tick labels
        ax_count = 0
        x_nested = []
        y_nested = []
                
        clusters = {}
                
        # separating the list into a dictionary where keys are cluster labels
        for x in range(k):
            temp = []
            for i in range(len(data_table)):
                if data_table["k{}, pred".format(k)][i] == x: # if cluster prediction == given cluster holder
                    temp_nested = []
                    temp_nested.append(data_table["temperature".format(k)][i]) # temperature (K)
                    temp_nested.append(data_table["cf_pattern".format(k)][i]) # CF thing
                    temp_nested.append(data_table["k{}, pred".format(k)][i]) # cluster prediction
                    temp_nested.append(data_table["k{}, x".format(k)][i]) # scatter plot x
                    temp_nested.append(data_table["k{}, y".format(k)][i]) # scatter plot y
                    temp.append(temp_nested)
            clusters["c" + str(x)] = temp
                
        #sorting in order of temperature and writing to file
        my_ass = []
        for weenie in clusters:
            for i in range(len(clusters[weenie])):
                my_ass.append(clusters[weenie][i])
            my_ass.sort(key = lambda i:i[0], reverse = False) #sorting by temperature
    
        clust_keys = {}
        start_key = k - 1
        for i in range(len(my_ass)):
            x_nested.append(int(my_ass[i][0]))
            old_y = int(my_ass[i][2])
            if old_y not in clust_keys:
                clust_keys[old_y] = start_key 
                start_key -= 1
            y_nested.append(clust_keys[old_y])

        plt.yticks(range(k))
        plt.plot(x_nested, y_nested, color="#0000FF", marker='o',linewidth=2)
        font = foom.FontProperties(family='Times New Roman', size=14)
        plt.title("x = {}, k = {}".format(conc, k))
        plt.savefig("{}{}/predictions_X-{}_k-{}.jpeg".format(main_dir, conc, conc, k), dpi=600)
        
        ############################# CLUSTER PLOTS ############################# 
        plt.clf()
        color_list = {0:'#CC0000', 1:'#FF7F50', 2:'#FFD700', 3:'#008000', 4:'#2ACAEA',
                        5: "#0000FF", 6: "#8A2BE2", 7: "#FF1493", 8: 'grey', 9:'brown'}

        centx = data_table["k{}, centx".format(k)]
        centy = data_table["k{}, centy".format(k)]

        centx = centx[centx != 0]
        centy = centy[centy != 0]      

        data = {}
        clust_keys = {}
        start_key = k - 1
        cunt = 0
        for p in range(len(data_table)):
            old = data_table["k{}, pred".format(k)][p]
            if old not in clust_keys:
                clust_keys[old] = start_key 
                start_key -= 1
            if clust_keys[old] not in data:
                data[clust_keys[old]] = {'x':[], 'y':[]}
            else:
                data[clust_keys[old]]['x'].append(data_table["k{}, x".format(k)][p])
                data[clust_keys[old]]['y'].append(data_table["k{}, y".format(k)][p])

        colors = dict(filter(lambda i:i[0] in range(k), color_list.items()))
                    
        for col in colors:
            plt.scatter(data[col]['x'], data[col]['y'], c=colors[col], label=col)
        plt.scatter(centx, centy, c="black", s=80, marker="^")

        font = foom.FontProperties(family='Times New Roman', size=10)
        plt.legend(loc="upper left", prop = font)
        plt.title("x = {}, k = {}".format(conc, k))  
        plt.savefig("{}{}/cluster-plots_X-{}_k-{}.jpeg".format(main_dir, conc, conc, k), dpi=600)
