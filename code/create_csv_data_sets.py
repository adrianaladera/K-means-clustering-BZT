# /Users/adrianaladera/Desktop/analysis/ML_BZT_doodoo/original_files/Alloy_{}
# =============================================================================
#  Author: Adriana Ladera
#
# Assuming that all input files are in the format "StripeX-T*_CF*.txt", where T*
# is an integer temperature in Kelvin (i.e. "T40" = 40 K) and CF* is an integer
# representing the pattern output at a certain time step (i.e. CF2 when MD
# time step = 100,000 means the output pattern at 20,000 time steps), this file
# takes all input from the stripe files and converts them into a 2D matrix, such 
# that each row corresponds to one input file, a domain pattern, and the columns
# represent the variables for that pattern (i.e. the positions and vector 
# components for all dipoles in the pattern). There is on matrix dataset per
# concentration.
# =============================================================================

import os
import re
import pandas as pd
import numpy as np
import time

concentrations = ["0.15"] # optionally a list if there is more than one concentration
# where input files are stored, must all be in the same directory
source = "<YOUR SOURCE DIR HERE>" 
destination = "<YOUR DESTINATION DIR HERE>" # directory to store all datasets
table_dir = destination + "data_tables/"

if not os.path.exists(table_dir):
  os.makedirs(table_dir)

start_t = time.time()
for conc in concentrations:
    df = pd.DataFrame()
    cftID = [] # lookup table: stores temperature and pattern CF number
    data = [] # store 2D dataset of M x N, where M = samples and N = variables per sample
    
    file_count = 1
    T = 10
    temperature, pattern = [], []
    while T <= 450:
        cf = 1
        while cf <= 10:
            temp_df = pd.DataFrame()
            kermit = "{}{}/StripeX-T{}_CF{}.txt".format(source, conc, T, cf) #name of file in directory
            split_swag = kermit.partition('.')
            print(kermit)
            weenie_file = open(kermit, "r", encoding = "windows-1252")
            lines = weenie_file.readlines()
            temp = []
            cunt = 1
            for line in lines: 
                #if cunt % 10 == 0 and line.split()[1] == "arrow":
                if line.split()[1] == "arrow":
                    n = len(line.split())
                    row = []
                    for i in range(n):
                        rep = re.sub(r',','', str(line.split()[i]))
                        row.append(rep)
                    temp.append(float(row[6])) # position_x
                    temp.append(float(row[7])) # p_y
                    temp.append(float(row[8])) #p_z
                    x_comp = float(row[10]) - float(row[6]) # r_x
                    y_comp = float(row[11]) - float(row[7]) # r_y
                    z_comp = float(row[12]) - float(row[8]) # r_z
                    temp.append(round(x_comp, 3))
                    temp.append(round(y_comp, 3))
                    temp.append(round(z_comp, 3))
                cunt += 1
            temperature.append(T)
            pattern.append(cf)
            data.append(temp)
            transpose = np.transpose(data) #transposing helps with faster data reading
            file_count += 1
            cf += 1
        T += 10

    # adds corresponding temp and CF output to dataframe CSV
    df["temperature"] = temperature 
    df["cf_pattern"] = pattern

    ass = 1
    for t in transpose:
        df[ass] = t
        ass += 1

    df.to_csv("{}BZT_C-{}_10K-450K_data.csv".format(table_dir, conc))

end_t = time.time()

time_iter = end_t - start_t
time_iter = time_iter % (24 * 3600)
hour = time_iter // 3600
time_iter %= 3600
minutes = time_iter // 60
time_iter %= 60

print("\nTotal time elapsed: {} hrs, {} min, {} sec\n".format(hour, minutes, time_iter))