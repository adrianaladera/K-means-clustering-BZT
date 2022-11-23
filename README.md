# K-means-clustering-BZT
Public access repository for scientists and curious people alike! Use k-means clustering to determine the phase transitions in barium zirconate supercells with 5% Zr concentration (Ba(Ti0.95,Zr,0.05). This source code accompanies the paper  <come back and check the concentrations later>
  
  - dependencies: Python >= 3.0, Anaconda, Scikit Learn = 1.1.3
  
  All of the following scripts can be run in the directory:
  
  `<PATH_TO_YOUR_DIR>/K-means-clustering-BZT/code/`
  
## Create the datasets
 ### Renaming all files
The dipole pattern plotting files usually come in the format "Stripe*.plt", where * = [X, Y, Z] depending on whether a viewer wants to view the dipoles in the positive or negative direction with respect to the desired axis. You can use `move_files.py` to rename all `.plt` files from a source directory and move them to a destination directory. This code assumes that all files are in a directory tree with the format: 
  
  `<SOURCE_DIR/alloy/T_<temp>/CF_<cf>/Stripe*.plt>`
  
  Ensure that all resulting files are in the same directory for a given concentration (i.e. if you have the concentrations $x = 0.05,0.15,0.25$, then you will have 3 separate directories, 0.05/, 0.15/, and 0.25/, each with their own set of resulting `.txt` files).
  
  You can run the code with the following command:
  
  `python move_files.py`
  
 ### Formatting to a CSV input file 
 Assuming that all input files (each which represent a singular dipole pattern) are in the format `StripeX-T*_CF*.txt`, where `T*` is an integer temperature in Kelvin (i.e. "T40" = 40 K) and `CF*` is an integer representing the pattern output at a certain time step (i.e. CF2 when MD time step = 100,000 means the output pattern at 20,000 time steps), the file `create_csv_data_sets.py` takes all coordinate and vector from the stripe files and converts them into a 2D matrix, such that each row corresponds to one input file (where each file describes a dipole pattern) and the columns represent the variables for that dipole pattern. The variables are represented as the coordinates and vector components of a dipole $d$, for each dipole $i$ in the total number of dipoles $I$:
  
  $d_i = (p_{ix}, p_{iy}, p_{iz}, r_{ix}, r_{iy}, r_{iz})$
  
  where $p_i$ is the position and $r_i$ are the vector components for the given dipole $d_i$. The dipoles are laid end to end to construct a single vector (i.e. a row of variables) corresponding to one dipole pattern. There is one matrix dataset per concentration.
  
  You can create your dataset with the following command:
  
  `python create_csv_datasets.py`
  
## K-means clustering
 `kmeans-BZT.py` is an unsupervised learning program that utilizes the K-means clustering algorithm, an unsupervised algorithm that groups data into cluster groups  without human input other than a K-value, which dictates how many clusters the algorithm should group data point into. It accepts data as an $P \times N$ array, where $P$ is the number of samples and $N$ is the dimension, i.e. variables of each sample. In this case, $P$ is the total number of dipole patterns for a given concentration, including variations in temperature and CF output pattern, and $N$ is the total number of dipoles $\times 6$, where 6 is the number of variables required to describe a single dipole. This script should be in the same directory as `K-means-clustering-BZT` folder.

  K-MEANS CLUSTERING DOCUMENTATION: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

  PRINCIPLE COMPONENT ANALYSIS DOCUMENTATION: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
  
  The total time to read one dataset (i.e. one concentration) is ~6 minutes, 20 seconds if using a remote host.
  
## Analyzing the results
  Run the command `python km_anal.py` to produce plots for principle component analysis, elbow methods, and cluster distributions. All resulting plots are saved in `<PATH_TO_YOUR_DIR>/K-means-clustering-BZT/<concentration>`
  
  ### Principle component analysis
  Saves the cumulative percentage of variance explained by the given number of components. The results in the paper recognize the number of principle components that yield a 99% variance.
  
  ### Elbow methods
  Plots the distortion/inertia as a function of $k$. Distortion is defined as the average distance of each data point from their respective cluster centroid, whereas intertia is the sum of the squared distances of each point from their respective cluster centroid. Ideally, the first value of $k$ at which the distortions/inertias are minimized (i.e. the $k$ at which the graph begins to decrease linearly) is the optimal $k$ value, $k_O$.
  
  ### Cluster plots
  The most variance is normally explained by the first two principle components, hence why the first two principle components are plotted against one another to see their cluster groupings. Each cluster is given an arbitrary integer to denote their group, as well as a separate color scheme.
  
  ### Phase vs. temperature
  Takes the data points and plots their cluster number as a function of temperature to create a K-means-predicted version of a temperature-polarization phase diagram. Each arbitrary number is mapped and reassigned such that the new number mapping is in descending order as the temperature increases; this helps to produce a step-like plotting scheme.
  
## Visualizing dipole patterns
  Accepts a dictionary such that:
  
  `dictionary = {conc1: [list of $k$ values], conc2: [list of $k$ values], etc.}`
  
  
  where conc is a key string of the desired concentraiton and its value is a list of the $k$ values desired to be visualized. For each $k$ and each concentration, the script produces a dipole pattern of the median temperature of each cluster number for the given $k$. All resulting figures are stored in `<PATH_TO_YOUR_DIR>/K-means-clustering-BZT/plot_med_clusters/<concentration>` with the naming scheme `k<K VALUE>_T<TEMPERATURE>_CF<CF PATTERN NUMBER>.jpeg`. For a given $k$, median temperature dipole patterns with the temperatures in ascending order correspond to the phase vs. temperature plots with the mapped cluster number in descending order.
  
