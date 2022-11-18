# K-means-clustering-BZT
Public access repository for scientists and curious people alike! Use k-means clustering to determine the phase transitions in barium zirconate supercells with 5% Zr concentration (Ba(Ti0.95,Zr,0.05) <come back and check the concentrations later>
  
## Create the datasets
 ### Renaming all files
The dipole pattern plotting files usually come in the format "Stripe*.plt", where * = [X, Y, Z] depending on whether a viewer wants to view the dipoles in the positive or negative direction with respect to the desired axis. You can use `move_files.py` to rename all `.plt` files from a source directory and move them to a destination directory. This code assumes that all files are in a directory tree with the format: 
  
  `<SOURCE_DIR/alloy/T_<temp>/CF_<cf>/Stripe*.plt>`
  
  Ensure that all resulting files are in the same directory for a given concentration (i.e. if you have the concentrations $x = 0.05,0.15,0.25$, then you will have 3 separate directories, 0.05/, 0.15/, and 0.25/, each with their own set of resulting `.txt` files).
  
 ### Formatting to a CSV input file 
 Assuming that all input files (each which represent a singular dipole pattern) are in the format `StripeX-T*_CF*.txt`, where `T*` is an integer temperature in Kelvin (i.e. "T40" = 40 K) and `CF*` is an integer representing the pattern output at a certain time step (i.e. CF2 when MD time step = 100,000 means the output pattern at 20,000 time steps), this file takes all coordinate and vector from the stripe files and converts them into a 2D matrix, such that each row corresponds to one input file, a dipole pattern, and the columns represent the variables for that dipole pattern. The variables are represented as the coordinates and vector components of a dipole $d$, for each dipole $i$ in the total number of dipoles $I$:
  
  $d_i = (p_ix, p_iy, p_iz, r_ix, r_iy, r_iz)$
  
  where $p_i$ is the position and $r_i$ are the vector components for the given dipole $d_i$. The dipoles are laid end to end to construct a single vector (i.e. a row of variables) corresponding to one dipole pattern. There is one matrix dataset per concentration.
  
## K-means clustering
  
## Cumulative variance, elbow methods, and cluster plots
  
## Visualizing dipole patterns
  
