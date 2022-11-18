# K-means-clustering-BZT
Public access repository for scientists and curious people alike! Use k-means clustering to determine the phase transitions in barium zirconate supercells with 5% Zr concentration (Ba(Ti0.95,Zr,0.05) <come back and check the concentrations later>
  
## Create the CSV datasets
 ### Renaming all files
The dipole pattern plotting files usually come in the format "Stripe*.plt", where * = [X, Y, Z] depending on whether a viewer wants to view the dipoles in the positive or negative direction with respect to the desired axis. You can use `move_files.py` to rename all `.plt` files from a source directory and move them to a destination directory. This code assumes that all files are in a directory tree with the format: 
  
  `<SOURCE_DIR/alloy/T_<temp>/CF_<cf>/Stripe*.plt>`
  
  Ensure that all resulting files are in the same directory for a given concentration (i.e. if you have the concentrations $x = 0.05,0.15,0.25$, then you will have 3 separate directories, 0.05/, 0.15/, and 0.25/, each with their own set of resulting `.txt` files).
  
  2. 
  
  
### K-means clustering
  
### Cumulative variance, elbow methods, and cluster plots
  
### Visualizing dipole patterns
  
