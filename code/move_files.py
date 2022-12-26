# ===============================================================================
#  Author: Adriana Ladera
#
# Assumes that the necessary input *.plt files are in a directory with the format:
# "<SOURCE_DIR/alloy/T_<temp>/CF_<cf>/Stripe*.plt>"
# can rename accordingly
# ===============================================================================
import os

source = "<YOUR SOURCE DIRECTORY HERE>"
destination = "<YOUR DESTINATION DIRECTORY HERE>"



alloy_list = ["0.20"] # optionally a list of concentrations, must be a string
for alloy in alloy_list:
    temp = 10
    while temp <= 450:
        cf = 1
        while cf <= 10:
            path_to_file = "{}{}/T_{}/CF_{}/StripeX.plt".format(source, alloy, temp, cf)
            if os.path.exists(path_to_file) and path_to_file.endswith(".plt"):
                print("path exists bestie")
                new_path_to_file = "{}{}/StripeX-T{}_CF{}.txt".format(destination, alloy, temp, cf)
                os.rename(path_to_file, new_path_to_file)
            cf += 1
        temp += 10
            