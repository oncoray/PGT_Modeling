#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:55:33 2023

@author: kiesli21
"""
import subprocess

# R script as a multi-line string
r_script = """
install.packages(c("randomForestSRC"), repos="http://cran.rstudio.com/", lib='/bigdata/invivo/machine_learning/pgt-range-reconstruction/PMMA_study/range_reconstruction_pmma_phantom/data/familiar/library')

"""

# Write the R script to a file
with open("install_xml2.R", "w") as file:
    file.write(r_script)

# Path to the Rscript executable - this might vary depending on your installation
rscript_path = "Rscript"

# Execute the R script
try:
    subprocess.run([rscript_path, "install_xml2.R"], check=True)
    print("Package installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
