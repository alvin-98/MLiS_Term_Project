This repository contains the files for the project - Using Unsupervised Learning Methods To Understand Characteristic Differences in
Clinical Features For Benign And Malignant Breast Cancer Tumours

There are 4 files in the repository
  1. Group C1 Project Notebook (Jupyter notebook) - Contains the code that executes the pipeline for the project from EDA, clustering to the statistical test results.
  2. clustering.py - contains implementation of clustering algorithms (K-means, HAC, and DBSCAN)
  3. metrics.py - contains implementation of clustering evaluation metrics (Silhouette score, Davies–Bouldin index, and Adjusted Rand Index)
  4. preprocessing.py - contains implementation of standard scaler


To run the notebook, make sure all the four files are in the same directory. Then, download the data from the UCI Repository (https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) and save it in a folder named 'data'.
The project uses very minimal libraries - numpy, pandas, scipy, and matplotlib. Make sure these are installed in the virtual environment you use to run the code. 
Finally, run the jupyter notebook - Group C1 Project Notebook.
