import pandas as pd
import numpy as np
from partition_data import partition_data
from plotting import plot_graph
from sklearn import model_selection
from sklearn import datasets
from linear_regression_and_error_functions import run_gradient_descent

# PREPARING DATA:
# Below we are working with the dataset "london-borough-profiles-jan2018.csv" and synthetic data.
# The synthetic data, is useful when learning new techniques, as the characteristics of the data
# can be specified and a tool can be used to generate random instances of that type of data.

# FORMATTING AND DISPLAYING DATASET:
# Reading in the dataset, focusing on specifically columns 70 and 71 we plot the data.
data = pd.read_csv('london-borough-profiles-jan2018.csv') # Reading in the .csv dataset
male_le = data[data.columns[70]].replace('.',np.nan).map(lambda x: float(x)) # Replacing "." in data with nan
female_le = data[data.columns[71]].replace('.',np.nan).map(lambda x: float(x)) # Replacing "." in data with nan
plot_graph(male_le,female_le,'age(men)','age(women)','raw data') # Plotting the dataset

# PARTIONING DATA INTO TRAINING AND TESTING:
# It is a bad idea to use the same data set for both training and evaluating. The evaluation will only inidicate
# how well your model represents your training data and not how well it represents "unseen" data.
# Therefore you want to divide the dataset into two partions. It is common to use 90% for training and 10% for testing.
partition_data(0.10,male_le, female_le,'age(men)', 'age(women)', 'partitioned data') # Partioning and plotting the data

# GENERATING A SYNTHETIC DATASET:
# Useful to generate a synthetic dataset where you specify the number and data type of attributes
# for which the properties of each attribute can be specified.
# A synthetic dataset was randomly generated of 100 data points (x,y) the target variables, y are drawn
# from a linear model y=p x+p with some randomly selected p and pre-defined noise factor q. We do that 
# using the sciki-learn function dataset.make_regression().
#
# The function parameters are:
# - n_features = defines how many attributes to generate for each instance.
# - n_informative = how many of those are used to infer the target variables.
# - noise = the noise factor q.
# - n_samples = the number of instances to generate.
# - coef = a flag indicating whether to return the p coefficients used to generate the taget values y.
#
# The function returns:
# - x the generated 2D array of features of size n_samples x n_features
# - y an array of target values, also of size n_samples
# - p an array of parameters also of size n_samples
x,y,p = datasets.make_regression(n_samples=100,n_features=1,n_informative=1,noise=10,coef=True)
plot_graph(x,y,'x','y','synthetic data un-partitioned') # Plotting the synthetic dataset
partition_data(0.10,x, y,'x', 'y', 'synthetic data partitioned')

num_features = 1
x, y, p = datasets.make_regression( n_samples=100, n_features=num_features, n_informative=1, noise=10, coef=True )
x_train, x_test, y_train, y_test = model_selection.train_test_split( x, y, test_size=0.10 )

a = 0.001
e = 0.001
iterations = 10000

run_gradient_descent(a, e, iterations, x_train, y_train, x_test, y_test)



