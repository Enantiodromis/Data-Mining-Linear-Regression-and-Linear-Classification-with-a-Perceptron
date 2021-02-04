import math
import numpy as np
import random
from plotting import plot_graph
import matplotlib.pyplot as plt


# APPLYING BASIC LINEAR REGRESSION TO A SYNTHETIC DATASET
# Linear regression is a useful method for investigating relationships between a target variable y
# and set of variables (or attitudes) X. Below two methods are developed, gradient descent and the 
# scikit-learn version of the linear regression.
#
# SOLVING LINEAR REGRESSION WITH GRADIENT DESCENT
# The below implementation performs gradient descnent in order to fit the parameters w of a linear model.
# In the first step the parameters w are initialised with some random values or set to 0. They are then 
# gradually updated in the direction that minimises the objective function, using the following update rule.
# wi ← wi + α(yj − ˆyj) xj,i

# gradient_descent_2(), solves linear regression with gradient descent for 2
# INPUTS:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameter values (of size 2)
#  y = list of target values
#  alpha = learning rate
# OUTPUTS:
#  w = updated list of parameter values
def gradient_descent_2(M,x,w,y,alpha):
    for j in range(M):
        y_hat = w[0] + w[1] * x[j] # Predicting the taget value for this instance
        error = y[j] - y_hat # Comparing the predicition to the observation and computing the error
        # Asjusting the parameters based on the computed error
        w[0] = w[0] + alpha * error * 1 * (1.0/M)
        w[1] = w[1] + alpha * error * x[j] * (1.0/M)
    return w

# compute_error(), computes the sum of squared errors for the model.
# INPUTS:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# OUTPUTS:
#  error (scalar)
def compute_error(M,x,w,y):
    error = 0
    y_hat = [0 for i in range(M)]
    for j in range(M):
        y_hat[j] = w[0] + w[1] * x[j]
        error = error + math.pow((y[j] - y_hat[j]), 2)
    error = error/M
    return(error)

# compute_r2(), computes R^2 for the model.
# INPUTS:
#  M = number of instances
#  x = list of variable values for M instances
#  w = list of parameters values (of size 2)
#  y = list of target values
# OUTPUT:
#  R^2 (scalar)
def compute_r2(M,x,w,y):
    u = 0
    v = 0
    y_hat = [0 for i in range(M)]
    y_mean = np.mean(y)
    for j in range(M):
        y_hat[j] = w[0] + w[1] * x[j]
        u = u + math.pow((y[j] - y_hat[j]), 2)
        v = v + math.pow((y[j] - y_mean), 2)
    r2 = 1.0 - (u/v)
    return(r2)

def run_gradient_descent(alpha, epsilon, iterations, x_train, y_train, x_test, y_test):
    M_train = len(x_train)
    M_test = len(x_test) 
    #-run gradient descent to compute the regression equation
    y_hat = [0 for i in range( M_train )] # initialise predictions
    w = [random.random() for i in range( 2 )] # initialise weights
    prev_error = compute_error( M_train, x_train, w, y_train ) # compute initial error

    for num_iters in range( iterations ):
        w = gradient_descent_2( M_train, x_train, w, y_train, alpha ) # adjust weights using gradient descent
        # compute error
        curr_error = compute_error( M_train, x_train, w, y_train ) 
        r2 = compute_r2( M_train, x_train, w, y_train )

        num_iters = num_iters + 1
        
        # plot results, when error difference is > 1
        if ( num_iters % 1000 == 0 ):
            print( 'num_iters = %d  prev_error = %f  curr_error = %f  r^2 = %f' % ( num_iters, prev_error, curr_error, r2 ))
            for j in range( M_train ):
                y_hat[j] = w[0] + w[1] * x_train[j]
            
            plot_graph(x_train, y_train)
            plot_graph(x_train, y_hat)
            print( 'iteration ' + str( num_iters ) + ': y = ' + str( w[0] ) + ' + ' + str( w[1] ) + 'x, error=' + str( curr_error) + ' r^2=' + str( r2 ))
        
        # iterate until error hasn't changed much from previous iteration
        if ( prev_error - curr_error < epsilon ):
            converged = True
        else:
            prev_error = curr_error

    #-evaluate the model using the test set
    test_error = compute_error( M_test, x_test, w, y_test )
    test_r2 = compute_r2( M_test, x_test, w, y_test )
    print( 'evaluation:  test_error = %f  test_r^2 = %f' % ( test_error, test_r2 ))

    #-plot and save regression line from testing
    plot_graph(x_test, y_test)
    y_hat = [0 for i in range( M_test )]
    for j in range( M_test ):
        y_hat[j] = w[0] + w[1] * x_test[j]
    plot_graph(x_test, y_hat)


