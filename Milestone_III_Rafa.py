# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:16:52 2022

@author: rafra
"""

# import numpy as np

from numpy import array, linspace, hstack
import LB_Error_and_Convergence_rate as LB_erc


def Kepler_Orbits_2N(X, t):
    '''
    This function only depends on the physics of the problem, it musts be an input argument
    
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Float
        Time instant in which F is being evaluated.

    Returns
    -------
    Array
        First derivate of the tate vector. dU/dt = F(U,t).

    '''
    F1 = X[2]
    
    F2 = X[3]
    
    F3 = -X[0]/(X[0]**2 + X[1]**2)**(3/2)
    
    F4 = -X[1]/(X[0]**2 + X[1]**2)**(3/2)

    return array([F1, F2, F3, F4])
    

# %% Initialitation

Temoral_schemes_available = {0:"Euler",
                             1:"Inverse Euler",
                             2:"RK4",
                             3:"Crank-Nicolson"}

'''\\\\\\\\\\\\\\\\\\\\\\'''
k = 1 # Selection of Numeric Scheme

Differential_operator = Kepler_Orbits_2N

tf = 10

r_0 = array([1, 0]) # Initial position

v_0 = array([0, 1]) # Initial velocity

U_0 = hstack((r_0,v_0)) # U_0 = np.array([r_0[0], r_0[1], v_0[0], v_0[1]])
print('Initial State Vector: U_0 = ', U_0, '\n\n\n')   
'''\\\\\\\\\\\\\\\\\\\\\\'''

scheme = Temoral_schemes_available[k]

Initial_conditions = U_0



# %% Convergence Rate
 
M = 9 # Number of points to compute q

LB_erc.Convergence_Rate(Differential_operator = Differential_operator, Initial_conditions = Initial_conditions,
                  tf = tf, temporal_scheme = scheme, M = M, Adjust = True, Save = True)
                 
# %% Richardson Extrapolation to compute Error


dt_R = 0.01

time_domain_Richardson = linspace(0, tf, int(tf/dt_R)+1)

LB_erc.Richardson_Error_Extrapolation(Differential_operator, Initial_conditions, time_domain_Richardson, scheme, Save = True)