"""
Created on Thu Oct  6 18:16:52 2022

@author: Rafael Rivero de Nicolás
"""
from numpy import array, linspace, hstack
import LB_Error_and_Convergence_rate as LB_erc # User's Module
import LB_Physics_Problems as ph # User's module

# %% Initialitation

Temporal_schemes_available = {0:"Euler",
                              1:"Inverse Euler",
                              2:"RK4",
                              3:"Crank-Nicolson"}

Physics_Problems_available = {0:"Kepler Orbits: 2 Bodies [2D]",
                              1:"Undamped Armonic Oscilator [1D]"}

'''\\\\\\\\\\\\\\\\\\\\\\'''
k = 3 # Selection of Numeric Scheme 0-1-2-3
P = 0 # 0-1
tf = 10

if P == 0:
    r_0 = array([1, 0]); v_0 = array([0, 1]) # Initial position and velocity, respectively. [Kepler Orbits 2 Bodies [2D]]
elif P == 1:
    r_0 = 0.5; v_0 = 0.5 # Initial position and velocity, respectively. [Undamped Armonic Oscilator [1D]]  
'''\\\\\\\\\\\\\\\\\\\\\\'''

Initial_conditions = hstack((r_0,v_0)); print('Initial State Vector: U_0 = ', Initial_conditions, '\n\n\n') 

Differential_operator = ph.Problem_Assignment(Physics_Problems_available[P],Physics_Problems_available)

physics_problem = Physics_Problems_available[P]
scheme = Temporal_schemes_available[k]

# %% Convergence Rate
 
M = 9 # Number of points to compute q

LB_erc.Convergence_Rate(Differential_operator = Differential_operator, Initial_conditions = Initial_conditions,
                  tf = tf, temporal_scheme = scheme, M = M, Adjust = True, Save = False)
                 
# %% Richardson Extrapolation to compute Error

dt_R = 0.01

time_domain_Richardson = linspace(0, tf, int(tf/dt_R)+1)

LB_erc.Richardson_Error_Extrapolation(Differential_operator, Initial_conditions, time_domain_Richardson, scheme, Save = False)