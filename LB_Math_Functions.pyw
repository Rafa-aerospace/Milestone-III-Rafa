# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:03:25 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import zeros, linalg, matmul
import LB_Temporal_Schemes as ts


def Cauchy_Problem(F, U_0, time_domain, Temporal_scheme='RK4'):
    
    print( 'Temporal Scheme used:: ' + Temporal_scheme )
    
    t = 0.; U_n1 = zeros(len(U_0))
    
    U = zeros([len(U_0), len(time_domain)])
    
    U[:,0] = U_0
    
    for i in range(0, len(time_domain)-1):
        
        # dt = round(time_domain[i+1] - time_domain[i], 14)
        
        # t = round(t + dt, 14)
        
        dt = time_domain[i+1] - time_domain[i]
        
        t = t + dt
        
        X = U[:,i]
        
        if Temporal_scheme == 'RK4':
            
            U_n1 = ts.RK4(X, t, dt, F)
            
        elif Temporal_scheme == 'Euler':
            
            U_n1 = ts.Euler(X, t, dt, F)
            
        elif Temporal_scheme == 'Crank-Nicolson':
            
            U_n1 = ts.Crank__Nicolson(X, t, dt, F)
            
        elif Temporal_scheme == 'Inverse Euler':
            
            U_n1 = ts.Inverse__Euler(X, t, dt, F)
            
        else:
            
            print('Introduce a valid Temporal scheme::\n\tEuler\n\tRK4\n\tCrank-Nicolson\n\tInverse Euler ')
            break
        
        U[:,i+1] = U_n1
            
    return U


def Newton_Raphson(Eq, x_i):
    
    eps = 1; iteration = 1
    
    while eps>1E-10 and iteration<1E3:
        
        Jacobian = Numeric_Jacobian(F = Eq, x = x_i)
        
        x_f = x_i - matmul( linalg.inv( Jacobian ), Eq(x_i) )
        
        iteration = iteration + 1
        
        eps = linalg.norm(x_f - x_i)
        
        x_i = x_f
    
    return x_f

def Numeric_Jacobian(F, x):
    '''
    

    Parameters
    ----------
    F : Function
        Vectorial function depending on x that is wanted to be solved.
    x : Array of floats
        Variable of F.

    Returns
    -------
    Jacobian : Matrix
        This matrix allows to compute the derivate of F.

    '''
    
    Jacobian = zeros([len(x), len(F(x))])
    
    for column in range(len(Jacobian[0,:])):
    
        dx = zeros(len(x))
        
        dx[column] = 1E-10
        
        Jacobian[:,column] = ( F(x+dx)  - F(x-dx) ) / linalg.norm( 2 * dx ) # Second order finite diferences aproximation

    return Jacobian

