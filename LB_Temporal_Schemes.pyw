# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 23:46:33 2022

@author: Rafael Rivero de Nicolás
"""


# import numpy as np
import LB_Math_Functions as mth # User's Module

from scipy.optimize import fsolve

# %% Functions Definitions


def Euler(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * Function(U_(n),t).

    '''
    
    U_n1 = X + dt*Function(X,t)
    
    return U_n1


def Inverse__Euler(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: X = U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * Function(U_(n+1),t+dt).

    '''
    
    def Inverse__Euler__Operator(U_n1):
        
        return U_n1 - X - dt * Function(U_n1, t)
    
    U_n1 = mth.Newton_Raphson(Inverse__Euler__Operator, x_i=X) # Rafa's Function
    
    # U_n1 = fsolve(Inverse__Euler__Operator, X)
    
    return U_n1


def RK4(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6.

    '''
   
    k1 = Function( X, t )
    
    k2 = Function( X + dt * k1/2, t + dt/2 )
    
    k3 = Function( X + dt * k2/2, t + dt/2 )
    
    k4 = Function( X + dt *k3,    t + dt   )
    
    U_n1 = X + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6
    
    return U_n1


def Crank__Nicolson(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = X + dt/2 * ( Function(X,t) + Function(U_(n+1), t+dt)).

    '''
    
    def  Crank_Nicolson_Operator(U_n1):
        return  U_n1 - X - dt/2 * ( Function(X,t) + Function(U_n1,t+dt) )
    
    # U_n1 = fsolve(Crank_Nicolson_Operator, X) # Scipy Function
    
    U_n1 = mth.Newton_Raphson(Crank_Nicolson_Operator, x_i=X) # Rafa's Function
    
    return U_n1



