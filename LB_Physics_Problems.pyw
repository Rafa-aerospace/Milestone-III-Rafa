"""
Created on Thu Oct 13 16:22:30 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import array

def Problem_Assignment(problem,Physics_Problems_available):
    
    if problem == Physics_Problems_available[0]:
        
        return Kepler_Orbits_2N
        
    elif problem == Physics_Problems_available[1]:
        
        return Undamped_Armonic_Oscilator
        
    else:
        print("Introduce a valid problem equation to solve\n\t", Physics_Problems_available )
        return "ERROR"

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
    return array([X[2], X[3], -X[0]/(X[0]**2 + X[1]**2)**(3/2), -X[1]/(X[0]**2 + X[1]**2)**(3/2)])


def Undamped_Armonic_Oscilator(X, t):
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
    return array([X[1], -X[0]])

