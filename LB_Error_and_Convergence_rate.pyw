# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:33:50 2022

@author: rafra
"""

from numpy import zeros, linspace, log10, append
from numpy.linalg import norm

import LB_Math_Functions as mth

import matplotlib.pyplot as plt

from matplotlib import rc # LaTeX tipography
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text', usetex=True); plt.rc('font', family='serif')

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

'''--------------------------------------------------------'''
'''--------------------------------------------------------'''
'''--------------------------------------------------------'''

def Richardson_Error_Extrapolation(Differential_operator, Initial_conditions, time_domain, temporal_scheme, Save=False):
    
    if temporal_scheme == "Euler":
        scheme_order = 1
    elif temporal_scheme == "Inverse Euler":
        scheme_order = 1
    elif temporal_scheme == "RK4":
        scheme_order = 4
    elif temporal_scheme == "Crank-Nicolson":
        scheme_order = 2
    
    t1 = time_domain; dt = t1[1]-t1[0]; tf = t1[-1]
    t2 = linspace(t1[0], t1[-1], 2*(len(t1)-1)+1)
    
    X_1 = mth.Cauchy_Problem( Differential_operator, Initial_conditions, t1, Temporal_scheme = temporal_scheme )
    X_2 = mth.Cauchy_Problem( Differential_operator, Initial_conditions, t2, Temporal_scheme = temporal_scheme )

    # Richardson_Error = np.zeros([len(Initial_conditions), len(t1)])
    
    Richardson_Error = zeros(len(t1))

    for i in range(0,len(t1)):
        
        Richardson_Error[i] = norm( X_2[:,2*i] - X_1[:,i] ) / ( 1- ( 1 / 2**scheme_order ) )
    
    
    x = t1
    y = Richardson_Error

    fig, ax = plt.subplots(1,1, figsize=(7,7), constrained_layout='true')
    ax.set_title('Error of '+temporal_scheme+r' for $t_f$ = '+str(tf), fontsize=20)
    ax.grid()
    ax.set_xlabel(r'$t$',fontsize=20)
    ax.set_ylabel(r'$|E|$',fontsize=20)
    ax.plot( x, y, c='b', label=r'$\Delta t$ = '+str(dt))
    plt.show()
    
    if Save == True:
        
        fig.savefig('Plots\HITO_3_Richardson_'+temporal_scheme+'_tf='+str(tf)+'.pdf', transparent = True, bbox_inches="tight")
        
    return Richardson_Error


def model_function(x, a, b):
    return a*x + b

def Convergence_Rate(Differential_operator, Initial_conditions, tf, temporal_scheme, M, Adjust = False, Save = False):
    
    t = {}; X = {}; log_DU = {}; log_N = {} # Dictionaries initialization
    
    if temporal_scheme == "Euler":
        dt = 0.01
    elif temporal_scheme == "Inverse Euler":
        dt = 0.01
    elif temporal_scheme == "RK4":
        dt = 0.1
    elif temporal_scheme == "Crank-Nicolson":
        dt = 0.1
    
    '''---------------- Computing based on Richardson extrapolation ----------------'''
    for i in range(M):
        
        if i == 0:
            t[i] = linspace(0, tf, int(tf/dt)+1)
        else:
            t[i] = linspace(t[0][0], t[0][-1], 2**i*(len(t[0])-1)+1)
        
        X[i] = mth.Cauchy_Problem( Differential_operator, Initial_conditions, t[i], Temporal_scheme = temporal_scheme )
        
        print(i)


    key_list = list(X.keys())

    for j in range( len(key_list)-1 ):
         
        log_DU[j] = log10( norm( X[key_list[j+1]][:,-1] - X[key_list[j]][:,-1] ) ) # DU = ||Un - U2n||
        
        log_N[j] = log10( len( t[key_list[j]] ) )



    '''----------------PLotting----------------'''
    x, y = [], []

    for key in log_DU:
        
        x = append(x, log_N[key])
        y = append(y, log_DU[key])
    
    if Adjust == True:
        
        [a_model, b_model], covariance_matrix = curve_fit(model_function, x, y, p0=[-1.5, -0.25])

        R_2 = r2_score(y, model_function(x, a_model, b_model))
        
        textstring = r'$\mbox{R}^2$ = ' + str(round(R_2,3))


    fig, ax = plt.subplots(1,1, figsize=(7,7), constrained_layout='true')
    # ax.set_xlim(-1.25,1.25)
    # ax.set_ylim(-1.25,1.25)
    ax.set_title('Convergence of temporal schemes: '+temporal_scheme+r' for $t_f$ = '+str(tf), fontsize=20)
    ax.grid()
    ax.set_xlabel(r'$\log(N)$',fontsize=20)
    ax.set_ylabel(r'$\log(U_1^n - U_2^{2n})$',fontsize=20)

    plt.scatter(x, y, c='r', marker="+", label='Numeric Results')

    if Adjust == True:
        
        plt.plot(linspace(x[0],x[-1]),model_function(linspace(x[0],x[-1]), a_model, b_model),c='k',alpha=0.8, label='Lineal         model:: '+r'$q$ = '+str(round(-a_model,3)))  

        ax.text(0.8, 0.8, textstring, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'), size=18)

    plt.show()
    ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16) 

    if Save == True:
        fig.savefig('Plots\HITO_3'+temporal_scheme+'_M=_'+str(M)+'_tf='+str(tf)+'.pdf', transparent = True, bbox_inches="tight")
