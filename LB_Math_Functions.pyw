# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:03:25 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import zeros, linalg, matmul, size, array, dot
import LB_Temporal_Schemes as ts # User's module


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
        
        #x_f = x_i - matmul( linalg.inv( Jacobian ), Eq(x_i) )
        x_f = x_i - matmul( Inverse( Jacobian ), Eq(x_i) )
        
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



def LU_Factorization(A):

	N = size(A,1)
	U = zeros([N,N]); L = zeros([N,N])

	U[0,:] = A[0,:]

	for i in range(0,N):

		L[i,i] = 1

	L[1:N,0] = A[1:N,0]/U[0,0]


	for i in range(1,N):

		for j in range(i,N):

			U[i,j] = A[i,j] - dot(L[i,0:i], U[0:i,j])

		for k in range(i+1,N):

			L[k,i] =(A[k,i] - dot(U[0:i,i], L[k,0:i])) / (U[i,i])

	return [L@U, L, U]


def LU_Solve(Matrix,vector):

	N=size(vector)
	y=zeros(N); x=zeros(N)

	[A,L,U] = LU_Factorization(Matrix)

	y[0] = vector[0]

	for i in range(0,N):

		y[i] = vector[i] - dot(A[i,0:i], y[0:i])
		

	x[N-1] = y[N-1]/A[N-1,N-1]

	for i in range(N-2,-1,-1):

		x[i] = ( y[i] - dot( A[i, i+1:N+1], x[i+1:N+1] ) ) / A[i,i]
		
	return x


def Inverse(M):

	N = size(M,1)

	B = zeros([N,N])

	for i in range(0,N):

		aux = zeros(N)

		aux[i] = 1

		B[:,i] = LU_Solve(M, aux)

	return B

