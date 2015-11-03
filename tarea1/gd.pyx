#cython: cdivision=True 
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.math cimport abs


ctypedef unsigned int uint
ctypedef cnp.float64_t float64_t
ctypedef cnp.uint8_t uint8_t
ctypedef cnp.ndarray ndarray


#Dot product: vector-vector
cdef float dot1(float64_t[::1] x, float64_t[::1] y):
    cdef:
        float r
        Py_ssize_t i,M
    M = x.shape[0]
    r = 0.
    for i in range(M):
        r += x[i]*y[i]
    return r


#Dot product: matrix-vector
cdef ndarray dot2(float64_t[:,::1] X, float64_t[::1] y):
    cdef:
        ndarray[float64_t, ndim=1] r
        Py_ssize_t i, j, M, N
    M = X.shape[0]
    N = X.shape[1]
    r = np.zeros(M)
    for i in range(M):
        r[i] = 0.
        for j in range(N):
            r[i] += X[i,j]*y[j]
    return r



#Overall cost function for linear regresion
cdef float J(ndarray[float64_t, ndim=2] X, ndarray[float64_t, ndim=1] y, ndarray[float64_t, ndim=1] beta):
    cdef:
        ndarray[float64_t, ndim=1] diff
    diff = dot2(X,beta)-y
    return 0.5*dot1(diff,diff)


#Online gradient descent for linear regression
cdef ndarray _gd_online(ndarray[float64_t, ndim=2] X, ndarray[float64_t, ndim=1] y, float alpha, float eps, uint max_iter):
    cdef:
        Py_ssize_t i, m, M, N
        float J0, J1
        ndarray[float64_t, ndim=1] beta
    M = X.shape[0]
    N = X.shape[1]
    beta = np.zeros(N)
    J1 = J(X,y,beta) #loss at previous iteration
    for i in range(max_iter):
        J0 = J1
        for m in range(M):
            beta -= alpha*(dot1(X[m],beta)-y[m])*X[m]
        J1 = J(X,y,beta)
        if abs(J1-J0)/J0 < eps: break
    return beta


"""
Wraper Functions
"""
def gd_online(X, y, alpha, eps=1e-5, max_iter=100000):
    return _gd_online(X, y, alpha, eps, max_iter)
