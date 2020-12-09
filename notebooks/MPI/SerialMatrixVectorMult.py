#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:26:19 2020

@author: kissami
"""
from scipy.sparse import csc_matrix
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
import timeit

SIZE = 10000

A = lil_matrix((SIZE, SIZE))
A[0, :1000] = rand(1000)
A[1, 1000:2000] = A[0, :1000]
A.setdiag(rand(SIZE))

A = A.tocsr()
b = rand(SIZE)

start = timeit.default_timer()
x = spsolve(A, b)
stop = timeit.default_timer()
print("spsolve ", (stop - start)*1000,"ms")

start = timeit.default_timer()
x_ = solve(A.toarray(), b)
stop = timeit.default_timer()
print("solve ", (stop - start)*1000,"ms")

err = norm(x-x_)
print(err < 1e-10)

row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csc_matrix((data, (row, col)), shape=(3, 3)).toarray()


indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
A = csc_matrix((data, indices, indptr), shape=(3, 3))
b = rand(3)
x = spsolve(A, b)


x_ = solve(A.toarray(), b)

err = norm(x-x_)
print(err < 1e-10)