import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numba import njit
from mpi4py import MPI



COMM = MPI.COMM_WORLD
nbporc = COMM.Get_size()
RANK = COMM.Get_rank()

seed(42)

def matrixVectorMult(A, b, x):

    row, col = A.shape
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]

    return 0

#matrix sizes
SIZE = 1000
localS = int(SIZE/nbporc)

# counts = block of each proc
counts =[localS*SIZE for i in range(nbporc)]

if RANK == 0:
    A = lil_matrix((SIZE, SIZE))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(SIZE))
    A = A.toarray()
    b = rand(SIZE)
else :
    A = None
    b = None



b = COMM.bcast(b, root=0)
LocalMatrix = np.zeros((localS, SIZE))
# Scatter the matrix A
COMM.Scatterv([A, counts, MPI.DOUBLE], LocalMatrix, root = 0)

locX = np.zeros(localS)

matrixVectorMult(LocalMatrix, b, locX)


sendcounts =np.array([localS for i in range(nbporc)])
if RANK == 0: 
    X = np.empty(sum(sendcounts), dtype=np.double)
else :
    X = None

COMM.Gatherv(locX, (X, sendcounts, MPI.DOUBLE), root=0)


if RANK == 0 :

    x_tmp = A.dot(b)
    print("Le result de A*b est :", np.max(x_tmp - X))

