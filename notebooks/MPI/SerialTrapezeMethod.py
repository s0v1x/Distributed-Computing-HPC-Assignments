from mpi4py import MPI
import numpy as np 
import time

def compute_integrale_trapeze(x, y, nbi):

    integrale = 0.
    for i in range(nbi):
        trap = (x[i+1]-x[i])/2 * (y[i]+y[i+1])
        integrale = integrale + trap
                
    return integrale

comm = MPI.COMM_WORLD
nb_proc = comm.Get_size()
rank = comm.Get_rank()

xmin = 0
xmax = 3*np.pi/2
nbx = 20
nbr = int(nbx/nb_proc)

X = np.linspace(xmin, xmax, nbx)
if rank == nb_proc-1 :
    sub_X = X[nbr*rank : ]
else :
    sub_X = X[nbr*rank : nbr*(rank+1)+1]
y = np.cos(sub_X)

result = compute_integrale_trapeze(sub_X, y, len(sub_X)-1)

print("Process {rank}, working on {sub}".format(rank = rank, sub = sub_X))

total = comm.reduce(result, op=MPI.SUM, root=0)

if total is not None:
    print('---------------------------------------------')
    print('Integral = ', total)
print('---------------------------------------------')