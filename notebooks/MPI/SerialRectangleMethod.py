from mpi4py import MPI
import numpy as np 
import matplotlib.pyplot as plt

def compute_integrale_rectangle(x, y, nbi):
    
    integrale =0.
    for i in range(nbi):
        integrale = integrale + y[i]*(x[i+1]-x[i])
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

start = MPI.Wtime()
result = compute_integrale_rectangle(sub_X, y, len(sub_X)-1)
end = MPI.Wtime()

print("Process {rank}".format(rank = rank))
print("Done in %.8f" % (end-start))

total = comm.reduce(result, op=MPI.SUM, root=0)

if total is not None:
    print('---------------------------------------------')
    print('Integral = ', total)
print('---------------------------------------------')
