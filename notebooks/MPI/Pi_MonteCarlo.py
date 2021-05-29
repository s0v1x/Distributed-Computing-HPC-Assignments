import random 
import timeit
from mpi4py import MPI


def serial_compute_points(a):

    circle_points= 0

    for i in range(a): 

        rand_x= random.uniform(-1, 1) 
        rand_y= random.uniform(-1, 1)
      
        # Distance between (x, y) from the origin 
        origin_dist= rand_x**2 + rand_y**2
        # Checking if (x, y) lies inside the circle 
        if origin_dist<= 1: 
            circle_points+= 1

    return circle_points

comm = MPI.COMM_WORLD
nb_proc = comm.Get_size()
rank = comm.Get_rank()

INTERVAL= 1000**2

ind = int((INTERVAL)/nb_proc)
if rank == nb_proc-1 :
    sub_inter = ((INTERVAL) - (ind*rank))
    print()
else :
    sub_inter = ind
    print()

circle_points = serial_compute_points(sub_inter) #785061
print('Process ', rank)
print("Circle points number :",circle_points)

total = comm.reduce(circle_points, op=MPI.SUM, root=1)

if total is not None:
    print('---------------------------------------------')
    print('Total Circle points number : ', total)
print('---------------------------------------------')