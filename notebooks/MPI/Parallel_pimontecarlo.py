#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:19:39 2020

@author: kissami
"""
import numpy as np
from mpi4py import MPI
import random 

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

INTERVAL= 1000

def parallel_compute_points():

    circle_points = 0
    random.seed(42)  
    
    for i in range(RANK, INTERVAL**2, SIZE):
        # Randomly generated x and y values from a 
        # uniform distribution 
        # Rannge of x and y values is -1 to 1 
        rand_x= random.uniform(-1, 1) 
        rand_y= random.uniform(-1, 1) 
      
        # Distance between (x, y) from the origin 
        origin_dist= rand_x**2 + rand_y**2
    
        # Checking if (x, y) lies inside the circle 
        if origin_dist<= 1.: 
            circle_points+= 1
        
    return circle_points


start = MPI.Wtime()
circle_points = parallel_compute_points()
     
circle_points = np.array(circle_points, 'd')
sum_circle_points = np.zeros(1)

COMM.Reduce(circle_points, sum_circle_points, MPI.SUM, 0)
end = MPI.Wtime()

if RANK == 0:
    print("Circle points number :",sum_circle_points)
    pi = 4* sum_circle_points/ INTERVAL**2
    print("Final Estimation of Pi=", pi, "cpu time :",end-start)
        
