#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:34:21 2020

@author: kissami
"""
import numpy as np
from mpi4py import MPI
from psydac.ddm.partition import mpi_compute_dims

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
import os


comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    #removing existing vtk files
    mypath = "results"
    if not os.path.exists(mypath):
        os.mkdir(mypath)
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))


nb_neighbours = 4
N = 0; E = 1; S = 2; W = 3

neighbour = np.zeros(nb_neighbours, dtype=np.int8)
ntx = 101
nty = 101

Nx = ntx+2
Ny = nty+2

npoints  =  [ntx, nty]
p1 = [2,2]
P1 = [False, False]
reorder = True


coef = np.zeros(3)
''' Grid spacing '''
hx = 1/(ntx+1.);
hy = 1/(nty+1.);

''' Equation Coefficients '''
coef[0] = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy);
coef[1] = 1./(hx*hx);
coef[2] = 1./(hy*hy);

def create_2d_cart(npoints, p1, P1, reorder):
    
    # Store input arguments                                                                                                                                                                                                                                               
    npts    = tuple(npoints)
    pads    = tuple(p1)
    periods = tuple(P1)
    reorder = reorder
    
    nprocs, block_shape = mpi_compute_dims(nb_procs, npts, pads )
    
    dims = nprocs
    
    if (rank == 0):
        print("Execution poisson with",nb_procs," MPI processes\n"
               "Size of the domain : ntx=",npoints[0], " nty=",npoints[1],"\n"
               "Dimension for the topology :",dims[0]," along x", dims[1]," along y\n"
               "-----------------------------------------")  
    
    cart2d = comm.Create_cart(
            dims    = dims,
            periods = periods,
            reorder = reorder
            )
    
    return dims, cart2d

def create_2dCoords(cart2d, npoints, dims):

    coord2d = cart2d.Get_coords(rank)
    
    ''' Computation of the local grid boundary coordinates (global indexes) '''
    sx = int((coord2d[0]*npoints[0])/dims[0]+1);
    ex = int(((coord2d[0]+1)*npoints[0])/dims[0]);
    
    sy = int((coord2d[1]*npoints[1])/dims[1]+1);
    ey = int(((coord2d[1]+1)*npoints[1])/dims[1]);

    print("Rank in the topology :",rank," Local Grid Index :", sx, " to ",ex," along x, ",
          sy, " to", ey," along y")
    
    return sx, ex, sy, ey

def create_neighbours(cart2d):

    neighbour[N],neighbour[S] = cart2d.Shift(direction=0,disp=1)
    neighbour[W],neighbour[E] = cart2d.Shift(direction=1,disp=1)
    
    print("Process", rank," neighbour: N", neighbour[N]," E",neighbour[E] ," S ",neighbour[S]," W",neighbour[W])
    
    return neighbour

def create_derived_type(sx, ex, sy, ey):
    type_ligne = MPI.DOUBLE.Create_contiguous(ey-sy+1)
    type_ligne.Commit()
    
    type_column = MPI.DOUBLE.Create_vector(ex-sx+1, 1, ey-sy+3)
    type_column.Commit()
     
    return type_ligne, type_column

def communications(u, sx, ex, sy, ey, type_column, type_ligne):
    
    comm.Send([u[IDX(sx, sy):],1,type_ligne], dest=neighbour[N])
    comm.Recv([u[IDX(ex+1, sy):],1,type_ligne],source=neighbour[S])
    
    comm.Send([u[IDX(ex, sy):],1,type_ligne], dest=neighbour[S])
    comm.Recv([u[IDX(sx-1, sy):],1,type_ligne],source=neighbour[N])
    
    comm.Send([u[IDX(sx, sy):],1,type_column], dest=neighbour[W])
    comm.Recv([u[IDX(sx, ey+1):],1,type_column],source=neighbour[E])
    
    comm.Send([u[IDX(sx, ey):],1,type_column], dest=neighbour[E])
    comm.Recv([u[IDX(sx, sy-1):],1,type_column],source=neighbour[W])

from scipy import interpolate

def interpolate_u(u, sx, ex, sy, ey, points):

    Nxloc = ex-sx+3
    Nyloc = ey-sy+3
    
    x = np.linspace(0, 1, Nxloc)
    y = np.linspace(0, 1, Nyloc)
    
    print(len(x), len(y))
    print(len(u))
    
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    
    print(grid_x)
    
    f = interpolate.interp2d(x, y, u, kind='cubic')
    
    unode = np.zeros(len(points))
    for i in range(len(points)):
        unode[i] = f(points[i][0], points[i][1])

    return unode
#@njit    
def define_points_cells(sx, ex, sy, ey):
    ''' Grid spacing in each dimension'''
    ''' Solution u and u_new at the n and n+1 iterations '''
    
    Nxloc = ex-sx+1
    Nyloc = ey-sy+1
    
    ''' Grid spacing '''
    hx = 1/(ntx);
    hy = 1/(nty);
    
    points = []
    for iterx in range(sx, ex+2, 1):
        for itery in range(sy, ey+2, 1):
            x = (iterx-1)*hx
            y = (itery-1)*hy
            points.append([x,y, 0.])
 
    it = 0
    cells = []
    for i in range(0, Nyloc, 1):
        for j in range(0, Nxloc, 1):
            k=i+j*(Nyloc+1);
            l = i+(j+1)*(Nyloc+1);
            
            cells.append([k, k+1, l+1, l])
    
            it=it+1;
            
    return points, cells

@njit    
def IDX(i, j):
    return ( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) )

@njit    
def initialization(sx, ex, sy, ey):
    ''' Grid spacing in each dimension'''
    ''' Solution u and u_new at the n and n+1 iterations '''
    
    SIZE = (ex-sx+3) * (ey-sy+3)
    
    u       = np.zeros(SIZE)
    u_new   = np.zeros(SIZE)
    f       = np.zeros(SIZE)
    u_exact = np.zeros(SIZE)
    
    for iterx in range(sx, ex+1, 1):
        for itery in range(sy, ey+1, 1):
            x = iterx*hx
            y = itery*hy
            
            f[IDX(iterx, itery)] = 2*(x*x-x+y*y-y)
            u_exact[IDX(iterx, itery)] = x*y*(x-1)*(y-1)
            u[IDX(iterx, itery)] = 1

    return u, u_new, u_exact, f

#@njit
def computation(u, u_new):
    for iterx in range(sx, ex+1, 1):
        for itery in range(sy, ey+1, 1):
            u_new[IDX(iterx, itery)] = coef[0]* (  coef[1]*(u[IDX(iterx+1, itery)]+u[IDX(iterx-1, itery)])
            + coef[2]*(u[IDX(iterx, itery+1)]+u[IDX(iterx, itery-1)]) - f[IDX(iterx, itery)]);
            
 
#@njit
def output_results(u, u_exact):
    
    print("Exact Solution u_exact - Computed Solution u - difference")
    for itery in range(sy, ey+1, 1):
        print(u_exact[IDX(1, itery)], '-', u[IDX(1, itery)], u_exact[IDX(1, itery)]-u[IDX(1, itery)] );

#@njit
def global_error(u, u_new):
    
    local_error = 0
     
    for iterx in range(sx, ex+1, 1):
        for itery in range(sy, ey+1, 1):
            temp = np.fabs( u[IDX(iterx, itery)] - u_new[IDX(iterx, itery)]  )
            if local_error < temp:
                local_error = temp;
    
    return local_error

import meshio
def paraview_plot(u, u_exact, points, cells):
    
    cells = np.array(cells, dtype=np.int)
    points = np.array(points, dtype=np.double)
    
    elements = {"quad": cells}
    u = np.array(u, dtype=np.double)
    
    data = {'u':u}
    
    meshio.write_points_cells("visu"+str(rank)+".vtu", 
                              points, elements, point_data=data, file_format="vtu")
    
    
    if(rank == 0 and nb_procs > 1):
        with open("results/visu.pvtu", "a") as text_file:
            text_file.write("<?xml version=\"1.0\"?>\n")
            text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
            text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
            text_file.write("<PPoints>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
            text_file.write("</PPoints>\n")
            text_file.write("<PCells>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"offsets\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"types\" format=\"binary\"/>\n")
            text_file.write("</PCells>\n")
          
            text_file.write("<PPointData Scalars=\"h\">\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"u\" format=\"binary\"/>\n")
            text_file.write("</PPointData>\n")
            for i in range(nb_procs):
                name1 = "visu"
                bu1 = [5]
                bu1 = str(i)
                name1 += bu1
                name1 += ".vtu"
                text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
            text_file.write("</PUnstructuredGrid>\n")
            text_file.write("</VTKFile>")

def plot_2d(f):

    f = np.reshape(f, (ex-sx+3, ey-sy+3))
    
    x = np.linspace(0, 1, ey-sy+3)
    y = np.linspace(0, 1, ex-sx+3)
    
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.gca(projection='3d')                      
    X, Y = np.meshgrid(x, y)      

    ax.plot_surface(X, Y, f, cmap=cm.viridis)
    
    plt.show()

dims, cart2d   = create_2d_cart(npoints, p1, P1, reorder)
neighbour      = create_neighbours(cart2d)

sx, ex, sy, ey = create_2dCoords(cart2d, npoints, dims)

type_ligne, type_column = create_derived_type(sx, ex, sy, ey)
u, u_new, u_exact, f             = initialization(sx, ex, sy, ey)

''' Time stepping '''
it = 0
convergence = False
it_max = 100000
eps = 2.e-16

''' Elapsed time '''
t1 = MPI.Wtime()

#import sys; sys.exit()
while (not(convergence) and (it < it_max)):
    it = it+1;

    temp = u.copy() 
    u = u_new.copy() 
    u_new = temp.copy()
    
    ''' Exchange of the interfaces at the n iteration '''
    communications(u, sx, ex, sy, ey, type_column, type_ligne)
   
    ''' Computation of u at the n+1 iteration '''
    computation(u, u_new)
    
    ''' Computation of the global error '''
    local_error = global_error(u, u_new);
    diffnorm = comm.allreduce(np.array(local_error), op=MPI.MAX )   
   
    ''' Stop if we obtained the machine precision '''
    convergence = (diffnorm < eps)
    
    ''' Print diffnorm for process 0 '''
    if ((rank == 0) and ((it % 100) == 0)):
        print("Iteration", it, " global_error = ", diffnorm);
        
''' Elapsed time '''
t2 = MPI.Wtime()

if (rank == 0):
    ''' Print convergence time for process 0 '''
    print("Convergence after",it, 'iterations in', t2-t1,'secs')

    ''' Compare to the exact solution on process 0 '''
    output_results(u, u_exact)

if rank == 0:
    plot_2d(u)

#points, cells = define_points_cells(sx, ex, sy, ey)
#
#unode = interpolate_u(u, sx, ex, sy, ey, points)
#
#paraview_plot(unode, u_exact, points, cells)