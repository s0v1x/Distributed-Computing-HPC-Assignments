#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:10:56 2020

@author: kissami
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_integrale_trapeze(x, y, nbi):

    integrale = 0.
    for i in range(nbi):
        trap = (x[i+1]-x[i])/2 * (y[i]+y[i+1])
        integrale = integrale + trap
                
    return integrale

def plot_integrale(x, y, nbi):
    for i in range(nbi):
        # dessin du rectangle
        x_trap = [x[i], x[i], x[i+1], x[i+1], x[i]] # abscisses des sommets
        y_trap = [0   , y[i], y[i+1],      0,        0   ] # ordonnees des sommets
        plt.plot(x_trap, y_trap,"r")
    
    return 0
    
xmin = 0
xmax = 3*np.pi/2
nbx = 20
nbi = nbx - 1 # nombre d'intervalles

x = np.linspace(xmin, xmax, nbx)
y = np.cos(x)
plt.plot(x,y,"bo-")


integrale = compute_integrale_trapeze(x, y, nbi)
plot_integrale(x, y, nbi)   
    
print("integrale =", integrale)

plt.show()