#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:47:23 2023

@author: pdavid
"""
import sys
import os 
path=os.path.dirname(__file__)
path_src=os.path.join(path, '../src_final')
path_figures=os.path.join(path, "/figures")
from Potentials_module import Classic, Alternatives
sys.path.append(path_src)


import numpy as np
from assembly_1D import AssemblyTransport1D
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg
import math

import pdb

import matplotlib.pylab as pylab
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)



#%% - This is the validation of the 1D transport eq without reaction

alpha=100

U = 0.2
D = 2
K=1
L = 960
R=0.1
cells_1D = 20

#%% - 

from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
import matplotlib.pyplot as plt

import math

from mesh_1D import mesh_1D
from GreenFast import GetSourcePotential
import pdb

from hybridFast import hybrid_set_up

from neighbourhood import GetNeighbourhood, GetUncommon
#%


BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

cells=10
n=20
L=np.array([1,1,1])
mesh=cart_mesh_3D(L,cells)

mesh.AssemblyBoundaryVectors()


#%%

startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L[0]/2, 0.01, L[0]/2],[L[0]/2, L[1]-0.01,L[0]/2]])
vertex_to_edge=[[0],[0]]
diameters=np.array([2*R])
h=np.array([L[0]])/cells_1D

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)

#%%
prob=hybrid_set_up(mesh, net, BC_type, BC_value, n, 1, np.zeros(len(diameters))+K, np.array([[0,1],[1,0]]))

mesh.GetOrderedConnectivityMatrix()
prob.AssemblyProblem('/home/pdavid/Bureau/Code/BMF_Code/kk')



#%%

sol=dir_solve(prob.Full_linear_matrix, -prob.Full_ind_array)

prob.s=sol[:prob.F]
prob.q=sol[prob.F:-prob.S]
prob.Cv=sol[-prob.S:]

plt.plot(prob.q)
plt.show()

#%%


#%%

f = Classic(net.L,R)
G=f.get_single_layer_vessel(len(net.pos_s)) 
H=f.get_double_layer_vessel(len(net.pos_s))  

E=prob.E_matrix.toarray()-np.identity(len(net.pos_s))
    



