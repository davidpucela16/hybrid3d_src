#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:52:07 2023

@author: pdavid
"""

import os 
path=os.path.dirname(__file__)
os.chdir(path)
from Potentials_module import Classic
from Potentials_module import Gjerde
path_src=os.path.join(path, '../src_final')
os.chdir(path_src)

path_matrices='/home/pdavid/Bureau/Code/BMF_Code/kk/One_vessel_analysis_self'
path_phi_bar='/home/pdavid/Bureau/Code/BMF_Code/kk/One_vessel_analysis_self/phi_bar'
os.makedirs(path_phi_bar, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
import numpy as np

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



BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])
L_vessel=240
cells_3D=11
n=11
L_3D=np.array([L_vessel, 3*L_vessel, L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.AssemblyBoundaryVectors()


#%%
# - This is the validation of the 1D transport eq without reaction

D = 1
K=np.array([0.0001,2,0.0001])*4

U = np.array([2,2,2])/L_vessel*10
startVertex=np.array([0,1,2])
endVertex=np.array([1,2,3])
pos_vertex=np.array([[L_vessel/2, 0, L_vessel/2],
                     [L_vessel/2, L_vessel,L_vessel/2],
                     [L_vessel/2, 2*L_vessel, L_vessel/2],
                     [L_vessel/2, L_vessel*3,L_vessel/2]
                     ])
BCs_1D=np.array([[0,1],
                 [3,0]]) #Not relevant since we are gonna set the intra concentration ourselves

vertex_to_edge=[[0],[0,1], [1,2], [2]]

temp_cpv=1000
alpha=12 #Aspect ratio
R_vessel=L_vessel/alpha
R_1D=np.zeros(3)+R_vessel
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
h=L_vessel/temp_cpv
    
net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)
prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
mesh.GetOrderedConnectivityMatrix()
x=net.pos_s[:,1]
    
Cv=np.ones(3*temp_cpv)
#Cv=np.concatenate((np.ones(cells_per_vessel), np.arange(cells_per_vessel)[::-1]/cells_per_vessel, np.zeros(cells_per_vessel)))
    
prob.AssemblyDEFFast(path_phi_bar, path_matrices)
prob.AssemblyABC(path_matrices)
Lin_matrix_1D=sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)), 
                                 sp.sparse.hstack((prob.D_matrix, prob.q_portion+prob.Gij))))
    
b=np.concatenate((-prob.I_ind_array, np.zeros(len(net.pos_s)) - prob.F_matrix.dot(Cv)))
sol_line=dir_solve(Lin_matrix_1D, b)
q_line=sol_line[mesh.size_mesh:]
s_line=sol_line[:mesh.size_mesh]

#Now we are gonna solve the same problem but using the elliptic integrals for the single layer 
P=Classic(3*L_vessel, R_vessel)
G_ij=P.get_single_layer_vessel(len(net.pos_s))/2/np.pi/R_vessel
#The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
new_E_matrix=G_ij+prob.q_portion
Lin_matrix_2D=sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)), 
                                 sp.sparse.hstack((prob.D_matrix, new_E_matrix))))
sol_2D=dir_solve(Lin_matrix_2D, b)
prob.q=sol_2D[mesh.size_mesh:]
prob.s=sol_2D[:mesh.size_mesh]
q_exact=sol_2D[mesh.size_mesh:]
x_exact=net.pos_s[:,1]

plt.plot(x_exact, q_exact)
plt.plot(x_exact, q_line)
    