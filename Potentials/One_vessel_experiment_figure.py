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

path_matrices='/home/pdavid/Bureau/Code/BMF_Code/test_data/One_vessel_figure'
path_phi_bar=os.path.join(path_matrices, 'phi_bar')
path_data=os.path.join(path_matrices, 'data')
os.makedirs(path_data, exist_ok=True)
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
from PrePostTemp import VisualizationTool


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
K=np.array([0.0001,1,0.0001])*4

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

alpha=20
c=0
temp_cpv=100
#alpha=20 #Aspect ratio
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

prob.AssemblyProblem(path_matrices)

prob.SolveProblem()

shrink_factor=(cells_3D-1)/cells_3D
resolution=100
corners_2D=np.array([[0,0],[0,L_3D[0]],[L_3D[0],0],[L_3D[0],L_3D[0]]])*shrink_factor + L_3D[0]*(1-shrink_factor)/2
data_vis=VisualizationTool(prob,1, 0, 2, corners_2D, 100)
data_vis.GetPlaneData(path_data)
data_vis.PlotData(path_data)

data_vis.GetVolumeData(1, 0, resolution, path_data, shrink_factor)










