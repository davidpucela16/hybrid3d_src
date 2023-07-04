#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:52:07 2023

@author: pdavid
"""

# Constants
cells_3D=10
n=10
shrink_factor=(cells_3D-1)/cells_3D
M_D=0.00001

import os
import sys

script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)
print(path_script)
name_script = script[script.rfind('/')+1:-3]
path_src = os.path.join(path_script, '../../src')
path_potentials = os.path.join(path_script, '../../Potentials')
sys.path.append(path_src)
sys.path.append(path_potentials)

#The folder where to save the results
path_output = os.path.join(path_src, '../../output_figures/' + name_script)

#Directory to save the assembled matrices and solution
path_matrices=os.path.join(path_output,"F{}_n{}".format(cells_3D, n))
#Directory to save the divided fiiles of the network
path_phi_bar=os.path.join(path_matrices, "E_portion")

os.makedirs(path_output, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
os.makedirs(path_phi_bar, exist_ok=True)
os.makedirs(os.path.join(path_matrices, "E_portion"), exist_ok=True)

var=True
user_input = input("Enter a value: ")
if user_input=="yes":
    var=True
    
    

import pandas as pd
from neighbourhood import GetNeighbourhood, GetUncommon
from hybridFast import hybrid_set_up
from GreenFast import GetSourcePotential
from mesh_1D import mesh_1D
from mesh import cart_mesh_3D
from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from Potentials_module import Gjerde
from Potentials_module import Classic
import matplotlib.pylab as pylab
import pdb
from scipy.sparse.linalg import bicg
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse import csc_matrix
import math
import scipy.sparse.linalg
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from PrePostTemp import Get1DCOMSOL, VisualizationTool
from PrePostTemp import GetCoarsekernels, GetProbArgs
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12,12),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'font.size': 24,
          'lines.linewidth': 2,
          'lines.markersize': 15}
pylab.rcParams.update(params)


#%%
BC_type = np.array(["Neumann", "Neumann", "Neumann",
                   "Neumann", "Neumann", "Neumann"])
#BC_type = np.array(["Dirichlet", "Dirichlet", "Neumann","Neumann", "Dirichlet", "Dirichlet"])
BC_value = np.array([0,0,0,0,0,0])
L_vessel = 240
L_3D = np.array([L_vessel, 3*L_vessel, L_vessel])

D = 1
K = np.array([0.0001, 1, 0.0001])
print(K)
U = np.array([1,1,1])
startVertex = np.array([0, 1, 2])
endVertex = np.array([1, 2, 3])
pos_vertex = np.array([[L_vessel/2, 0, L_vessel/2],
                       [L_vessel/2, L_vessel, L_vessel/2],
                       [L_vessel/2, 2*L_vessel, L_vessel/2],
                       [L_vessel/2, L_vessel*3, L_vessel/2]
                       ])
BCs_1D = np.array([[0, 1],
                   [3, 0]])  # Not relevant since we are gonna set the intra concentration ourselves

vertex_to_edge = [[0], [0, 1], [1, 2], [2]]

temp_cpv = 30
alpha = 50 # Aspect ratio
R_vessel = L_vessel/alpha
R_1D = np.zeros(3)+R_vessel
diameters = np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
h = L_vessel/temp_cpv

mesh = cart_mesh_3D(L_3D, cells_3D)
h = L_vessel/temp_cpv
net = mesh_1D(startVertex, endVertex, vertex_to_edge,
              pos_vertex, diameters, h+np.zeros(len(diameters)*temp_cpv), 1)
net.U = U
net.D = D
net.PositionalArraysFast(mesh)
mesh.AssemblyBoundaryVectors()


prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
mesh.GetOrderedConnectivityMatrix()
x=net.pos_s[:,1]
    
Cv=np.ones(3*temp_cpv)
#Cv=np.concatenate((np.ones(cells_per_vessel), np.arange(cells_per_vessel)[::-1]/cells_per_vessel, np.zeros(cells_per_vessel)))

if var:
    prob.I_assembly_bool=False
    prob.phi_bar_bool=False
else:
    prob.I_assembly_bool=True
    prob.phi_bar_bool=True

kernel_array, row_array,sources_array=GetCoarsekernels(GetProbArgs(prob), path_matrices)
Si_V=csc_matrix((kernel_array, (row_array, sources_array)), shape=(prob.F, prob.S))
sp.sparse.save_npz(os.path.join(path_matrices, 'Si_V'), Si_V)
prob.AssemblyProblem(path_matrices)
prob.A_matrix-=sp.sparse.diags(np.ones(prob.F), 0)*M_D*mesh.h**3
prob.B_matrix-=Si_V*M_D*mesh.h**3
    
Lin_matrix=prob.ReAssemblyMatrices()

sol=dir_solve(Lin_matrix, -prob.Full_ind_array)
s=sol[:prob.F]
q=sol[prob.F:-prob.S]
Cv=sol[-prob.S:]
prob.s=s
prob.q=q
prob.Cv=Cv


#%%
from post_processing import ReconstructionCoordinatesFast, GetProbArgs
file_path = os.path.join(path_output + '/COMSOL_data/traverse_phi.txt'.format(alpha))
t_phi, x_C = Get1DCOMSOL(file_path)
res=50
x_cyl=np.linspace(0,L_vessel, res)*0.99+0.005*L_vessel
crds=np.vstack((L_vessel/2+ np.zeros(res), 3*L_vessel/2*np.ones(res),  x_cyl)).T
phi_line=ReconstructionCoordinatesFast(crds, GetProbArgs(prob), s, q)

plt.plot(x_cyl, phi_line, '.-')
plt.plot(x_C, t_phi)
plt.ylabel("$\overline{\phi}$", rotation=0)
plt.xlabel("y")
plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
plt.show()

crds=np.vstack((x_cyl, 3*L_vessel/2*np.ones(res),  L_vessel/2+ np.zeros(res))).T
phi_line=ReconstructionCoordinatesFast(crds, GetProbArgs(prob), s, q)

plt.plot(x_cyl, phi_line, '.-')
plt.plot(x_C, t_phi)
plt.ylabel("$\overline{\phi}$", rotation=0)
plt.xlabel("y")
plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
plt.show()
#%%
from PrePostTemp import GetCoarsePhi
phi_coarse=GetCoarsePhi( prob.q, np.ones(prob.S), prob.s, GetProbArgs(prob))

slic=mesh.GetYSlice(L_vessel/2).reshape(cells_3D, cells_3D)
phi_coarse_middle=phi_coarse[0][slic[int(cells_3D/2)]]

plt.plot(mesh.x, phi_coarse_middle)

#%%
from PrePostTemp import VisualizationTool
res=50
prob.Cv=Cv
shrink_factor=((cells_3D-1)/cells_3D)
corners_2D=np.array([[0,0],[0,1],[1,0],[1,1]])*L_3D[0]*shrink_factor+L_3D[0]*(1/cells_3D/2)
aa=VisualizationTool(prob, 1,0,2, corners_2D, res)
shrink_factor_perp=((mesh.cells[2]-1)/mesh.cells[2])
aa.GetPlaneData(path_output)
aa.PlotData(path_output)


