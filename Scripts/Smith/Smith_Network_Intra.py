#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:04:46 2023

@author: pdavid
"""

# Constants
factor_flow=1
factor_K=10
cells_3D=20
n=3
shrink_factor=(cells_3D-1)/cells_3D
Network=1
gradient="x"
M_D=0.001

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

#Network
path_network=os.path.join(path_src, "../../networks/Synthetic_ROIs_300x300x300")
filename=os.path.join(path_network,"Rea{}_synthetic_{}.Smt.SptGraph.am".format(Network, gradient))

#Not used for now, but it will come in handy
path_thesis = os.path.join(path_src, '../../path_thesis/' + name_script)

#The folder where to save the results
path_output = os.path.join(path_src, '../../output_figures/' + name_script)
path_output_data=os.path.join(path_output,"F{}_n{}/".format(cells_3D, n)+gradient)

#Directory to save the assembled matrices and solution
path_matrices=os.path.join(path_output,"F{}_n{}".format(cells_3D, n))
#Directory to save the divided fiiles of the network
path_am=os.path.join(path_matrices, "divided_files_{}".format(gradient))
path_I_matrix=os.path.join(path_matrices, gradient)
#path_phi_bar = os.path.join(path_matrices, 'path_phi_bar')

os.makedirs(path_output, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
#os.makedirs(path_phi_bar, exist_ok=True)
os.makedirs(path_I_matrix, exist_ok=True)
os.makedirs(os.path.join(path_matrices, "E_portion"), exist_ok=True)
os.makedirs(path_am, exist_ok=True)

import shutil
destination_directory = os.path.join(path_am, 'network.am')
# Copy the directory recursively
shutil.copy(filename, destination_directory)


#True if no need to compute
phi_bar_bool=os.path.exists(os.path.join(path_matrices, 'phi_bar_q.npz')) and os.path.exists(os.path.join(path_matrices, 'phi_bar_s.npz')) 
#phi_bar_bool=False
B_assembly_bool=os.path.exists(os.path.join(path_matrices, 'B_matrix.npz'))
#B_assembly_bool=False
I_assembly_bool=os.path.exists(os.path.join(path_matrices, 'I_matrix.npz'))
#I_assembly_bool=False
#True if need to compute
Computation_Si_V=os.path.exists(os.path.join(path_matrices, 'Si_V.npz'))
Computation_bool= not os.path.exists(os.path.join(path_matrices, 'sol.npy'))
rec_bool=False
simple_plotting=False
Constant_Cv=False
already_loaded=False
linear_consumption=True

# =============================================================================
# #When changing flow and consumption, change the following:
# phi_bar_bool=False
# I_assembly_bool=False
# Computation_bool=True
# =============================================================================
#%%%%%%%%%%%%%

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
from PrePostTemp import GetCoarsekernels,GetInitialGuess, SplitFile, SetArtificialBCs, ClassifyVertices, get_phi_bar, Get9Lines, VisualizationTool, GetCoarsePhi
from assembly_1D import AssembleVertexToEdge, PreProcessingNetwork, CheckLocalConservativenessFlowRate
from post_processing import GetProbArgs

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




#######################################################################################
if Computation_bool: 
    output_files = SplitFile(filename, path_am)
    print("Split files:")
    for file in output_files:
        print(file)

#%
df = pd.read_csv(path_am + '/output_0.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(path_am + '/output_0.txt', 'r') as file:
    # Read the first line
    output_zero = file.readline()
pos_vertex=df.values

df = pd.read_csv(path_am + '/output_1.txt', skiprows=1, sep="\s+", names=["init", "end"])
with open(path_am + '/output_1.txt', 'r') as file:
    # Read the first line
    output_one = file.readline()
edges=df.values

df = pd.read_csv(path_am + '/output_2.txt', skiprows=1, sep="\s+", names=["cells_per_segment"])
with open(path_am + '/output_2.txt', 'r') as file:
    # Read the first line
    output_two = file.readline()
cells_per_segment=np.ndarray.flatten(df.values)

df = pd.read_csv(path_am + '/output_3.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(path_am + '/output_3.txt', 'r') as file:
    # Read the first line
    output_three= file.readline()
points=np.ndarray.flatten(df.values)

df = pd.read_csv(path_am + '/output_4.txt', skiprows=1, sep="\s+", names=["length"])
with open(path_am + '/output_4.txt', 'r') as file:
    # Read the first line
    output_four= file.readline()
diameters=np.ndarray.flatten(df.values)

diameters=diameters[np.arange(len(edges))*2]

df = pd.read_csv(path_am + '/output_5.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(path_am + '/output_5.txt', 'r') as file:
    # Read the first line
    output_five= file.readline()
Pressure=np.ndarray.flatten(df.values)

df = pd.read_csv(path_am + '/output_6.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(path_am + '/output_6.txt', 'r') as file:
    # Read the first line
    output_six= file.readline()

#I increase the Pe because it is way too slow
Flow_rate=np.ndarray.flatten(df.values)*factor_flow


K=np.average(diameters)/np.ndarray.flatten(diameters)*factor_K
#The flow rate is given in nl/s
U = 4*Flow_rate/np.pi/diameters**2*1e9*factor_flow #To convert to speed in micrometer/second

startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()
vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)

########################################################################
#   THIS IS A CRUCIAL OPERATION
########################################################################
print("Modifying edges according to pressure gradient")
for i in range(len(edges)):
    gradient=Pressure[2*i+1] - Pressure[2*i]
    
    if gradient<0:
        #print("Modifying edge ", i)
        edges[i,0]=endVertex[i]
        edges[i,1]=startVertex[i]    
    
startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()

CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, Flow_rate)
#%%
# =============================================================================
# L=np.sum((pos_vertex[endVertex] - pos_vertex[startVertex])**2, axis=1)**0.5
# 
# from assembly_1D import flow, AssignFlowBC
# from PrePostTemp import LabelVertexEdge
# 
# label_vertex, label_edge=LabelVertexEdge(vertex_to_edge,startVertex)
# bc_uid, bc_value=AssignFlowBC(0.0000000000005, 'x', pos_vertex, label_vertex)
# 
# F=flow( bc_uid, bc_value, L, diameters, startVertex, endVertex)
# F.solver()
# Pressure=dir_solve(F.A, F.P)
# uu=F.get_U()
# Flow_rate=uu*np.pi/4*diameters**2
# 
# print("Modifying edges according to pressure gradient")
# for i in range(len(edges)):
#     if Pressure[endVertex[i]] - Pressure[startVertex[i]]:
#         #print("Modifying edge ", i)
#         edges[i,0]=endVertex[i]
#         edges[i,1]=startVertex[i]  
# 
# CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, Flow_rate)
# =============================================================================

#%% - Creation of the 3D and Network objects
L_3D=np.array([300,300,300])

#Set artificial BCs for the network 
BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, startVertex, endVertex)

#BC_type=np.array(["Dirichlet", "Neumann","Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])

net=mesh_1D( startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, np.average(diameters)/2,1)
net.U=np.ndarray.flatten(U)

mesh=cart_mesh_3D(L_3D,cells_3D)
net.PositionalArraysFast(mesh)

cumulative_flow=np.zeros(3)
for i in range(len(Flow_rate)):
    cumulative_flow+=Flow_rate[i]*net.tau[i]
    
print("cumulative flow= ", cumulative_flow)


#%%
prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, K, BCs_1D)
#prob.intra_exit_BC="Dir"
#TRUE if no need to compute the matrices
prob.phi_bar_bool=phi_bar_bool
prob.I_assembly_bool=False
prob.intra_exit_BC='zero_flux'
prob.AssemblyGHI(path_matrices)

phi_bar=np.zeros(len(net.pos_s))

k=K*net.h/np.pi/net.R**2

# LINEAR SYSTEM
A=prob.I_matrix - np.diag(np.repeat(k, net.cells))
b=prob.III_ind_array - np.repeat(k, net.cells)*phi_bar

A=prob.I_matrix 
b=prob.III_ind_array + prob.H_matrix.dot(np.ones(prob.S))


Cv=dir_solve(A,-b)
plt.plot(Cv)






















