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
n=1
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
path_network=os.path.join(path_src, "../../networks/Network_Flo/All_files/")
filename=os.path.join(path_network,"Network1_Figure_Data.txt".format(Network, gradient))

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

#When changing flow and consumption, change the following:
# =============================================================================
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


#%%

if Computation_bool: 
    output_files = SplitFile(filename, path_am)
    print("Split files:")
    for file in output_files:
        print(file)
        
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

point_diameters=diameters[np.arange(len(edges))*2]

df = pd.read_csv(path_am + '/output_5.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(path_am + '/output_5.txt', 'r') as file:
    # Read the first line
    output_five= file.readline()
Hematocrit=np.ndarray.flatten(df.values)

df = pd.read_csv(path_am + '/output_6.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(path_am + '/output_6.txt', 'r') as file:
    # Read the first line
    output_six= file.readline()
Fictional_edge=np.ndarray.flatten(df.values)


df = pd.read_csv(path_am + '/output_7.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(path_am + '/output_7.txt', 'r') as file:
    # Read the first line
    output_seve= file.readline()
ArtVenCap_Label=np.ndarray.flatten(df.values)

df = pd.read_csv(path_am + '/output_8.txt', skiprows=1, sep="\s+", names=["length"])
with open(path_am + '/output_8.txt', 'r') as file:
    # Read the first line
    output_eight= file.readline()
diameters=np.ndarray.flatten(df.values)


df = pd.read_csv(path_am + '/output_10.txt', skiprows=1, sep="\s+", names=["length"])
with open(path_am + '/output_10.txt', 'r') as file:
    # Read the first line
    output_te= file.readline()
Flow_rate=np.ndarray.flatten(df.values)

df = pd.read_csv(path_am + '/output_12.txt', skiprows=1, sep="\s+", names=["length"])
with open(path_am + '/output_12.txt', 'r') as file:
    # Read the first line
    output_twelve= file.readline()
Pressure=np.ndarray.flatten(df.values)

#Is this a good idea?
pos_vertex-=np.min(pos_vertex, axis=0)


K=np.average(diameters)/np.ndarray.flatten(diameters)
#The flow rate is given in nl/s
U = 4*Flow_rate/np.pi/diameters**2*1e9 #To convert to speed in micrometer/second

startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()
vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)


########################################################################
#   THIS IS A CRUCIAL OPERATION
########################################################################
c=0
for i in edges:
    grad=Pressure[i[1]] - Pressure[i[0]]
    #pdb.set_trace()
    if grad<0 and Flow_rate[c]<0:
    #if gradient>0:
        print("errror")
        edges[c,0]=endVertex[c]
        edges[c,1]=startVertex[c]    
    c+=1
    
startVertex=edges[:,0]
endVertex=edges[:,1]

CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, Flow_rate)
L_3D=np.max(pos_vertex, axis=0)*1.01-np.min(pos_vertex, axis=0)*0.99

#Set artificial BCs for the network 
BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, startVertex, endVertex)

#BC_type=np.array(["Dirichlet", "Neumann","Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])

h_approx=np.average(diameters)*3
net=mesh_1D( startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h_approx ,np.average(U))
net.U=np.ndarray.flatten(U)

mesh=cart_mesh_3D(L_3D,cells_3D)
net.PositionalArraysFast(mesh)

cumulative_flow=np.zeros(3)
for i in range(len(Flow_rate)):
    cumulative_flow+=Flow_rate[i]*net.tau[i]
    
print("cumulative flow= ", cumulative_flow)

prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, K, BCs_1D)
#TRUE if no need to compute the matrices

prob.phi_bar_bool=phi_bar_bool
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool
sol_linear_system=Computation_bool

if not Computation_Si_V:
    kernel_array, row_array,sources_array=GetCoarsekernels(GetProbArgs(prob), path_matrices)
    Si_V=csc_matrix((kernel_array, (row_array, sources_array)), shape=(prob.F, prob.S))
    sp.sparse.save_npz(os.path.join(path_matrices, 'Si_V'), Si_V)
else:
    Si_V=sp.sparse.load_npz(os.path.join(path_matrices, 'Si_V.npz'))

if sol_linear_system:
    D_E_F=prob.AssemblyDEFFast(path_matrices + "/E_portion", path_matrices)
    A_B_C=prob.AssemblyABC(path_matrices)
    pos_gradient=np.where(np.array(["x","y","z"])==gradient)[0][0]
    Cv=(L_3D[pos_gradient]-net.pos_s[:,pos_gradient])/L_3D[pos_gradient]
    Lin_matrix=sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix-Si_V*M_D*mesh.h**3)),
                                 sp.sparse.hstack((prob.D_matrix, prob.q_portion + prob.Gij))))
    
    b=np.concatenate((prob.I_ind_array, prob.F_matrix.dot(Cv)))
    print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))
    print("Deleting matrices")
    del(D_E_F)
    del(A_B_C)
    print("Converting matrix")
    Lin_matrix=Lin_matrix.astype('float32')
    b=b.astype('float32')
    #sol=dir_solve(prob.Full_linear_matrix,-prob.Full_ind_array)
    print("solve problem")
 
    sol=dir_solve(Lin_matrix, -b) 
    np.save(os.path.join(path_output_data, 'sol'),sol)

sol=np.load(os.path.join(path_output_data, 'sol.npy'))
prob.q=sol[-prob.S:]
prob.s=sol[:-prob.S]
prob.Cv=Cv

pdb.set_trace()

prob.AssemblyGHI(path_matrices)
inv=sp.sparse.linalg.inv(prob.I_matrix)
np.save(os.path.join(path_output_data, 'I_inv'),inv)
