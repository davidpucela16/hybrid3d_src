#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:04:46 2023

@author: pdavid
"""
# Constants
factor_flow=1
factor_K=1
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
path_network=os.path.join(path_src, "../../networks/Synthetic_960")
filename=os.path.join(path_network,"Lc=75_facearea=75_min_dist=2.5_use_data=0_Rea{}_synthetic_network.Smt.SptGraph.am".format(Network))

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
Uncoupled=True
already_loaded=False
linear_consumption=True
# =============================================================================
# #When changing flow and consumption, change the following:
# 
# phi_bar_bool=False
# I_assembly_bool=False
# Computation_bool=False
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
diameters_points=np.ndarray.flatten(df.values)

diameters=diameters_points[np.arange(len(edges))*2]
#%%
#I increase the Pe because it is way too slow
Flow_rate=np.ndarray.flatten(df.values)*factor_flow

K=np.average(diameters)/np.ndarray.flatten(diameters)*factor_K
#The flow rate is given in nl/s
U = 4*Flow_rate/np.pi/diameters_points**2*1e9*factor_flow #To convert to speed in micrometer/second

startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()
vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)

#%% - Flow

L=np.sum((pos_vertex[endVertex] - pos_vertex[startVertex])**2, axis=1)**0.5

from assembly_1D import flow, AssignFlowBC
from PrePostTemp import LabelVertexEdge

label_vertex, label_edge=LabelVertexEdge(vertex_to_edge,startVertex)
bc_uid, bc_value=AssignFlowBC(0.5, 'x', pos_vertex, label_vertex)

F=flow( bc_uid, bc_value, L, diameters, startVertex, endVertex)
F.solver()
Pressure=dir_solve(F.A, F.P)
uu=F.get_U()
Flow_rate=uu*np.pi*diameters**2/4
########################################################################
#   THIS IS A CRUCIAL OPERATION
########################################################################
print("Modifying edges according to pressure gradient")
for i in range(len(edges)):
    if Pressure[endVertex[i]] - Pressure[startVertex[i]]:
        #print("Modifying edge ", i)
        edges[i,0]=endVertex[i]
        edges[i,1]=startVertex[i]    
    
startVertex=edges[:,0]
endVertex=edges[:,1]

CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, uu)

#%% - Creation of the 3D and Network objects
L_3D=np.max(pos_vertex, axis=0) - np.min(pos_vertex, axis=0)

#Set artificial BCs for the network 
BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, startVertex, endVertex)

#BC_type=np.array(["Dirichlet", "Neumann","Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])
from mesh_1D import mesh_1D
net=mesh_1D( startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, np.average(diameters)/2,np.average(uu))
net.U=np.ndarray.flatten(uu)

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
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool

import time
from scipy.sparse import csc_matrix



if not Computation_Si_V:
    kernel_array, row_array,sources_array=GetCoarsekernels(GetProbArgs(prob), path_matrices)
    Si_V=csc_matrix((kernel_array, (row_array, sources_array)), shape=(prob.F, prob.S))
    sp.sparse.save_npz(os.path.join(path_matrices, 'Si_V'), Si_V)
else:
    Si_V=sp.sparse.load_npz(os.path.join(path_matrices, 'Si_V.npz'))


#%%
if Computation_bool:
    if not already_loaded:
        prob.AssemblyProblem(path_matrices)
        A=prob.Full_linear_matrix
        #M_D=0.001
        
        Real_diff=1.2e5 #\mu m^2 / min
        CMRO2=Real_diff * M_D
        b=prob.Full_ind_array.copy()
        print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))
        plt.spy(prob.Full_linear_matrix)
        if not linear_consumption:
            
            b[:prob.F]-=M_D*mesh.h**3
        else:
            orig_A=prob.A_matrix.copy()
            orig_B=prob.B_matrix.copy()
            
            prob.A_matrix-=sp.sparse.diags(np.ones(prob.F), 0)*M_D*mesh.h**3
            prob.B_matrix-=Si_V*M_D*mesh.h**3
            
            A=prob.ReAssemblyMatrices()
            
            prob.A_matrix=orig_A.copy()
            prob.B_matrix=orig_B.copy()
            
        print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))
        plt.spy(prob.Full_linear_matrix)
        already_loaded=True
    #
    if Uncoupled:
        #Linear Cv decay
        pos_gradient=np.where(np.array(['x', 'y', 'z'])==gradient)[0][0]
        Cv=(L[pos_gradient]-net.pos_s[:,pos_gradient])/L[pos_gradient]
        
        Lin_matrix=prob.Full_linear_matrix.tolil()[:prob.F+prob.S,:prob.F+prob.S]
        ind_array=b[:prob.F+prob.S]
        ind_array[-prob.S:]+=prob.F_matrix.dot(Cv)
        pdb.set_trace()
        sol=dir_solve(Lin_matrix,-ind_array)
        sol=np.concatenate((sol, np.ones(prob.S)))
    else:
        #sol=dir_solve(prob.Full_linear_matrix,-prob.Full_ind_array)
        start=time.time()
        aa=GetInitialGuess(np.ones(len(edges)), prob)
        end=time.time()
        print("time= ", end-start)
        sol=np.concatenate((aa[0], aa[1], aa[2]))
        pdb.set_trace()
        sol = dir_solve(A, -b)
        np.save(os.path.join(path_output_data, 'sol'),sol)


#%%
#sol=np.load(os.path.join(path_output_data, 'sol.npy'))
prob.q=sol[prob.F:prob.F+prob.S]
prob.s=sol[:prob.F]
prob.Cv=sol[-prob.S:]
# =============================================================================
# prob.q=a[0]
# prob.s=a[1]
# prob.Cv=a[2]
# =============================================================================

#Test Conservativeness!!!!
CMRO2_tot=M_D*mesh.h**3*cells_3D**3
exchanges=np.dot(prob.q, np.repeat(net.h, net.cells))

print("Unconserved mass error: ", np.abs(exchanges-CMRO2_tot)/CMRO2_tot)

prob_args=GetProbArgs(prob)

phi_coarse=prob.s+Si_V.dot(prob.q)
vmin=np.min(phi_coarse)
vmax=np.max(phi_coarse)

fig, axs = plt.subplots(2, 2, figsize=(30,16))
im1 = axs[0, 0].imshow(phi_coarse[mesh.GetXSlice(L_3D[0]/5)].reshape(cells_3D, cells_3D), cmap='jet', vmin=vmin, vmax=vmax)
im2 = axs[0, 1].imshow(phi_coarse[mesh.GetXSlice(L_3D[0]/5*2)].reshape(cells_3D, cells_3D), cmap='jet', vmin=vmin, vmax=vmax)
im3 = axs[1, 0].imshow(phi_coarse[mesh.GetXSlice(L_3D[0]/5*3)].reshape(cells_3D, cells_3D), cmap='jet', vmin=vmin, vmax=vmax)
im4 = axs[1, 1].imshow(phi_coarse[mesh.GetXSlice(L_3D[0]/5*4)].reshape(cells_3D, cells_3D), cmap='jet', vmin=vmin, vmax=vmax)
# Adjust spacing between subplots
fig.tight_layout()

# Move the colorbar to the right of the subplots
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', shrink=0.8)
cbar_ax = cbar.ax
cbar_ax.set_position([0.83, 0.15, 0.03, 0.7])  # Adjust the position as needed

# Show the plot
plt.show()

#%%
res=40

corners=np.array([[0,0],[0,L_3D[0]],[L_3D[0],0],[L_3D[0],L_3D[0]]])*shrink_factor + L_3D[0]*(1-shrink_factor)/2
if simple_plotting:    
    
    aax=VisualizationTool(prob, 0,1,2, corners, res)
    aax.GetPlaneData(path_output_data)
    aax.PlotData(path_output_data)
    aay=VisualizationTool(prob, 1,0,2, corners, res)
    aay.GetPlaneData(path_output_data)
    aay.PlotData(path_output_data)
    aaz=VisualizationTool(prob, 2,0,1, corners, res)
    aaz.GetPlaneData(path_output_data)
    aaz.PlotData(path_output_data)
# =============================================================================
#     aaz=VisualizationTool(prob, 2,1,0, corners, res)
#     aaz.GetPlaneData(path_output_data)
#     aaz.PlotData(path_output_data)
#     aax2=VisualizationTool(prob, 0,2,1, corners, res)
#     aax2.GetPlaneData(path_output_data)
#     aax2.PlotData(path_output_data)
# =============================================================================

if rec_bool:
    num_processes=10
    process=0 #This must be kept to zero for the parallel reconstruction to go right
    perp_axis_res=res
    path_vol_data=os.path.join(path_output_data, "vol_data")
    aaz=VisualizationTool(prob, 2,0,1, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
    aaz.GetVolumeData(num_processes, process, perp_axis_res, path_vol_data, shrink_factor)

phi_coarse=GetCoarsePhi(prob, prob.q, prob.Cv, prob.s)

#%% - Data for Avizo
from PrePostTemp import GetEdgesConcentration,GetSingleEdgeSources, GetEdgesConcentration, LabelVertexEdge

edges_concentration=GetEdgesConcentration(net.cells, prob.Cv)
vertex_label, edge_label=LabelVertexEdge(vertex_to_edge, startVertex)
title="@8 # Edge Concentration"
np.savetxt(os.path.join(path_matrices, "Edge_concentration.txt"), edges_concentration, fmt='%f', delimiter=' ', header=title, comments='')
title="@9 # Entry Exit"
np.savetxt(os.path.join(path_matrices, "Entry_Exit.txt"), vertex_label, fmt='%d', delimiter=' ', header=title, comments='')

title="@10 # Entry Exit Edge"
np.savetxt(os.path.join(path_matrices, "Entry_Exit_Edge.txt"), vertex_label, fmt='%d', delimiter=' ', header=title, comments='')

#%%
#def GetVertexConc(vertex_to_edge, startVertex, )

for i in np.where(edge_label==1)[0][:]: #1 if entering, 2 if exiting
    #plt.plot(GetSingleEdgeConc(net.cells, prob.Cv, i))
    plt.plot(prob.Cv[GetSingleEdgeSources(net.cells, i)])
    plt.plot(np.zeros(net.cells[i])+ edges_concentration[i])
    #plt.ylim((0.98,1))
#plt.show()


#%% - To calculate different PDFs

# Assuming you have the arrays phi and x defined
bins=50

h = L_3D[0]/bins  # Set the distance threshold 'h'

# Calculate average concentration for each position within distance 'h'
unique_x = np.linspace(h/2, L_3D[0]-h/2, bins)
average_phi = []
for pos in unique_x:
    mask = np.abs(net.pos_s[:,0] - pos) <= h
    average_phi.append(np.mean(prob.Cv[mask]))

# Plotting the average concentration
plt.plot(unique_x, average_phi)
plt.xlabel('Position')
plt.ylabel('Average Concentration')
plt.title('Average Concentration vs Position (within distance h)')
#plt.show()
#%%

# Assuming you have the arrays phi and x defined
bins=50

h = L_3D[0]/bins  # Set the distance threshold 'h'

# Calculate average concentration for each position within distance 'h'
unique_x = np.linspace(h/2, L_3D[0]-h/2, bins)
average_phi = []
for pos in unique_x:
    mask = np.abs(net.pos_s[:,0] - pos) <= h
    average_phi.append(np.sum(mask))

# Plotting the average concentration
plt.plot(unique_x, average_phi)
plt.xlabel('Position')
plt.ylabel('Average Concentration')
plt.title('Average Concentration vs Position (within distance h)')
#plt.show()
