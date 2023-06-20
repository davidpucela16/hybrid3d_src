#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:04:46 2023

@author: pdavid
"""

import os
import sys

script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)
print(path_script)

name_script = script[script.rfind('/')+1:-3]

path_src = os.path.join(path_script, '../../src')
path_potentials = os.path.join(path_script, '../../Potentials')
# Now the creation of the relevant folders to store the output
path_matrices = os.path.join(path_src, '../../linear_system/' + name_script)
path_output = os.path.join(path_src, '../../output_figures/' + name_script)

path_thesis = os.path.join(path_src, '../../path_thesis/' + name_script)

os.makedirs(path_matrices, exist_ok=True)
os.makedirs(path_output, exist_ok=True)

sys.path.append(path_src)
sys.path.append(path_potentials)

# Constants
factor_flow=1
cells_3D=20
n=3
shrink_factor=(cells_3D-1)/cells_3D
Network=1
gradient="x"
M_D=0.0002

# Paths
script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)
print(path_script)
#path_network="/home/pdavid/Bureau/PhD/Network_Flo/Synthetic_ROIs_300x300x300" #The path with the network
path_network=os.path.join(path_src, "../../networks/Synthetic_ROIs_300x300x300")

#Directory to save the assembled matrices and solution
path_matrices=os.path.join(path_matrices,"F{}_n{}".format(cells_3D, n))
path_phi_bar = os.path.join(path_matrices, 'path_phi_bar')

#Directory to save the divided fiiles of the network
output_dir_network=os.path.join(path_matrices, "divided_files_{}".format(gradient))

filename=os.path.join(path_network,"Rea{}_synthetic_{}.Smt.SptGraph.am".format(Network, gradient))

#Directory to save the reconstruction

#output_dir_network = '/home/pdavid/Bureau/PhD/Network_Flo/All_files/Split/'  # Specify the output directory here
os.makedirs(output_dir_network, exist_ok=True)  # Create the output directory if it doesn't exist
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
path_output_data=os.path.join(path_matrices, gradient)
os.makedirs(path_output_data, exist_ok=True)
os.makedirs(path_phi_bar, exist_ok=True)
os.makedirs(os.path.join(path_matrices, "E_portion"), exist_ok=True)

#True if no need to compute
phi_bar_bool=os.path.exists(os.path.join(path_matrices, 'phi_bar_q.npz')) and os.path.exists(os.path.join(path_matrices, 'phi_bar_s.npz')) 
#phi_bar_bool=False
B_assembly_bool=os.path.exists(os.path.join(path_matrices, 'B_matrix.npz'))
#B_assembly_bool=False
I_assembly_bool=os.path.exists(os.path.join(path_matrices, 'I_matrix.npz'))
#I_assembly_bool=False
#True if need to compute
#Computation_bool= not os.path.exists(os.path.join(path_output_data, 'sol.npy'))
Computation_Si_V=os.path.exists(os.path.join(path_matrices, 'Si_V.npz'))
Computation_bool=True
test_iterative=True
rec_bool=False
simple_plotting=True
Constant_Cv=False
already_loaded=False





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
    output_files = SplitFile(filename, output_dir_network)
    print("Split files:")
    for file in output_files:
        print(file)

#%
df = pd.read_csv(output_dir_network + '/output_0.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(output_dir_network + '/output_0.txt', 'r') as file:
    # Read the first line
    output_zero = file.readline()
pos_vertex=df.values

df = pd.read_csv(output_dir_network + '/output_1.txt', skiprows=1, sep="\s+", names=["init", "end"])
with open(output_dir_network + '/output_1.txt', 'r') as file:
    # Read the first line
    output_one = file.readline()
edges=df.values

df = pd.read_csv(output_dir_network + '/output_2.txt', skiprows=1, sep="\s+", names=["cells_per_segment"])
with open(output_dir_network + '/output_2.txt', 'r') as file:
    # Read the first line
    output_two = file.readline()
cells_per_segment=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_3.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(output_dir_network + '/output_3.txt', 'r') as file:
    # Read the first line
    output_three= file.readline()
points=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_4.txt', skiprows=1, sep="\s+", names=["length"])
with open(output_dir_network + '/output_4.txt', 'r') as file:
    # Read the first line
    output_four= file.readline()
diameters=np.ndarray.flatten(df.values)

diameters=diameters[np.arange(len(edges))*2]

df = pd.read_csv(output_dir_network + '/output_5.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_5.txt', 'r') as file:
    # Read the first line
    output_five= file.readline()
Pressure=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_6.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_6.txt', 'r') as file:
    # Read the first line
    output_six= file.readline()

#I increase the Pe because it is way too slow
Flow_rate=np.ndarray.flatten(df.values)*factor_flow


K=np.average(diameters)/np.ndarray.flatten(diameters)
#The flow rate is given in nl/s
U = 4*Flow_rate/np.pi/diameters**2*1e9 #To convert to speed in micrometer/second

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
    
startVertex=edges[:,0]
endVertex=edges[:,1]

CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, Flow_rate)

#%% - Creation of the 3D and Network objects
L_3D=np.array([300,300,300])

#Set artificial BCs for the network 
BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, startVertex, endVertex)

#BC_type=np.array(["Dirichlet", "Neumann","Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])

net=mesh_1D( startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, np.average(diameters)/2,np.average(U))
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





#%%
if Computation_bool:
    if not already_loaded:
        prob.AssemblyProblem(path_matrices)
        A=prob.Full_linear_matrix
        #M_D=0.001
        
        Real_diff=1.2e5 #\mu m^2 / min
        CMRO2=Real_diff * M_D
        b=prob.Full_ind_array.copy()
        b[:prob.F]-=M_D*mesh.h**3
        print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))
        plt.spy(prob.Full_linear_matrix)
        already_loaded=True
    #
    if Constant_Cv:
        #Constant Cv
        Lin_matrix=prob.Full_linear_matrix.tolil()[:prob.F+prob.S,:prob.F+prob.S]
        ind_array=b[:prob.F+prob.S]
        ind_array[-prob.S:]+=prob.F_matrix.dot(np.ones(prob.S))
        sol=dir_solve(Lin_matrix,-ind_array)
        pdb.set_trace()
        sol=np.concatenate((sol, np.ones(prob.S)))
    else:
        #sol=dir_solve(prob.Full_linear_matrix,-prob.Full_ind_array)
        start=time.time()
        aa=GetInitialGuess(np.ones(len(edges)), prob)
        end=time.time()
        print("time= ", end-start)
        sol=np.concatenate((aa[0], aa[1], aa[2]))
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
