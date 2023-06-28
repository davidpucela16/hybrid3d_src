#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:04:46 2023

@author: pdavid
"""

import os 
import sys
import pandas as pd
import numpy as np 
import pdb 
from numba import njit
import scipy as sp
from scipy.sparse.linalg import spsolve as dir_solve
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15,15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)


script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)
print(path_script)
#path_network="/home/pdavid/Bureau/PhD/Network_Flo/Synthetic_ROIs_300x300x300" #The path with the network
path_network=os.path.join(path_script, "../Synthetic_ROIs_300x300x300")


Network=1
gradient="x"

#path_output="/home/pdavid/Bureau/Code/hybrid3d/Synthetic_Rea{}".format(Network)
path_output=os.path.join(path_script, "../Synthetic_Rea{}".format(Network))
cells_3D=20
n=1

#Directory to save the assembled matrices and solution
path_matrices=os.path.join(path_output,"F{}_n{}".format(cells_3D, n))
#Directory to save the divided fiiles of the network
output_dir_network=os.path.join(path_matrices, "divided_files_{}".format(gradient))

#Path where the .am is located and the name of the file 
#path_network="/home/pdavid/Bureau/PhD/Network_Flo/Synthetic_ROIs_300x300x300" #The path with the network
path_network=os.path.join(path_script, "../Synthetic_ROIs_300x300x300")
filename=os.path.join(path_network,"Rea{}_synthetic_{}.Smt.SptGraph.am".format(Network, gradient))

#Directory to save the reconstruction

#output_dir_network = '/home/pdavid/Bureau/PhD/Network_Flo/All_files/Split/'  # Specify the output directory here
os.makedirs(output_dir_network, exist_ok=True)  # Create the output directory if it doesn't exist
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
path_output_data=os.path.join(path_matrices, gradient)
os.makedirs(path_output_data, exist_ok=True)
os.makedirs(os.path.join(path_matrices, "E_portion"), exist_ok=True)

#True if no need to compute
#True if no need to compute 
phi_bar_bool=os.path.exists(os.path.join(path_matrices, 'phi_bar_q.npz')) and os.path.exists(os.path.join(path_matrices, 'phi_bar_s.npz')) 
B_assembly_bool=os.path.exists(os.path.join(path_matrices, 'B_matrix.npz'))
I_assembly_bool=os.path.exists(os.path.join(path_matrices, 'I_matrix.npz'))
#True if need to compute

Computation_bool = not os.path.exists(os.path.join(path_matrices, 'sol.npy'))
rec_bool=True

sys.path.append(os.path.join(path_script, "src_final"))
from mesh_1D import mesh_1D
from hybridFast import hybrid_set_up
from mesh import cart_mesh_3D
from post_processing import GetPlaneReconstructionFast
from assembly_1D import AssembleVertexToEdge, PreProcessingNetwork, CheckLocalConservativenessFlowRate, CheckLocalConservativenessVelocity
from PrePostTemp import SplitFile, SetArtificialBCs, ClassifyVertices, get_phi_bar, Get9Lines, VisualizationTool

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
Flow_rate=np.ndarray.flatten(df.values)*1e3


K=np.average(diameters)/np.ndarray.flatten(diameters)
#The flow rate is given in nl/s
U = 4*Flow_rate/np.pi/diameters**2*1e9 #To convert to speed in micrometer/second

startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()
vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)

#%% - Pre processing flow_rate

########################################################################
#   THIS IS A CRUCIAL OPERATION
########################################################################

for i in range(len(edges)):
    gradient=Pressure[2*i+1] - Pressure[2*i]
    if gradient<0:
        edges[i,0]=endVertex[i]
        edges[i,1]=startVertex[i]    
    
startVertex=edges[:,0]
endVertex=edges[:,1]

CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, Flow_rate)

#%% - Creation of the 3D and Network objects
L_3D=np.array([305,305,305])

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
#TRUE if no need to compute the matrices
prob.phi_bar_bool=phi_bar_bool
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool
sol_linear_system=Computation_bool

#%%

import time
if sol_linear_system:
    prob.AssemblyProblem(path_matrices)
    #M_D=0.001
    M_D=0.0002
    Real_diff=1.2e5 #\mu m^2 / min
    CMRO2=Real_diff * M_D
    prob.Full_ind_array[:cells_3D**2]-=M_D*mesh.h**3
    print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))
    begin=time.time()
    plt.spy(prob.Full_linear_matrix)
    pdb.set_trace()
    #sol=dir_solve(prob.Full_linear_matrix,-prob.Full_ind_array)
    sol=sp.sparse.linalg.bicg(prob.Full_linear_matrix, -prob.Full_ind_array)
    end=time.time()
    np.save(os.path.join(path_matrices, 'sol'),sol)

sol=np.load(os.path.join(path_matrices, 'sol.npy'))
prob.q=sol[-2*prob.S:-prob.S]
prob.s=sol[:-2*prob.S]
prob.Cv=sol[-prob.S:]
# =============================================================================
# for i in range(3):
#     phi,crds, others, points_a, points_b=Get9Lines(i, 200, L_3D, prob)
#     for k in range(9):
#         plt.plot(phi[k], label=str(np.array(["x","y","z"])[others]) + "={:.1f}, {:.1f}".format(points_a[k//3], points_b[k%3]))
#     plt.xlabel(np.array(["x","y","z"])[i])
#     plt.legend()
#     plt.show()
# =============================================================================

#%%
res=50
simple_plotting=False
if simple_plotting:    
    
    aax=VisualizationTool(prob, 0,1,2, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
    aax.GetPlaneData(path_output_data)
    aay=VisualizationTool(prob, 1,0,2, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
    aay.GetPlaneData(path_output_data)
    
    aaz=VisualizationTool(prob, 2,1,0, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
    aaz.GetPlaneData(path_output_data)
    
    aax2=VisualizationTool(prob, 0,2,1, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
    aax2.GetPlaneData(path_output_data)
    
    aax.PlotData(path_output_data)
    aay.PlotData(path_output_data)
    aaz.PlotData(path_output_data)
    
    aax2.PlotData(path_output_data)

if rec_bool:
    num_processes=25
    process=0 #This must be kept to zero for the parallel reconstruction to go right
    perp_axis_res=50
    path_vol_data=os.path.join(path_output_data, "vol_data")
    aaz=VisualizationTool(prob, 2,0,1, np.array([[16,16],[16,289],[289,16],[289,289]]), res)
    aaz.GetVolumeData(num_processes, process, perp_axis_res, path_vol_data)
