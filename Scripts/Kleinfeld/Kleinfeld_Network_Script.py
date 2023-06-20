#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:04:46 2023

@author: pdavid
"""

import pandas as pd
import numpy as np 
import pdb 
from numba import njit
import scipy as sp
from scipy.sparse.linalg import spsolve as dir_solve
import matplotlib.pyplot as plt
import os 
import sys


#path_current_file=os.path.dirname(__file__)
script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)
print(path_script)
#path_network="/home/pdavid/Bureau/PhD/Network_Flo/All_files"
path_network = os.path.join(path_script, "..") #The path with the network
#os.chdir('/home/pdavid/Bureau/Code/hybrid3d/src_final')
#sys.path.append('/home/pdavid/Bureau/Code/hybrid3d/src_final')
sys.path.append(os.path.join(path_script, "src_final"))
#%%
path_output=os.path.join(path_network, "Kleinfeld")
###cells_3D=20
###n=1
cells_3D, n = int(sys.argv[1]), int(sys.argv[2])
print(cells_3D, n)
path_matrices=os.path.join(path_output,"F{}_n{}".format(cells_3D, n))
#output_dir_network="/home/pdavid/Bureau/Code/hybrid3d/Synthetic_Rea{}/{}/divided_files".format(Network, gradient)
output_dir_network=os.path.join(path_network, "Kleinfeld_divided")
filename=os.path.join(path_network,"Network1_Figure_Data.txt")
path_output_data=path_matrices + '/out_data'

#output_dir_network = '/home/pdavid/Bureau/PhD/Network_Flo/All_files/Split/'  # Specify the output directory here
os.makedirs(output_dir_network, exist_ok=True)  # Create the output directory if it doesn't exist
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
os.makedirs(path_output_data, exist_ok=True)

#True if no need to compute 
phi_bar_bool=os.path.exists(os.path.join(path_matrices, 'phi_bar_q.npz')) and os.path.exists(os.path.join(path_matrices, 'phi_bar_s.npz')) 
B_assembly_bool=os.path.exists(os.path.join(path_matrices, 'B_matrix.npz'))
I_assembly_bool=os.path.exists(os.path.join(path_matrices, 'I_matrix.npz'))
#True if need to compute
string_value=False
string_value = sys.argv[3]
Computation_bool = False if string_value == 'False' else bool(string_value)
rec_bool=True
print("Computation: ", Computation_bool)
from mesh_1D import mesh_1D
from hybridFast import hybrid_set_up
from mesh import cart_mesh_3D
from post_processing import GetPlaneReconstructionFast
from assembly_1D import AssembleVertexToEdge, PreProcessingNetwork, CheckLocalConservativenessFlowRate
from PrePostTemp import SplitFile, SetArtificialBCs, ClassifyVertices, get_phi_bar, Get9Lines, VisualizationTool
from PrePostTemp import GetInitialGuess


if Computation_bool: 
    output_files = SplitFile(filename, output_dir_network)
    print("Split files:")
    for file in output_files:
        print(file)

#%%
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

point_diameters=diameters[np.arange(len(edges))*2]

df = pd.read_csv(output_dir_network + '/output_5.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_5.txt', 'r') as file:
    # Read the first line
    output_five= file.readline()
Hematocrit=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_6.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_6.txt', 'r') as file:
    # Read the first line
    output_six= file.readline()
Fictional_edge=np.ndarray.flatten(df.values)


df = pd.read_csv(output_dir_network + '/output_7.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_7.txt', 'r') as file:
    # Read the first line
    output_seve= file.readline()
ArtVenCap_Label=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_8.txt', skiprows=1, sep="\s+", names=["length"])
with open(output_dir_network + '/output_8.txt', 'r') as file:
    # Read the first line
    output_eight= file.readline()
diameters=np.ndarray.flatten(df.values)


df = pd.read_csv(output_dir_network + '/output_10.txt', skiprows=1, sep="\s+", names=["length"])
with open(output_dir_network + '/output_10.txt', 'r') as file:
    # Read the first line
    output_te= file.readline()
Flow_rate=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_12.txt', skiprows=1, sep="\s+", names=["length"])
with open(output_dir_network + '/output_12.txt', 'r') as file:
    # Read the first line
    output_twelve= file.readline()
Pressure=np.ndarray.flatten(df.values)


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
    gradient=Pressure[i[1]] - Pressure[i[0]]
    #pdb.set_trace()
    if gradient<0 and Flow_rate[c]<0:
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

if sol_linear_system:
    D_E_F=prob.AssemblyDEFFast(path_matrices + "/E_portion", path_matrices)
    A_B_C=prob.AssemblyABC(path_matrices)
    G_H_I=prob.AssemblyGHI(path_matrices)
    prob.Full_linear_matrix=sp.sparse.vstack((A_B_C,D_E_F,G_H_I))
    III_ind_array=np.load(os.path.join(path_matrices, 'III_ind_array.npy'))
    Full_ind_array=np.concatenate((prob.I_ind_array, np.zeros(len(prob.mesh_1D.pos_s)), III_ind_array))
    #M_D=0.001
    M_D=0.0002
    Real_diff=1.2e5 #\mu m^2 / min
    CMRO2=Real_diff * M_D
    prob.Full_ind_array=Full_ind_array
    prob.Full_ind_array[:cells_3D**2]-=M_D*mesh.h**3
    print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))
    print("Deleting matrices")
    del(D_E_F)
    del(A_B_C)
    del(G_H_I)
    print("Converting matrix")
    prob.Full_linear_matrix=prob.Full_linear_matrix.astype('float32')
    prob.Full_ind_array=prob.Full_ind_array.astype('float32')
    #sol=dir_solve(prob.Full_linear_matrix,-prob.Full_ind_array)
    print("solve problem")
    #a=GetInitialGuess(ArtVenCap_Label, prob)
    #initial_guess=np.concatenate((a[0], a[1], a[2]))
    #sol=sp.sparse.linalg.bicgstab(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_bicgstab'),sol[0])
    #sol=sp.sparse.linalg.cg(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_cg'),sol[0])
    #sol=sp.sparse.linalg.cgs(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_cgs'),sol[0])
    #sol=sp.sparse.linalg.gmres(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_gmres'),sol[1])
    #sol=sp.sparse.linalg.lgmres(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_lgmres'),sol[0])
    #sol=sp.sparse.linalg.minres(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_minres'),sol[0])
    #sol=sp.sparse.linalg.qmr(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_qmr'),sol[0])
    #sol=sp.sparse.linalg.gcrotmk(prob.Full_linear_matrix,-prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_gcrotmk'),sol[1])
    sol=dir_solve(prob.Full_linear_matrix, -prob.Full_ind_array); np.save(os.path.join(path_matrices, 'sol_dir'),sol)
    print("Exit code", sol[1])

sol=np.load(os.path.join(path_matrices, 'sol.npy'))
prob.q=sol[-2*prob.S:-prob.S]
prob.s=sol[:-2*prob.S]
prob.Cv=sol[-prob.S:]


#%%
if rec_bool:
    res=500
    path_vol_data=os.path.join(path_output_data, "vol_data")
    aaz=VisualizationTool(prob, 2,0,1, np.array([[mesh.h/2,mesh.h/2],
                                                 [mesh.h/2,mesh.L[1]-mesh.h/2],
                                                 [mesh.L[0]-mesh.h/2,mesh.h/2],
                                                 [mesh.L[0]-mesh.h/2,mesh.L[1]-mesh.h/2]]), res)
    aaz.GetPlaneData(path_vol_data)




