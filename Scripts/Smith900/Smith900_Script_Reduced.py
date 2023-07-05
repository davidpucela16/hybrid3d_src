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
n=1
shrink_factor=(cells_3D-1)/cells_3D
Network=1
ratio=1/2
gradient="x"
M=1e-4

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
path_output_data=os.path.join(path_output,"F{}_n{}_r{}/".format(cells_3D, n, int(ratio*10))+gradient)

#Directory to save the assembled matrices and solution
path_matrices=os.path.join(path_output,"F{}_n{}_r{}".format(cells_3D, n, int(ratio*10)))
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
diameters_points=np.ndarray.flatten(df.values)*10

diameters=diameters_points[np.arange(len(edges))*2]
#%%
K=np.average(diameters)/np.ndarray.flatten(diameters)*factor_K

startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()
vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)

#%% - Flow

L=np.sum((pos_vertex[endVertex] - pos_vertex[startVertex])**2, axis=1)**0.5

from assembly_1D import flow, AssignFlowBC
from PrePostTemp import LabelVertexEdge

label_vertex, label_edge=LabelVertexEdge(vertex_to_edge,startVertex)
bc_uid, bc_value=AssignFlowBC(5, 'x', pos_vertex, label_vertex)

F=flow( bc_uid, bc_value, L, diameters, startVertex, endVertex)
F.solver()
Pressure=dir_solve(F.A, F.P)
U=F.get_U()*factor_flow
Flow_rate=U*np.pi*diameters**2/4
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

CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, U)

D=2
U_D=U/D
M_D=M/D

#%% - Creation of the 3D and Network objects
L_3D=np.max(pos_vertex, axis=0) - np.min(pos_vertex, axis=0)

#Set artificial BCs for the network 
BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, startVertex, endVertex)

#BC_type=np.array(["Dirichlet", "Neumann","Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])
from mesh_1D import mesh_1D
h_approx=diameters/ratio
net=mesh_1D( startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h_approx,1)
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

#%%
from PrePostTemp import AssembleReducedProblem, InitialGuessSimple
import time
prob.phi_bar_bool=phi_bar_bool
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool
sol_linear_system=Computation_bool

def callback(residual):
    print("Residual norm:", np.average(residual))


if not Computation_Si_V:
    kernel_array, row_array,sources_array=GetCoarsekernels(GetProbArgs(prob), path_matrices)
    Si_V=csc_matrix((kernel_array, (row_array, sources_array)), shape=(prob.F, prob.S))
    sp.sparse.save_npz(os.path.join(path_matrices, 'Si_V'), Si_V)
else:
    Si_V=sp.sparse.load_npz(os.path.join(path_matrices, 'Si_V.npz'))


if sol_linear_system:
    D_E_F=prob.AssemblyDEFFast(path_matrices + "/E_portion", path_matrices)
    A_B_C=prob.AssemblyABC(path_matrices)
    prob.B_matrix-=Si_V*M_D*mesh.h**3
    G_H_I=prob.AssemblyGHI(path_matrices)
    prob.E_matrix=prob.Gij+prob.q_portion
    #The matrices are ready 
    A,b=AssembleReducedProblem(prob.A_matrix-sp.sparse.diags(np.ones(prob.F), 0)*M_D*mesh.h**3,
                               prob.B_matrix-Si_V*M_D*mesh.h**3,
                               prob.D_matrix,
                               prob.Gij+prob.q_portion,
                               prob.F_matrix,
                               prob.aux_arr,
                               prob.I_matrix, 
                               prob.I_ind_array, 
                               prob.III_ind_array)
    print("Solving matrix!")
    
    arr_time=[]
    arr_tol=[]
    
    print("Solving gcrotmk")
    a=time.time()
    sol_gcrotmk=sp.sparse.linalg.gcrotmk(A,-b,x0=InitialGuessSimple(Si_V, np.repeat(prob.K, net.cells), 0.1, np.ones(prob.S)), tol=1e-3, maxiter=300, callback=callback)
    bb=time.time()
    np.save(os.path.join(path_matrices, 'sol_gcrotmk.npy'), sol_gcrotmk[0])
    t=bb-a
    arr_time.append(t)
    arr_tol.append(sol_gcrotmk[1])
    np.save(os.path.join(path_matrices, 'time.npy'), np.array([arr_time]))
    np.savetxt(os.path.join(path_matrices, 'tol.txt'), np.array(arr_tol[1]), fmt='%f')
    
    print("Solving lgmres")
    a=time.time()
    sol_lgmres=sp.sparse.linalg.lgmres(A,-b,x0=InitialGuessSimple(Si_V, np.repeat(prob.K, net.cells), 0.1, np.ones(prob.S)), tol=1e-3, maxiter=300, callback=callback)
    bb=time.time()
    np.save(os.path.join(path_matrices, 'sol_lgmres.npy'), sol_lgmres[0])
    t=bb-a
    arr_time.append(t)
    arr_tol.append(sol_lgmres[1])
    np.save(os.path.join(path_matrices, 'time.npy'), np.array([arr_time]))
    np.savetxt(os.path.join(path_matrices, 'tol.txt'), np.array(arr_tol[1]), fmt='%f')
    
    print("Solving grad")
    a=time.time()
    sol_grad=sp.sparse.linalg.bicg(A,-b,x0=InitialGuessSimple(Si_V, np.repeat(prob.K, net.cells), 0.1, np.ones(prob.S)), tol=1e-3, maxiter=300, callback=callback)
    bb=time.time()
    np.save(os.path.join(path_matrices, 'sol_grad.npy'), sol_grad[0])
    t=bb-a
    arr_time.append(t)
    arr_tol.append(sol_grad[1])
    np.save(os.path.join(path_matrices, 'time.npy'), np.array([arr_time]))
    np.savetxt(os.path.join(path_matrices, 'tol.txt'), np.array(arr_tol[1]), fmt='%f')    
    
    print("Solving gradstab")
    a=time.time()
    sol_gradstab=sp.sparse.linalg.bicgstab(A,-b,x0=InitialGuessSimple(Si_V, np.repeat(prob.K, net.cells), 0.1, np.ones(prob.S)), tol=1e-3, maxiter=300, callback=callback)
    bb=time.time()
    np.save(os.path.join(path_matrices, 'sol_gradstab.npy'), sol_gradstab[0])
    t=bb-a
    arr_time.append(t)
    arr_tol.append(sol_gradstab[1])
    np.save(os.path.join(path_matrices, 'time.npy'), np.array([arr_time]))
    np.savetxt(os.path.join(path_matrices, 'tol.txt'), np.array(arr_tol[1]), fmt='%f')
    
    print("Solving gmres")
    a=time.time()
    sol_gmres=sp.sparse.linalg.gmres(A,-b,x0=InitialGuessSimple(Si_V, np.repeat(prob.K, net.cells), 0.1, np.ones(prob.S)), tol=1e-3, maxiter=300, callback=callback)
    bb=time.time()
    np.save(os.path.join(path_matrices, 'sol_gmres.npy'), sol_gmres[0])
    t=bb-a
    arr_time.append(t)
    arr_tol.append(sol_gmres[1])
    np.save(os.path.join(path_matrices, 'time.npy'), np.array([arr_time]))
    np.savetxt(os.path.join(path_matrices, 'tol.txt'), np.array(arr_tol[1]), fmt='%f')

    
    
    
sol=np.load(os.path.join(path_matrices, 'sol.npy'))


#%% - Data for Avizo
from PrePostTemp import GetEdgesConcentration,GetSingleEdgeSources, GetEdgesConcentration, LabelVertexEdge
from PrePostTemp import GetPointsAM, GetConcentrationAM, GetConcentrationVertices


title="\n@7 # FlowRate"
np.savetxt(os.path.join(path_am, "FlowRate.txt"), net.U*net.R**2*np.pi, fmt='%f', delimiter=' ', header=title, comments='')
edges_concentration=GetEdgesConcentration(net.cells, prob.Cv)
vertex_label, edge_label=LabelVertexEdge(vertex_to_edge, startVertex)
title="\n@8 # EdgeConcentration"
np.savetxt(os.path.join(path_am, "EdgeConcentration.txt"), edges_concentration, fmt='%f', delimiter=' ', header=title, comments='')
title="\n@9 # EntryExitVertex"
np.savetxt(os.path.join(path_am, "EntryExitVertex.txt"), vertex_label.astype(int), fmt='%d', delimiter=' ', header=title, comments='')

title="\n@10 # EntryExitEdge"
np.savetxt(os.path.join(path_am, "EntryExitEdge.txt"), edge_label.astype(int), fmt='%d', delimiter=' ', header=title, comments='')

if simple_plotting:
    for i in np.where(edge_label==2)[0][:]: #1 if entering, 2 if exiting
        #plt.plot(GetSingleEdgeConc(net.cells, prob.Cv, i))
        plt.plot(prob.Cv[GetSingleEdgeSources(net.cells, i)])
        plt.plot(np.zeros(net.cells[i])+ edges_concentration[i])
        #plt.ylim((0.98,1))
    plt.show()

points_position=GetPointsAM(edges, pos_vertex, net.pos_s, net.cells)
title="\n@4 # EdgePointCoordinates"
np.savetxt(os.path.join(path_am, "EdgePointCoordinates.txt"), points_position, fmt='%f', delimiter=' ', header=title, comments='')

title="\n@3 # NumEdgePoints"
np.savetxt(os.path.join(path_am, "NumEdgePoints.txt"), net.cells+2, fmt='%d', delimiter=' ', header=title, comments='')

title="\n@2 # EdgeConnectivity"
np.savetxt(os.path.join(path_am, "EdgeConnectivity.txt"), edges, fmt='%d', delimiter=' ', header=title, comments='')

vertices_diams=GetConcentrationVertices(vertex_to_edge, startVertex, net.cells, np.repeat(diameters, net.cells))
points_thickness=GetConcentrationAM(edges, vertices_diams, np.repeat(diameters, net.cells), net.cells)
title="\n@5 # Thickness"
np.savetxt(os.path.join(path_am, "Thickness.txt"), points_thickness, fmt='%f', delimiter=' ', header=title, comments='')

vertices_pressure=GetConcentrationVertices(vertex_to_edge, startVertex, cells_per_segment, Pressure)
title="\n@6 # VertexPressure"
np.savetxt(os.path.join(path_am, "VertexPressure.txt"), vertices_pressure, fmt='%f', delimiter=' ', header=title, comments='')

#%%

vertices_concentration=GetConcentrationVertices(vertex_to_edge, startVertex, net.cells, prob.Cv)
title="\n@11 # VertexConcentration"
np.savetxt(os.path.join(path_am, "VertexConcentration.txt"), vertices_concentration, fmt='%f', delimiter=' ', header=title, comments='')

points_concentration=GetConcentrationAM(edges, vertices_concentration, prob.Cv, net.cells)
title="\n@12 # PointConcentration"
np.savetxt(os.path.join(path_am, "PointConcentration.txt"), points_concentration, fmt='%f', delimiter=' ', header=title, comments='')
