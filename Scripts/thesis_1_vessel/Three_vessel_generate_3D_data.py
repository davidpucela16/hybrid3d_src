#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:52:07 2023

@author: pdavid
"""
import os 
import sys

script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)

cells_3D=7
n=int(cells_3D/4)*20
name_script=script[script.rfind('/')+1:-3]
print(path_script)
path_src=os.path.join(path_script, '../../src')
path_potentials=os.path.join(path_script, '../../Potentials')
#Now the creation of the relevant folders to store the output
path_output=os.path.join(path_src, '../../output_figures/' + name_script)
path_matrices=os.path.join(path_output,"F{}_n{}".format(cells_3D, n))
#Directory to save the divided fiiles of the network
path_am=os.path.join(path_matrices, "am")

os.makedirs(path_matrices, exist_ok=True)
os.makedirs(path_output, exist_ok=True)

sys.path.append(path_src)
sys.path.append(path_potentials)


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg
import math
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import pdb
import matplotlib.pylab as pylab
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)

from Potentials_module import Classic
from Potentials_module import Gjerde
from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from mesh import cart_mesh_3D
from mesh_1D import mesh_1D
from GreenFast import GetSourcePotential
from hybridFast import hybrid_set_up
from neighbourhood import GetNeighbourhood, GetUncommon
from assembly_1D import FullAdvectionDiffusion1D


BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])
L_vessel=240

L_3D=np.array([L_vessel, L_vessel, 3*L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.AssemblyBoundaryVectors()


#%%
# - This is the validation of the 1D transport eq without reaction

D = 1
K=np.array([0.0001,10,0.0001])

U = np.array([2,2,2])*100/L_vessel
alpha=10
R_vessel=L_vessel/alpha
R_1D=np.zeros(3)+R_vessel

startVertex=np.array([0,1,2])
endVertex=np.array([1,2,3])
pos_vertex=np.array([[L_vessel/2, L_vessel/2, 1],
                     [L_vessel/2,L_vessel/2, L_vessel],
                     [L_vessel/2, L_vessel/2, 2*L_vessel],
                     [L_vessel/2,L_vessel/2, L_vessel*3-1]
                     ])

vertex_to_edge=[[0],[0,1], [1,2], [2]]
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])

#%% - Fine Problem

from PrePostTemp import VisualizationTool

alpha=20
R_vessel=L_vessel/alpha
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
cells_per_vessel=30
cpv=cells_per_vessel
h=L_vessel/cells_per_vessel
#Create 1D objects

path_kk2=path_matrices

M=0
l=50
c=-1
U = np.array([1,1,1])*l/L_vessel
k=1
net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)
c+=1
#Assign the current value of the permeability (Dahmkoler membrane)
K=np.array([0.0001,k,0.0001])
BCs_1D=np.array([[0,1],
                 [3,0]])

prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
var=False
prob.phi_bar_bool=var
prob.B_assembly_bool=var
prob.I_assembly_bool=var
#True if no need to compute
phi_bar_bool=os.path.exists(os.path.join(path_matrices, 'phi_bar_q.npz')) and os.path.exists(os.path.join(path_matrices, 'phi_bar_s.npz')) 
B_assembly_bool=os.path.exists(os.path.join(path_matrices, 'B_matrix.npz'))
I_assembly_bool=os.path.exists(os.path.join(path_matrices, 'I_matrix.npz'))
prob.phi_bar_bool=phi_bar_bool
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool

rec_bool=False
Computation_bool= not os.path.exists(os.path.join(path_matrices, "sol_cyl.npy"))

if Computation_bool:
    prob.AssemblyProblem(path_kk2)
    prob.Full_ind_array[:prob.F]-=M*mesh.h**3
    ###################################################################
    # PROBLEM - Cyl
    ####################################################################
    #Create the object for the analytical potentials
    P=Classic(3*L_vessel, R_vessel)
    G_ij=P.get_single_layer_vessel(len(net.pos_s))/2/np.pi/R_vessel
    #The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
    new_E_matrix=G_ij+prob.q_portion
    prob.E_matrix=new_E_matrix
    Exact_full_linear=prob.ReAssemblyMatrices() 
    sol_cyl=dir_solve(Exact_full_linear, -prob.Full_ind_array)
    np.save(path_matrices + "/sol_cyl", sol_cyl)
else:
    sol_cyl=np.load(path_matrices + "/sol_cyl.npy")

prob.s=sol_cyl[:-2*prob.S]
prob.q=sol_cyl[-prob.S*2:-prob.S]
prob.Cv=sol_cyl[-prob.S:]
#%% - Write .am field

title="\n@1 # VertexCoordinates"
np.savetxt(os.path.join(path_output, "VertexCoordinates.txt"), pos_vertex, fmt='%f', delimiter=' ', header=title, comments='')

title="\n@2 # EdgeConnectivity"
np.savetxt(os.path.join(path_output, "EdgeConnectivity.txt"), np.vstack((startVertex, endVertex)).T, fmt='%d', delimiter=' ', header=title, comments='')

from PrePostTemp import VisualizationTool
res=100
num_processes=30
process=0 #This must be kept to zero for the parallel reconstruction to go right
perp_axis_res=res*3
path_vol_data=os.path.join(path_output, "vol_data")
os.makedirs(path_vol_data, exist_ok=True)
shrink_factor=((cells_3D-1)/cells_3D)

#%%
corners_2D=np.array([[0,0],[0,1],[1,0],[1,1]])*L_3D[0]*shrink_factor+L_3D[0]*(1/cells_3D/2)
if rec_bool:
    aaz=VisualizationTool(prob, 2,0,1, corners_2D, res)
    shrink_factor_perp=((mesh.cells[2]-1)/mesh.cells[2])
    aaz.GetVolumeData(num_processes, process, perp_axis_res, path_vol_data, shrink_factor_perp)

#%%
from PrePostTemp import GetPointsAM, GetConcentrationAM, GetConcentrationVertices

# =============================================================================
# def GetPointsAM(edges, pos_vertex, pos_s, cells_1D):
#     points_array=np.zeros(((0,3)), dtype=np.float64)
#     ed=-1
#     for i in edges:
#         ed+=1
#         init_pos=np.sum(cells_1D[:ed])
#         end_pos=np.sum(cells_1D[:ed+1])
#         local_arr=np.vstack((pos_vertex[i[0]], pos_s[init_pos:end_pos],pos_vertex[i[1]]))
#         points_array=np.vstack((points_array, local_arr))
#     return points_array
# 
# def GetConcentrationAM(edges, pos_vertex, pos_s, cells_1D, Cv):
#     points_array=np.zeros(0, dtype=np.float64)
#     ed=-1
#     for i in edges:
#         ed+=1
#         init_pos=np.sum(cells_1D[:ed])
#         end_pos=np.sum(cells_1D[:ed+1])
#         local_arr=np.vstack((pos_vertex[i[0]], pos_s[init_pos:end_pos],pos_vertex[i[1]]))
#         points_array=np.vstack((points_array, local_arr))
#     return points_array
# 
# from PrePostTemp import GetSingleEdgeSources
# def GetConcentrationVertices(vertex_to_edge, starVertex, cells_per_segment):
#     value_array=np.zeros(0)
#     for i in range(len(pos_vertex)):
#         for ed in vertex_to_edge[i]:
#             value=0
#             sources=GetSingleEdgeSources(cells_per_segment,  ed)
#             if startVertex[ed]==i:
#                 value+=sources[0]
#             else:
#                 value+=sources[-1]
#         value/=len(vertex_to_edge[i])
#         value_array=np.append(value_array, value)
#     return value_array
# =============================================================================


edges=np.array([startVertex, endVertex]).T
points_position=GetPointsAM(edges, pos_vertex, net.pos_s, net.cells)


title="\n@3 # NumEdgePoints"
np.savetxt(os.path.join(path_output, "NumEdgePoints.txt"), np.repeat(cpv+2, np.array([3])), fmt='%d', delimiter=' ', header=title, comments='')

title="\n@4 # EdgePointCoordinates"
np.savetxt(os.path.join(path_output, "EdgePointCoordinates.txt"), points_position, fmt='%f', delimiter=' ', header=title, comments='')


#%%


from PrePostTemp import GetPointsAM, GetConcentrationAM, GetConcentrationVertices

vertices_concentration=GetConcentrationVertices(vertex_to_edge, startVertex, net.cells, prob.Cv)
title="@5 # vertices concentration"
np.savetxt(os.path.join(path_output, "vertices_concentration.txt"), vertices_concentration, fmt='%f', delimiter=' ', header=title, comments='')

points_concentration=GetConcentrationAM(edges, vertices_concentration, prob.Cv, net.cells)
title="@6 # concentration points"
np.savetxt(os.path.join(path_output, "Points.txt"), points_concentration, fmt='%f', delimiter=' ', header=title, comments='')




