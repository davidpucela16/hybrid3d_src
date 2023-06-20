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
print(path_script)

name_script = script[script.rfind('/')+1:-3]

path_src = os.path.join(path_script, '../../src')
path_potentials = os.path.join(path_script, '../../Potentials')
# Now the creation of the relevant folders to store the output
path_matrices = os.path.join(path_src, '../../linear_system/' + name_script)
path_output = os.path.join(path_src, '../../output_figures/' + name_script)
path_phi_bar = os.path.join(path_matrices, 'path_phi_bar')
path_thesis = os.path.join(path_src, '../../path_thesis/' + name_script)

os.makedirs(path_phi_bar, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)
os.makedirs(path_output, exist_ok=True)

sys.path.append(path_src)
sys.path.append(path_potentials)

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


BC_type = np.array(["Neumann", "Neumann", "Neumann",
                   "Neumann", "Neumann", "Neumann"])
BC_type = np.array(["Dirichlet", "Dirichlet", "Neumann",
                   "Neumann", "Dirichlet", "Dirichlet"])
BC_value = np.array([0, 0, 0, 0, 0, 0])
L_vessel = 240
cells_3D = 5
n = 11
L_3D = np.array([L_vessel, 3*L_vessel, L_vessel])

D = 1
K = np.array([0.0001, 1, 0.0001])*8
print(K)
U = np.array([2, 2, 2])/L_vessel*10
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

temp_cpv = 100
alpha = 50 # Aspect ratio
R_vessel = L_vessel/alpha
R_1D = np.zeros(3)+R_vessel
diameters = np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
h = L_vessel/temp_cpv

mesh = cart_mesh_3D(L_3D, cells_3D)
h = L_vessel/temp_cpv
net = mesh_1D(startVertex, endVertex, vertex_to_edge,
              pos_vertex, diameters, h, 1)
net.U = U
net.D = D
net.PositionalArraysFast(mesh)
mesh.AssemblyBoundaryVectors()


prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
mesh.GetOrderedConnectivityMatrix()
x=net.pos_s[:,1]
    
Cv=np.ones(3*temp_cpv)
#Cv=np.concatenate((np.ones(cells_per_vessel), np.arange(cells_per_vessel)[::-1]/cells_per_vessel, np.zeros(cells_per_vessel)))
    
prob.AssemblyDEFFast(path_phi_bar, path_matrices)
prob.AssemblyABC(path_matrices)
Lin_matrix_1D=sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)), 
                                 sp.sparse.hstack((prob.D_matrix, prob.q_portion+prob.Gij))))
    
b=np.concatenate((-prob.I_ind_array, np.zeros(len(net.pos_s)) - prob.F_matrix.dot(Cv)))
sol_line=dir_solve(Lin_matrix_1D, b)
q_line=sol_line[mesh.size_mesh:]
s_line=sol_line[:mesh.size_mesh]

#Now we are gonna solve the same problem but using the elliptic integrals for the single layer 
P=Classic(3*L_vessel, R_vessel)
G_ij=P.get_single_layer_vessel(len(net.pos_s))/2/np.pi/R_vessel
#The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
new_E_matrix=G_ij+prob.q_portion
Lin_matrix_2D=sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)), 
                                 sp.sparse.hstack((prob.D_matrix, new_E_matrix))))
sol_2D=dir_solve(Lin_matrix_2D, b)
prob.q=sol_2D[mesh.size_mesh:]
prob.s=sol_2D[:mesh.size_mesh]
s_cyl=sol_2D[:mesh.size_mesh]
q_cyl=sol_2D[mesh.size_mesh:]
x_cyl=net.pos_s[:,1]

H_ij=P.get_double_layer_vessel(len(net.pos_s))
F_matrix_dip=prob.F_matrix+H_ij
E_matrix_dip=new_E_matrix-H_ij*1/K[net.source_edge]
prob.E_matrix=E_matrix_dip
Lin_matrix_dip = sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)),
                                  sp.sparse.hstack((prob.D_matrix, E_matrix_dip))))
b = np.concatenate(
    (-prob.I_ind_array, np.zeros(len(net.pos_s)) - np.dot(np.array([F_matrix_dip]), Cv)[0]))

sol_dip = dir_solve(Lin_matrix_2D, b)
q_dip = sol_dip[mesh.size_mesh:]



file_path = os.path.join(path_output + '/COMSOL_data/alpha_{}_q.txt'.format(alpha))
q_COMSOL, x_COMSOL = Get1DCOMSOL(file_path)
pos_com=np.where((x_COMSOL<2*L_vessel) & (x_COMSOL>L_vessel))
def GetComsol(x):
    closest_positions = np.argmin(np.abs(x_COMSOL[:, np.newaxis] - x), axis=0)
    return closest_positions

plt.plot(x_cyl[temp_cpv:temp_cpv*2], q_cyl[temp_cpv:temp_cpv*2], label="Cylinder")
plt.plot(x_cyl[temp_cpv:temp_cpv*2], q_line[temp_cpv:temp_cpv*2], label="Line")
plt.plot(x_cyl[temp_cpv:temp_cpv*2], q_dip[temp_cpv:temp_cpv*2], label="dip")
plt.plot(x_COMSOL[pos_com], q_COMSOL[pos_com], label="Reference")
plt.ylabel("q", rotation=0)
plt.xlabel("y")
plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
plt.legend()
#plt.savefig(path_output + "/alpha12.svg")
plt.show()

#%%

phi_bar_cyl=1-q_cyl/K[1]
phi_bar_line=1-q_line/K[1]
phi_bar_dip=1-q_dip/K[1]
file_path=os.path.join(path_output, 'COMSOL_data/alpha_{}_phi_bar_Rv{}.txt'.format(alpha, 0))
kk, x_C = Get1DCOMSOL(file_path)
phi_COMSOL=kk
x_COMSOL=x_C

plt.plot(x_cyl, phi_bar_cyl, '--')
plt.plot(x_cyl, phi_bar_line, '.-')
plt.plot(x_cyl, phi_bar_dip, linestyle='dotted')
plt.plot(x_COMSOL, phi_COMSOL)
plt.ylabel("$\overline{\phi}$", rotation=0)
plt.xlabel("y")
plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
plt.legend()
plt.xlim((L_vessel, 2*L_vessel))
plt.ylim((0.2,1.1))


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




#%%




from post_processing import ReconstructionCoordinatesFast, GetProbArgs

phi_cyl=[]
phi_line=[]
phi_COMSOL=[]
x_COMSOL=[]
for i in range(4):
    if i==0:
        crds=np.vstack((L_vessel/2+np.ones([prob.S]), x_cyl, L_vessel/2+ np.zeros([prob.S]))).T
    else:
        crds=np.vstack((L_vessel/2+np.ones([prob.S])*R_vessel*(i+1), x_cyl, L_vessel/2+ np.zeros([prob.S]))).T
    phi_line.append(ReconstructionCoordinatesFast(crds, GetProbArgs(prob), s_line, q_line))
    phi_cyl.append(ReconstructionCoordinatesFast(crds, GetProbArgs(prob), s_cyl, q_cyl))
    file_path=os.path.join(path_output, 'COMSOL_data/alpha_{}_phi_bar_Rv{}.txt'.format(alpha, i))
    kk, x_C = Get1DCOMSOL(file_path)
    phi_COMSOL.append(kk)
    x_COMSOL.append(x_C)

#%%





#%%
colors = ['blue', 'red', 'green', 'purple']

for i in range(4):
    plt.plot(x_cyl, phi_cyl[i], '--', color=colors[i])
    plt.plot(x_cyl, phi_line[i], '.-', color=colors[i])
    plt.plot(x_COMSOL[i], phi_COMSOL[i],color=colors[i])
    plt.ylabel("$\overline{\phi}$", rotation=0)
    plt.xlabel("y")
    plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
    plt.legend()
    plt.xlim((L_vessel, 2*L_vessel))
    plt.ylim((0.2,1.1))
    #plt.savefig(path_output + "/alpha12.svg")
plt.legend()

