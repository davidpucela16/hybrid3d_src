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

temp_cpv = 10
alpha = 50  # Aspect ratio
R_vessel = L_vessel/alpha
R_1D = np.zeros(3)+R_vessel
diameters = np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
h = L_vessel/temp_cpv


# %% - COMSOL Data
from PrePostTemp import Get1DCOMSOL
# Define the file path
file_path = path_output + '/data_COMSOL.txt'
phi_bar_COMSOL, x_COMSOL = Get1DCOMSOL(file_path)

# %%
# - This is the validation of the 1D transport eq without reaction
q_array = np.zeros((0, 3*temp_cpv))
phi_bar_list=[]
x_list=[]

def GetComsol(x):
    closest_positions = np.argmin(np.abs(x_COMSOL[:, np.newaxis] - x), axis=0)
    return closest_positions
#for cells_3D in np.array([5, 9, 13,19]):
cells_1D_array=np.array([3,5,10,20,30,40,50])
for temp_cpv in cells_1D_array:
    mesh = cart_mesh_3D(L_3D, cells_3D)
    h = L_vessel/temp_cpv
    net = mesh_1D(startVertex, endVertex, vertex_to_edge,
                  pos_vertex, diameters, h, 1)
    net.U = U
    net.D = D
    net.PositionalArraysFast(mesh)
    mesh.AssemblyBoundaryVectors()
    prob = hybrid_set_up(mesh, net, BC_type, BC_value, n,
                         1, np.zeros(len(diameters))+K, BCs_1D)
    mesh.GetOrderedConnectivityMatrix()
    x = net.pos_s[:, 1]

    # Linearly decaying concentration profile
    Cv = np.ones(3*temp_cpv)
    Cv[temp_cpv:temp_cpv*2] = np.arange(temp_cpv)[::-1]/temp_cpv
    Cv[temp_cpv*2:] = 0
    #Cv=np.concatenate((np.ones(cells_per_vessel), np.arange(cells_per_vessel)[::-1]/cells_per_vessel, np.zeros(cells_per_vessel)))
    current_folder=path_matrices + '/disc_{}_cpv{}'.format(cells_3D, temp_cpv)
    os.makedirs(current_folder, exist_ok=True)
    prob.B_assembly_bool=os.path.exists(current_folder + '/B_matrix.npz')
    prob.phi_bar_bool=os.path.exists(current_folder + '/phi_bar_s.npz')
    prob.AssemblyDEFFast(path_phi_bar, current_folder)
    prob.AssemblyABC(current_folder)
    
    Lin_matrix_1D = sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)),
                                      sp.sparse.hstack((prob.D_matrix, prob.q_portion+prob.Gij))))

    b = np.concatenate(
        (-prob.I_ind_array, np.zeros(len(net.pos_s)) - prob.F_matrix.dot(Cv)))
    sol_line = dir_solve(Lin_matrix_1D, b)
    q_line = sol_line[mesh.size_mesh:]
    s_line = sol_line[:mesh.size_mesh]

    # Now we are gonna solve the same problem but using the elliptic integrals for the single layer
    P = Classic(3*L_vessel, R_vessel)
    
    
    if temp_cpv>25:
        G_ij = P.get_single_layer_vessel(len(net.pos_s))/2/np.pi/R_vessel
    else:
        G_ij = P.get_single_layer_vessel_coarse(len(net.pos_s), 10)/2/np.pi/R_vessel
    
    # The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
    new_E_matrix = G_ij+prob.q_portion
    Lin_matrix_2D = sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)),
                                      sp.sparse.hstack((prob.D_matrix, new_E_matrix))))
    sol_2D = dir_solve(Lin_matrix_2D, b)
    prob.q = sol_2D[mesh.size_mesh:]
    prob.s = sol_2D[:mesh.size_mesh]
    q_exact = sol_2D[mesh.size_mesh:]
    x_exact = net.pos_s[:, 1]
    
    
    if temp_cpv>25:
        H_ij=P.get_double_layer_vessel(len(net.pos_s))
    else:
        H_ij=P.get_double_layer_vessel_coarse(len(net.pos_s), 10)
    #H_ij[:,:]=0
    F_matrix_dip=prob.F_matrix+H_ij
    E_matrix_dip=new_E_matrix-H_ij*1/K[net.source_edge]
    prob.E_matrix=E_matrix_dip
    Lin_matrix_dip = sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)),
                                      sp.sparse.hstack((prob.D_matrix, E_matrix_dip))))
    b = np.concatenate(
        (-prob.I_ind_array, np.zeros(len(net.pos_s)) - np.dot(np.array([F_matrix_dip]), Cv)[0]))
    
    sol_dip = dir_solve(Lin_matrix_2D, b)
    q_dip = sol_dip[mesh.size_mesh:]
    
    phi_bar_line=Cv-q_line/np.repeat(K, net.cells)
    phi_bar_cyl=Cv-q_exact/np.repeat(K, net.cells)
    phi_bar_dip=Cv-q_dip/np.repeat(K, net.cells)
    
    phi_bar_list.append(phi_bar_cyl)
    
    x_list.append(x_exact)
    
    plt.plot(x_exact[temp_cpv:temp_cpv*2],
             phi_bar_cyl[temp_cpv:temp_cpv*2], label="Cylinder")
    plt.plot(x_exact[temp_cpv:temp_cpv*2],
             phi_bar_line[temp_cpv:temp_cpv*2], label="Line")
    

    plt.plot(x_exact[temp_cpv:temp_cpv*2],
             phi_bar_dip[temp_cpv:temp_cpv*2], label="dip")
    pos_com=np.where((x_COMSOL<x_exact[temp_cpv*2-1]) & (x_COMSOL>x_exact[temp_cpv]))
    plt.plot(x_COMSOL[pos_com],phi_bar_COMSOL[pos_com] , label='Reference')
    plt.ylabel("q", rotation=0)
    plt.xlabel("y")
    plt.legend(loc="lower left")
    plt.title('disc_{}_cpv{}'.format(cells_3D, temp_cpv))
    #plt.savefig(path_output + "/alpha12.svg")
    plt.show()
    #q_array = np.vstack((q_array, q_exact))
    #phi_bar_array = np.vstack((phi_bar_array, Cv-q_exact/np.repeat(K, net.cells)))



plt.plot(x_COMSOL[pos_com],phi_bar_COMSOL[pos_com] , label='COMSOL', linewidth=4)
for i in range(len(cells_1D_array)):
    temp_cpv=cells_1D_array[i]
    x_exact=x_list[i]
    phi_bar_cyl=phi_bar_list[i]
    plt.plot(x_exact[temp_cpv:temp_cpv*2],phi_bar_cyl[temp_cpv:temp_cpv*2], label="$L_v/h=${}".format(temp_cpv))
    
plt.legend()
plt.ylabel('$\overline{\phi}$', rotation=0, labelpad=20)
plt.xlabel('$z (\mu m)$')
plt.savefig(path_output + "/phi_bar_convergence.svg")