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

name_script=script[script.rfind('/')+1:-3]

path_src=os.path.join(path_script, '../../src')
path_potentials=os.path.join(path_script, '../../Potentials')
#Now the creation of the relevant folders to store the output
path_matrices=os.path.join(path_src, '../../linear_system/' + name_script)
path_output=os.path.join(path_src, '../../output_figures/' + name_script)
path_phi_bar=os.path.join(path_matrices, 'path_phi_bar')
path_thesis=os.path.join(path_src, '../../path_thesis/' + name_script)

os.makedirs(path_phi_bar, exist_ok=True)
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


#%%
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])
L_vessel=240
cells_3D=11
n=11
L_3D=np.array([L_vessel, 3*L_vessel, L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.AssemblyBoundaryVectors()


#%%
# - This is the validation of the 1D transport eq without reaction

D = 1
K=np.array([0.0001,2,0.0001])*4

U = np.array([2,2,2])/L_vessel*10
startVertex=np.array([0,1,2])
endVertex=np.array([1,2,3])
pos_vertex=np.array([[L_vessel/2, 0, L_vessel/2],
                     [L_vessel/2, L_vessel,L_vessel/2],
                     [L_vessel/2, 2*L_vessel, L_vessel/2],
                     [L_vessel/2, L_vessel*3,L_vessel/2]
                     ])
BCs_1D=np.array([[0,1],
                 [3,0]]) #Not relevant since we are gonna set the intra concentration ourselves

vertex_to_edge=[[0],[0,1], [1,2], [2]]

temp_cpv=10
ref_cells=temp_cpv
alpha=12 #Aspect ratio
R_vessel=L_vessel/alpha
R_1D=np.zeros(3)+R_vessel
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
h=L_vessel/temp_cpv
    
net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)
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
q_exact=sol_2D[mesh.size_mesh:]
x_exact=net.pos_s[:,1]


plt.plot(x_exact[temp_cpv:temp_cpv*2], q_exact[temp_cpv:temp_cpv*2], label="Cylinder")
plt.plot(x_exact[temp_cpv:temp_cpv*2], q_line[temp_cpv:temp_cpv*2], label="Line")
plt.ylabel("q", rotation=0)
plt.xlabel("y")
plt.legend()
plt.savefig(path_output + "/alpha12.svg")
plt.show()
    

#%%
from PrePostTemp import Get1DCOMSOL


def GetComsol(x):
    closest_positions = np.argmin(np.abs(x_COMSOL[:, np.newaxis] - x), axis=0)
    return closest_positions



q_array_line=[]
q_array_cyl=[]
cells_array=np.array([20,50,100,200,500,1000])
alpha_array=np.array([5,10,15,20,25,30,35,40])

arr_line_error=np.zeros((len(cells_array), len(alpha_array)))
arr_cyl_error=np.zeros((len(cells_array), len(alpha_array)))

for c in range(len(cells_array)):
    for a in range(len(alpha_array)):
        
        file_path = os.path.join(path_output + '/constant_Cv/alpha_{}.txt'.format(alpha_array[a]))
        phi_bar_COMSOL, x_COMSOL = Get1DCOMSOL(file_path)
        pos_com=np.where((x_COMSOL<2*L_vessel) & (x_COMSOL>L_vessel))
        q_COMSOL=K[1]*(1-phi_bar_COMSOL[pos_com])
        
        temp_cpv=cells_array[c]
        alpha=alpha_array[a] #Aspect ratio
        R_vessel=L_vessel/alpha
        R_1D=np.zeros(3)+R_vessel
        diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
        h=L_vessel/temp_cpv
            
        net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
        net.U=U
        net.D=D
        net.PositionalArraysFast(mesh)
        prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
        mesh.GetOrderedConnectivityMatrix()
        x=net.pos_s[:,1]
            
        Cv=np.ones(3*temp_cpv)
        #Cv=np.concatenate((np.ones(cells_per_vessel), np.arange(cells_per_vessel)[::-1]/cells_per_vessel, np.zeros(cells_per_vessel)))

        prob.phi_bar_bool=False
        prob.B_assembly_bool=False
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
        if cells_array[c]<25:
            G_ij=P.get_single_layer_vessel_coarse(len(net.pos_s))/2/np.pi/R_vessel
        else:
            G_ij=P.get_single_layer_vessel(len(net.pos_s))/2/np.pi/R_vessel
        #The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
        new_E_matrix=G_ij+prob.q_portion
        Lin_matrix_2D=sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)), 
                                         sp.sparse.hstack((prob.D_matrix, new_E_matrix))))
        sol_2D=dir_solve(Lin_matrix_2D, b)
        prob.q=sol_2D[mesh.size_mesh:]
        prob.s=sol_2D[:mesh.size_mesh]
        q=sol_2D[mesh.size_mesh:]
        x=net.pos_s[:,1]
        q_array_cyl.append(q)
        q_array_line.append(q_line)
        
# =============================================================================
#         plt.plot(x[temp_cpv:temp_cpv*2], q[temp_cpv:temp_cpv*2], label="Cylinder")
#         plt.plot(x[temp_cpv:temp_cpv*2], q_line[temp_cpv:temp_cpv*2], label="Line")
#         plt.plot(x_COMSOL[pos_com], q_COMSOL, label="Reference")
#         #plt.plot(x_exact[ref_cells:ref_cells*2], q_exact[ref_cells:ref_cells*2], label="Exact")
#         plt.ylabel("q", rotation=0)
#         plt.xlabel("y")
#         plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
#         plt.legend()
#         plt.show()
# =============================================================================
        plt.plot(x[temp_cpv:temp_cpv*2], q[temp_cpv:temp_cpv*2], label="Cylinder")
        plt.plot(x[temp_cpv:temp_cpv*2], q_line[temp_cpv:temp_cpv*2], label="Line")
        plt.plot(x_COMSOL[pos_com], q_COMSOL, label="Reference")
        #plt.plot(x_exact[ref_cells:ref_cells*2], q_exact[ref_cells:ref_cells*2], label="Exact")
        plt.ylabel("q", rotation=0)
        plt.xlabel("y")
        plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
        plt.legend()
        plt.show()
        err_line=(np.average(q_line[temp_cpv:2*temp_cpv])-np.average(q_COMSOL))/np.average(q_COMSOL)
        err_cyl=(np.average(q[temp_cpv:2*temp_cpv])-np.average(q_COMSOL))/np.average(q_COMSOL)
        
        arr_line_error[c,a]=err_line
        arr_cyl_error[c,a]=err_cyl
np.save(path_output + "/arr_line_error" , arr_line_error)
np.save(path_output + "/arr_cyl_error" , arr_cyl_error)


#%% - Make a nice figure with the evaluation of the concentration at several distances
alpha=20
temp_cpv=100

R_vessel=L_vessel/alpha
R_1D=np.zeros(3)+R_vessel
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
h=L_vessel/temp_cpv
    
net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)
prob_cyl=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
x=net.pos_s[:,1]
Cv=np.ones(3*temp_cpv)

prob_cyl.phi_bar_bool=False
prob_cyl.B_assembly_bool=False
prob_cyl.AssemblyDEFFast(path_phi_bar, path_matrices)
prob_cyl.AssemblyABC(path_matrices)
Lin_matrix_1D=sp.sparse.vstack((sp.sparse.hstack((prob.A_matrix, prob.B_matrix)), 
                                 sp.sparse.hstack((prob.D_matrix, prob.q_portion+prob.Gij))))
    
b=np.concatenate((-prob.I_ind_array, np.zeros(len(net.pos_s)) - prob.F_matrix.dot(Cv)))
prob_line=prob_cyl.copy()
sol_line=dir_solve(Lin_matrix_1D, b)
q_line=sol_line[mesh.size_mesh:]
s_line=sol_line[:mesh.size_mesh]
prob_line.s=s_line
prob_line.q=q_line
prob_line.Cv=Cv


#Now we are gonna solve the same problem but using the elliptic integrals for the single layer 
P=Classic(3*L_vessel, R_vessel)
G_ij=P.get_single_layer_vessel(len(net.pos_s))/2/np.pi/R_vessel
#The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
new_E_matrix=G_ij+prob_cyl.q_portion
Lin_matrix_2D=sp.sparse.vstack((sp.sparse.hstack((prob_cyl.A_matrix, prob_cyl.B_matrix)), 
                                 sp.sparse.hstack((prob_cyl.D_matrix, new_E_matrix))))
sol_2D=dir_solve(Lin_matrix_2D, b)
prob_cyl.q=sol_2D[mesh.size_mesh:]
prob_cyl.s=sol_2D[:mesh.size_mesh]
q=sol_2D[mesh.size_mesh:]

plt.plot(x[temp_cpv:temp_cpv*2], q[temp_cpv:temp_cpv*2], label="Cylinder")
plt.plot(x[temp_cpv:temp_cpv*2], q_line[temp_cpv:temp_cpv*2], label="Line")
plt.plot(x_COMSOL[pos_com], q_COMSOL, label="Reference")
#plt.plot(x_exact[ref_cells:ref_cells*2], q_exact[ref_cells:ref_cells*2], label="Exact")
plt.ylabel("q", rotation=0)
plt.xlabel("y")
plt.title("Cells={}, alpha={}".format(temp_cpv, alpha))
plt.legend()
plt.show()
pdb.set_trace()
err_line=(np.average(q_line[temp_cpv:2*temp_cpv])-np.average(q_COMSOL))/np.average(q_COMSOL)
err_cyl=(np.average(q[temp_cpv:2*temp_cpv])-np.average(q_COMSOL))/np.average(q_COMSOL)

arr_line_error[c,a]=err_line
arr_cyl_error[c,a]=err_cyl



