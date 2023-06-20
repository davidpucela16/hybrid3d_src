#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:52:07 2023

@author: pdavid
"""

import os 
path=os.path.dirname(__file__)
path_matrices='/home/pdavid/Bureau/Code/BMF_Code/test_data/dipoles'
os.chdir(path)
from Potentials_module import Classic
from Potentials_module import Gjerde
path_src=os.path.join(path, '../src_final')
path_figures=os.path.join(path_matrices, "figures")
import sys 
sys.path.append(path_src)

import numpy as np

import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg
import math

import pdb

import matplotlib.pylab as pylab
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)


from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
import matplotlib.pyplot as plt

import math
from assembly_1D import FullAdvectionDiffusion1D
from mesh_1D import mesh_1D
from GreenFast import GetSourcePotential
import pdb

from hybridFast import hybrid_set_up

from neighbourhood import GetNeighbourhood, GetUncommon
import copy 

from PrePostTemp import GetCoarsePhi, GetProbArgs


#BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])

#BC_type=np.array(["Dirichlet", "Dirichlet","Dirichlet","Dirichlet", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])
L_vessel=240
cells_3D=7
n=int(cells_3D/4)*20
L_3D=np.array([L_vessel, 3*L_vessel, L_vessel])
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
pos_vertex=np.array([[L_vessel/2, 1, L_vessel/2],
                     [L_vessel/2, L_vessel,L_vessel/2],
                     [L_vessel/2, 2*L_vessel, L_vessel/2],
                     [L_vessel/2, L_vessel*3-1,L_vessel/2]
                     ])

vertex_to_edge=[[0],[0,1], [1,2], [2]]
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])

#%% - Fine Problem

from PrePostTemp import VisualizationTool

alpha=10
R_vessel=L_vessel/alpha
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])
cells_per_vessel=30
h=L_vessel/cells_per_vessel
#Create 1D objects
net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)
path_kk='/home/pdavid/Bureau/Code/BMF_Code/kk'


#%%
# =============================================================================
# l=1
# c=-1
# U = np.array([1,1,1])*5*l/L_vessel
# #for k in np.array([1,10,50,100,150, 200]):
# #M=0.5*1.2e-6
# M=0
# for k in np.array([0.1,0.5,1,5, 10,20,50])[::-1]:
#     h=L_vessel/cells_per_vessel
#     #Create 1D objects
#     net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
#     net.U=U
#     net.D=D
#     net.PositionalArraysFast(mesh)
#     #M=k*1.2e-6
#     title_string="_cells{}_K{}".format(cells_per_vessel, k)
#     c+=1
#     #Assign the current value of the permeability (Dahmkoler membrane)
#     K=np.array([0.0001,k,0.0001])
#     BCs_1D=np.array([[0,1],
#                      [3,0]])
#     
#     prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
#     if c==0: 
#         var=False
#     else:
#         var=True
#     prob.phi_bar_bool=var
#     prob.B_assembly_bool=var
#     prob.I_assembly_bool=var
#     prob.AssemblyProblem(path_kk)
#     prob.Full_ind_array[:prob.F]-=M*mesh.h**3
#     
#     ###################################################################
#     # PROBLEM - Cyl
#     ####################################################################
#     #Create the object for the analytical potentials
#     P=Classic(3*L_vessel, R_vessel)
#     G_ij=P.get_single_layer_vessel(len(net.pos_s))/2/np.pi/R_vessel
#     #The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
#     new_E_matrix=G_ij+prob.q_portion
#     #plt.plot(1/np.diagonal(prob.q_portion.toarray()))
#     prob.E_matrix=new_E_matrix
#     Exact_full_linear=prob.ReAssemblyMatrices() 
#     
#     sol_cyl=dir_solve(Exact_full_linear, -prob.Full_ind_array)
#     s_cyl=sol_cyl[:-2*prob.S]
#     q_cyl=sol_cyl[-prob.S*2:-prob.S]
#     Cv_cyl=sol_cyl[-prob.S:]
#     phi_bar_s=sp.sparse.load_npz(path_kk + '/phi_bar_s.npz')
#     phi_bar_cyl=phi_bar_s.dot(s_cyl)+G_ij.dot(q_cyl)
#     
# # =============================================================================
#     plt.plot(net.pos_s[:,1],phi_bar_cyl, label='Cylinder')
# #     
# #    plt.plot(net.pos_s[:,1],q_cyl, label='{}'.format(k))
# #     plt.plot(net.pos_s[:,1],Cv_cyl, label='Cylinder')
# # =============================================================================
#     res=50
#     prob.s=s_cyl
#     prob.q=q_cyl
#     prob.Cv=Cv_cyl
#     prob.q=q_cyl
#     prob.s=s_cyl
#     prob.Cv=Cv_cyl
# 
# # =============================================================================
# #     corners_2D=np.array([[0,0],[0,240],[240,0],[240,240]])*(cells_3D-1)/cells_3D+L_3D[0]/cells_3D/2
# #     aa=VisualizationTool(prob, 1,0,2, corners_2D, res)
# #     aa.pos_array=np.array([0.4,0.5,0.6,0.7])
# #     aa.GetPlaneData(path_kk)
# #     aa.PlotData(path_kk)
# # =============================================================================
# plt.legend()
# plt.show()
# corners_2D=np.array([[0,0],[0,240],[240,0],[240,240]])*(cells_3D-1)/cells_3D+L_3D[0]/cells_3D/2
# aay=VisualizationTool(prob, 1,0,2, corners_2D, res)
# aay.pos_array=np.array([0.4,0.5,0.6,0.7])
# aay.GetPlaneData(path_kk)
# aay.PlotData(path_kk)
# =============================================================================

#%%
M=0
l=1
c=-1
U = np.array([1,1,1])*5*l/L_vessel
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
#for k in np.array([50,100,150, 200]):
vessel_range=np.arange(cells_per_vessel, 2*cells_per_vessel)
for k in np.array([0.1,0.5,1,5, 10,20,100])[::-1]:
    h=L_vessel/cells_per_vessel
    #Create 1D objects
    net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
    net.U=U
    net.D=D
    net.PositionalArraysFast(mesh)
    
    title_string="_cells{}_K{}".format(cells_per_vessel, k)
    c+=1
    #Assign the current value of the permeability (Dahmkoler membrane)
    K=np.array([0.0001,k,0.0001])
    BCs_1D=np.array([[0,1],
                     [3,0]])
    
    prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
    if c==0: 
        var=False
    else:
        var=True
    prob.phi_bar_bool=var
    prob.B_assembly_bool=var
    prob.I_assembly_bool=var
    prob.AssemblyProblem(path_kk)
    prob.Full_ind_array[:prob.F]-=M*mesh.h**3
    prob.SolveProblem()
    q_line=prob.q.copy()
    Cv_line=prob.Cv.copy()
    phi_bar_line=prob.D_matrix.dot(prob.s) + (prob.Gij).dot(prob.q)
    
    
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
    s_cyl=sol_cyl[:-2*prob.S]
    q_cyl=sol_cyl[-prob.S*2:-prob.S]
    Cv_cyl=sol_cyl[-prob.S:]
    
    H_ij=P.get_double_layer_vessel(len(net.pos_s))
    #H_ij[:,:]=0
    F_matrix_dip=prob.F_matrix+H_ij
    E_matrix_dip=new_E_matrix-H_ij*1/K[net.source_edge]
    prob.E_matrix=E_matrix_dip
    Dip_full_linear=prob.ReAssemblyMatrices() 

    sol_dip=dir_solve(Dip_full_linear, -prob.Full_ind_array)
    s_dip=sol_dip[:-2*prob.S]
    q_dip=sol_dip[-prob.S*2:-prob.S]
    Cv_dip=sol_dip[-prob.S:]
    
    #Get phi bar
    phi_bar_s=sp.sparse.load_npz(path_kk + '/phi_bar_s.npz')
    phi_bar_q=sp.sparse.load_npz(path_kk + '/phi_bar_q.npz')
    
    phi_bar_line=phi_bar_s.dot(prob.s)+phi_bar_q.dot(prob.q)
    phi_bar_cyl=phi_bar_s.dot(s_cyl)+G_ij.dot(q_cyl)
    phi_bar_dip=Cv_dip - q_dip*np.diagonal(prob.q_portion.toarray())
    
    np.save(path_kk+ "/phi_line" + title_string, phi_bar_line)
    np.save(path_kk+ "/phi_cyl" + title_string, phi_bar_cyl)
    np.save(path_kk+ "/phi_dip" + title_string, phi_bar_dip)
    
    np.save(path_kk+ "/q_line" + title_string, q_line)
    np.save(path_kk+ "/q_cyl" + title_string, q_cyl)
    np.save(path_kk+ "/q_dip" + title_string, q_dip)
    
    np.save(path_kk+ "/Cv_line" + title_string, Cv_line)
    np.save(path_kk+ "/Cv_cyl" + title_string, Cv_cyl)
    np.save(path_kk+ "/Cv_dip" + title_string, Cv_dip)
    
# =============================================================================
#         plt.plot(net.pos_s[:,1],phi_bar_line, label='q_line')
#         plt.plot(net.pos_s[:,1],phi_bar_cyl, label='Cylinder')
#         plt.plot(net.pos_s[:,1],phi_bar_dip, label='exact')
#         plt.legend()
#         #plt.title("phi_bar for Pe={:.2f}, k={:.2f} and M={} and cells={}".format(U[1]*L_vessel, k, cells_per_vessel))
#         plt.savefig(path_kk + "/phi_bar" + title_string  + ".svg")
#         plt.show()
# =============================================================================
    
    #plt.plot(net.pos_s[:,1],q_line, label='q_line')
    #plt.plot(net.pos_s[:,1],q_cyl, label='q Cylinder')
    #plt.plot(net.pos_s[:,1],q_dip, label='q Cylinder + DL')
    #plt.legend()
    #plt.title("q for Pe={:.2f}, k={:.2f} and M={} and cells={}".format(U[1]*L_vessel, k,int(M*1.2e6), cells_per_vessel))
    #plt.savefig(path_kk + "/q" + title_string  + ".svg")
    #plt.plot(net.pos_s[:,1],Cv_line, label='Cv_line')
    plt.plot(net.pos_s[vessel_range,1],Cv_cyl[vessel_range],'--', color=colors[c], label='k={}'.format(k))
    plt.plot(net.pos_s[vessel_range,1],Cv_dip[vessel_range], color=colors[c])
    
    
# =============================================================================
#         res=50
#         prob.s=s_cyl
#         prob.q=q_cyl
#         prob.Cv=Cv_cyl
#     
#         corners_2D=np.array([[0,0],[0,240],[240,0],[240,240]])*(cells_3D-1)/cells_3D+L_3D[0]/cells_3D/2
#         aa=VisualizationTool(prob, 1,0,2, corners_2D, res)
#         aa.pos_array=np.array([0.4,0.5,0.6,0.7])
#         aa.GetPlaneData(path_kk)
#         aa.PlotData(path_kk)
# =============================================================================
legend=plt.legend()
legend.set_bbox_to_anchor((0.5, 0.5))
#plt.savefig(path_kk + "/Cv" + title_string  + ".svg")
plt.show()
#%%
U = np.array([1,1,1])*5*l/L_vessel
#for k in np.array([50,100,150, 200]):
c=0
for k in np.array([0.1,0.5,1,5, 10,20,100])[::-1]:
    h=L_vessel/cells_per_vessel
    #Create 1D objects
    net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
    net.U=U
    net.D=D
    net.PositionalArraysFast(mesh)
    
    title_string="_cells{}_K{}".format(cells_per_vessel, k)
    c+=1
    #Assign the current value of the permeability (Dahmkoler membrane)
    K=np.array([0.0001,k,0.0001])
    BCs_1D=np.array([[0,1],
                     [3,0]])
    
    prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
    if c==0: 
        var=False
    else:
        var=True
    prob.phi_bar_bool=var
    prob.B_assembly_bool=var
    prob.I_assembly_bool=var
    prob.AssemblyProblem(path_kk)
    prob.Full_ind_array[:prob.F]-=M*mesh.h**3
    prob.SolveProblem()
    q_line=prob.q.copy()
    Cv_line=prob.Cv.copy()
    phi_bar_line=prob.D_matrix.dot(prob.s) + (prob.Gij).dot(prob.q)
    
    
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
    s_cyl=sol_cyl[:-2*prob.S]
    q_cyl=sol_cyl[-prob.S*2:-prob.S]
    Cv_cyl=sol_cyl[-prob.S:]
    
    H_ij=P.get_double_layer_vessel(len(net.pos_s))
    #H_ij[:,:]=0
    F_matrix_dip=prob.F_matrix+H_ij
    E_matrix_dip=new_E_matrix-H_ij*1/K[net.source_edge]
    prob.E_matrix=E_matrix_dip
    Dip_full_linear=prob.ReAssemblyMatrices() 

    sol_dip=dir_solve(Dip_full_linear, -prob.Full_ind_array)
    s_dip=sol_dip[:-2*prob.S]
    q_dip=sol_dip[-prob.S*2:-prob.S]
    Cv_dip=sol_dip[-prob.S:]
    
    #Get phi bar
    phi_bar_s=sp.sparse.load_npz(path_kk + '/phi_bar_s.npz')
    phi_bar_q=sp.sparse.load_npz(path_kk + '/phi_bar_q.npz')
    
    phi_bar_line=phi_bar_s.dot(prob.s)+phi_bar_q.dot(prob.q)
    phi_bar_cyl=phi_bar_s.dot(s_cyl)+G_ij.dot(q_cyl)
    phi_bar_dip=Cv_dip - q_dip*np.diagonal(prob.q_portion.toarray())
    
    np.save(path_kk+ "/phi_line" + title_string, phi_bar_line)
    np.save(path_kk+ "/phi_cyl" + title_string, phi_bar_cyl)
    np.save(path_kk+ "/phi_dip" + title_string, phi_bar_dip)
    
    np.save(path_kk+ "/q_line" + title_string, q_line)
    np.save(path_kk+ "/q_cyl" + title_string, q_cyl)
    np.save(path_kk+ "/q_dip" + title_string, q_dip)
    
    np.save(path_kk+ "/Cv_line" + title_string, Cv_line)
    np.save(path_kk+ "/Cv_cyl" + title_string, Cv_cyl)
    np.save(path_kk+ "/Cv_dip" + title_string, Cv_dip)
    
# =============================================================================
#         plt.plot(net.pos_s[:,1],phi_bar_line, label='q_line')
#         plt.plot(net.pos_s[:,1],phi_bar_cyl, label='Cylinder')
#         plt.plot(net.pos_s[:,1],phi_bar_dip, label='exact')
#         plt.legend()
#         #plt.title("phi_bar for Pe={:.2f}, k={:.2f} and M={} and cells={}".format(U[1]*L_vessel, k, cells_per_vessel))
#         plt.savefig(path_kk + "/phi_bar" + title_string  + ".svg")
#         plt.show()
# =============================================================================
    
    #plt.plot(net.pos_s[:,1],q_line, label='q_line')
    #plt.plot(net.pos_s[:,1],q_cyl, label='q Cylinder')
    #plt.plot(net.pos_s[:,1],q_dip, label='q Cylinder + DL')
    #plt.legend()
    #plt.title("q for Pe={:.2f}, k={:.2f} and M={} and cells={}".format(U[1]*L_vessel, k,int(M*1.2e6), cells_per_vessel))
    #plt.savefig(path_kk + "/q" + title_string  + ".svg")
    #plt.plot(net.pos_s[:,1],Cv_line, label='Cv_line')
    #plt.plot(net.pos_s[:,1],Cv_cyl,'--', label='C, k={}'.format(k))
    #plt.plot(net.pos_s[:,1],Cv_dip, label='C + DL, k={}'.format(k))
    #plt.plot(net.pos_s[:,1],Cv_line, label='Cv_line')
    plt.plot(net.pos_s[vessel_range,1],q_cyl[vessel_range],'--', color=colors[c-1], label='k={}'.format(k))
    plt.plot(net.pos_s[vessel_range,1],q_dip[vessel_range], color=colors[c-1])
    
    
# =============================================================================
#         res=50
#         prob.s=s_cyl
#         prob.q=q_cyl
#         prob.Cv=Cv_cyl
#     
#         corners_2D=np.array([[0,0],[0,240],[240,0],[240,240]])*(cells_3D-1)/cells_3D+L_3D[0]/cells_3D/2
#         aa=VisualizationTool(prob, 1,0,2, corners_2D, res)
#         aa.pos_array=np.array([0.4,0.5,0.6,0.7])
#         aa.GetPlaneData(path_kk)
#         aa.PlotData(path_kk)
# =============================================================================
legend=plt.legend()
legend.set_bbox_to_anchor((0.5, 0.5))
#plt.savefig(path_kk + "/Cv" + title_string  + ".svg")
plt.show()