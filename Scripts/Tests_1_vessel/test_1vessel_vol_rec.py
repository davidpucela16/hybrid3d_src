#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:47:23 2023

@author: pdavid
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
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
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg

#Set up the correct path to import the source code
import os 
import sys 

script = os.path.abspath(sys.argv[0])
path_script = os.path.dirname(script)

path_src=os.path.join(path_script, '../hybrid_3d_clean/src_final')
path_matrices=os.path.join(path_script, "matrices")
path_vol_data=os.path.join(path_matrices, "vol_data")
os.makedirs(path_vol_data, exist_ok=True)
os.makedirs(path_matrices, exist_ok=True)

sys.path.append(path_src)

from assembly_1D import FullAdvectionDiffusion1D
from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from mesh import cart_mesh_3D
from mesh_1D import mesh_1D
from GreenFast import GetSourcePotential
from hybridFast import hybrid_set_up
from post_processing import GetPlaneReconstructionFast, ReconstructionCoordinatesFast
from neighbourhood import GetNeighbourhood, GetUncommon


#%%

BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

cells=20
n=int(cells/4)
L=np.array([240,480,240])
mesh=cart_mesh_3D(L,cells)

mesh.AssemblyBoundaryVectors()

# - This is the validation of the 1D transport eq without reaction
U = np.array([0.1])
D = 1
K=1
L_vessel = L[1]

alpha=100
R=L[0]/alpha
cells_1D = np.array([cells*2])



startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L[0]/2, 0.1, L[0]/2],[L[0]/2, L_vessel-0.1,L[0]/2]])
vertex_to_edge=[[0],[0]]
diameters=np.array([2*R])
h=np.array([L_vessel])/cells_1D

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)

BCs_1D=np.array([[0,1],
                 [1,0]])

prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)

mesh.GetOrderedConnectivityMatrix()
#True if no need to compute
prob.phi_bar_bool=False
prob.B_assembly_bool=False
prob.I_assembly_bool=False
#True if compute
Computation_bool=False
rec_bool=True
res=50

#%% - We solve the 2D problem in a 3D setting to be able to validate. Ideally, there is no transport in the y direction
if Computation_bool:
    prob.AssemblyProblem(path_matrices)
    print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/net.L)
    C_v_array=np.ones(len(net.pos_s)) #Though this neglects completely the contribution from the dipoles
    
    L1=sp.sparse.hstack((prob.A_matrix,prob.B_matrix))
    L2=sp.sparse.hstack((prob.D_matrix, prob.Gij+prob.q_portion))
    
    Li=sp.sparse.vstack((L1,L2))
    
    M=1.2/1.2e5
    
    ind=np.concatenate((prob.I_ind_array-M*mesh.h**3, prob.F_matrix.dot(C_v_array)))
    
    sol=dir_solve(Li, -ind)
    
    
    plt.plot(net.pos_s[:,1],sol[-prob.S:], label='hybrid reaction')
    plt.title("q(s) with C_v=1")
    plt.legend()
    plt.show()
    
    prob.s=sol[:-prob.S]
    prob.q=sol[-prob.S:]
    prob.Cv=C_v_array
    sol=np.concatenate((sol, C_v_array))
    np.save(os.path.join(path_matrices, 'sol'),sol)
    
    
    corners=np.array([[0,240,0], [0,240,240],[240,240,0], [240,240,240]])
    kk=GetPlaneReconstructionFast(240,1, 0, 2,  corners, res, prob, prob.Cv)
    plt.imshow(kk[0])
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar()
    plt.show()
    
    corners=np.array([[0,0,120], [0,480,120],[240,0,120], [240,480,120]])
    kk=GetPlaneReconstructionFast(240,2, 0, 1,  corners, res, prob, prob.Cv)
    plt.imshow(kk[0], origin="lower", extent=[0,L[0],0,L[1]])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()
    corners=np.array([[120,0,0],[120,0,240],[120,480,0],[120,480,240]])
    kk=GetPlaneReconstructionFast(240,0, 1, 2,  corners, res, prob, prob.Cv)
    plt.imshow(kk[0], origin="lower", extent=[0,L[1],0,L[2]])
    plt.xlabel("y")
    plt.ylabel("z")
    plt.colorbar()
    plt.show()
sol=np.load(os.path.join(path_matrices, "sol.npy"))
prob.q=sol[-2*prob.S:-prob.S]
prob.s=sol[:-prob.S]
prob.Cv=sol[-prob.S:]
#%%


from PrePostTemp import VisualizationTool
if rec_bool:
    num_processes=10
    process=0 #This must be kept to zero for the parallel reconstruction to go right
    perp_axis_res=50
    aaz=VisualizationTool(prob, 2,0,1, np.array([[0,0],[0,L[0]],[L[0],0],[L[0], L[0]]]), res)
    aaz.GetVolumeData(num_processes, process, perp_axis_res, path_vol_data)


