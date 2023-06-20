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
path=os.path.dirname(__file__)
path_src=os.path.join(path, '../hybrid_3d_clean/src_final')
os.chdir(path_src)

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
prob.phi_bar_bool=False
prob.B_assembly_bool=False
prob.I_assembly_bool=False
prob.AssemblyProblem(path + "/matrices")

print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/net.L)


#%% - We solve the 2D problem in a 3D setting to be able to validate. Ideally, there is no transport in the y direction
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
res=100

corners=np.array([[0,240,0], [0,240,240],[240,240,0], [240,240,240]])
kk=GetPlaneReconstructionFast(240,1, 0, 2,  corners, res, prob, prob.Cv)
plt.imshow(kk[0])
plt.xlabel("x")
plt.ylabel("z")
plt.colorbar()
plt.show()

#%%
corners=np.array([[0,0,120], [0,480,120],[240,0,120], [240,480,120]])
kk=GetPlaneReconstructionFast(240,2, 0, 1,  corners, res, prob, prob.Cv)
plt.imshow(kk[0], origin="lower", extent=[0,L[0],0,L[1]])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
#%%
corners=np.array([[120,0,0],[120,0,240],[120,480,0],[120,480,240]])
kk=GetPlaneReconstructionFast(240,0, 1, 2,  corners, res, prob, prob.Cv)
plt.imshow(kk[0], origin="lower", extent=[0,L[1],0,L[2]])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar()
plt.show()

#%%  - Validation with 2D code

from hybrid2d.Reconstruction_functions import coarse_cell_center_rec
from hybrid2d.Small_functions import get_MRE, plot_sketch
from hybrid2d.Testing import (
    FEM_to_Cartesian,
    Testing,
    extract_COMSOL_data,
    save_csv,
    save_var_name_csv,
)

from hybrid2d.Module_Coupling import assemble_SS_2D_FD
from hybrid2d.Module_Coupling_sparse import non_linear_metab_sparse
from hybrid2d.reconst_and_test_module import reconstruction_sans_flux
# 0-Set up the sources
# 1-Set up the domain

D = 1
K0 = K
L = 240

cells = 10
h_coarse = L / cells

# Definition of the Cartesian Grid
x_coarse = np.linspace(h_coarse / 2, L - h_coarse / 2, int(np.around(L / h_coarse)))
y_coarse = x_coarse

# V-chapeau definition
directness = 3
print("directness=", directness)

S = 1
Rv = L / alpha + np.zeros(S)
pos_s = np.array([[0.5, 0.5]]) * L

# ratio=int(40/cells)*2
ratio = int(100 * h_coarse // L / 4) * 2

print("h coarse:", h_coarse)
K_eff = K0 / (np.pi * Rv**2)


C_v_array = np.ones(S)

# =============================================================================
# BC_value = np.array([0, 0.2, 0, 0.2])
# BC_type = np.array(["Periodic", "Periodic", "Neumann", "Dirichlet"])
# =============================================================================
BC_value = np.array([0, 0,0,0])
#BC_type = np.array(["Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"])
BC_type = np.array(["Neumann", "Neumann", "Neumann", "Neumann"])
t = Testing(
    pos_s, Rv, cells, L, K_eff, D, directness, ratio, C_v_array, BC_type, BC_value
)
n = non_linear_metab_sparse(
    pos_s,
    Rv,
    h_coarse,
    L,
    K_eff,
    D,
    directness,
)

n.pos_arrays()  # Creates the arrays with the geometrical position of the sources


# Assembly of the Laplacian and other arrays for the linear system
LIN_MAT = n.assembly_linear_problem_sparse(BC_type, BC_value)
n.set_intravascular_terms(
    C_v_array
)  
# Sets up the intravascular concentration as BC
ind_array=n.H0.copy()
ind_array[:cells**2]-=M*n.h**2

sol = sp.sparse.linalg.spsolve(LIN_MAT, -ind_array)
n.sol=sol
s_FV = sol[: -n.S].reshape(len(n.x), len(n.y))
q = sol[-n.S :]

ratio=20
a = reconstruction_sans_flux(sol, n, n.L, ratio, n.directness)
a.reconstruction()
a.reconstruction_boundaries_short(BC_type,BC_value)
a.rec_corners()
plt.imshow(a.rec_final, origin="lower", vmax=np.max(a.rec_final))
plt.title("bilinear reconstruction \n coupling model Metabolism")
plt.colorbar()
# =============================================================================
# c = 0
# plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")
# plt.xlabel("x")
# plt.legend()
# plt.title("linear 2D reference")
# plt.show()
# 
# plt.plot(t.y_fine, t.array_phi_field_y_Multi[c], label="Multi")
# plt.xlabel("y")
# plt.legend()
# plt.title("linear 2D reference")
# plt.show()
# 
# =============================================================================
    


#%%
plt.plot(np.linspace(0,L*(1-1/res), res)+L[0]/2/res,a.data[5,int(res/2),:])
plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")


