#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:47:23 2023

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

cells_3D=10
n=11
gradient="x"
path_matrices = os.path.join(path_src, '../../linear_system/' + name_script)
path_matrices=os.path.join(path_matrices,"F{}_n{}_{}".format(cells_3D, n, gradient))

path_output = os.path.join(path_src, '../../output_figures/' + name_script)
path_phi_bar = os.path.join(path_matrices, 'path_phi_bar')
#path_thesis = os.path.join(path_src, '../../path_thesis/' + name_script)

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
from assembly_1D import FullAdvectionDiffusion1D

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



#True if no need to compute
phi_bar_bool=os.path.exists(os.path.join(path_matrices, 'phi_bar_q.npz')) and os.path.exists(os.path.join(path_matrices, 'phi_bar_s.npz')) 
B_assembly_bool=os.path.exists(os.path.join(path_matrices, 'B_matrix.npz'))
I_assembly_bool=False
#True if need to compute
Computation_bool = True
rec_bool=False


# =============================================================================
# from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
# from mesh import cart_mesh_3D
# from mesh_1D import mesh_1D
# from GreenFast import GetSourcePotential
# from hybridFast import hybrid_set_up
# from post_processing import GetPlaneReconstructionFast, ReconstructionCoordinatesFast
# from neighbourhood import GetNeighbourhood, GetUncommon
# from PrePostTemp import VisualizationTool
# =============================================================================


#%%
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_type=np.array([ "Dirichlet","Dirichlet", "Dirichlet","Dirichlet","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])
#BC_value=np.array([0.1,0.1,0.1,0.1,0,0])



# - This is the validation of the 1D transport eq without reaction
U = np.array([0.1])
D = 1
K=1
L_vessel = 240
L=np.array([1,1,1])*L_vessel
mesh=cart_mesh_3D(L,cells_3D)


mesh.AssemblyBoundaryVectors()
alpha=20
R=L[0]/alpha
cells_1D = np.array([200])


startVertex=np.array([0])
endVertex=np.array([1])

position_within=0.5
if gradient=="x":
    pos_vertex=np.array([[0.1,L_vessel/2,  L_vessel*position_within],[L_vessel-0.1, L_vessel/2, L_vessel*position_within]])
elif gradient=="y":
    pos_vertex=np.array([[L_vessel/2, 0.1, L_vessel*position_within],[L_vessel/2, L_vessel-0.1,  L_vessel*position_within]])
elif gradient=="z":
    pos_vertex=np.array([[L_vessel*position_within,L_vessel/2,  0.1],[L_vessel*position_within,L_vessel/2, L_vessel-0.1]])


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
#%%
mesh.GetOrderedConnectivityMatrix()
prob.phi_bar_bool=phi_bar_bool
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool
prob.AssemblyProblem(path_matrices)

print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/net.L)


#%% - We solve the 2D problem in a 3D setting to be able to validate. Ideally, there is no transport in the y direction
C_v_array=np.ones(len(net.pos_s)) #Though this neglects completely the contribution from the dipoles

L1=sp.sparse.hstack((prob.A_matrix,prob.B_matrix))
L2=sp.sparse.hstack((prob.D_matrix, prob.Gij+prob.q_portion))

Li=sp.sparse.vstack((L1,L2))

M=2e-6
M=0
ind=np.concatenate((prob.I_ind_array-M*mesh.h**3, prob.F_matrix.dot(C_v_array)))

sol=dir_solve(Li, -ind)


plt.plot(net.pos_s[:,0],sol[-prob.S:], label='hybrid reaction')
plt.title("q(s) with C_v=1")
plt.legend()
plt.show()

prob.s=sol[:-prob.S]
prob.q=sol[-prob.S:]
prob.Cv=C_v_array

#%%

from PrePostTemp import VisualizationTool
res=50
#corners=np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]])*(1-1/cells_3D)+mesh.h/2
corners=np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]])
aax=VisualizationTool(prob, 0,1,2, corners, res)
aax.GetPlaneData(path_output)
aax.PlotData(path_output)


aay=VisualizationTool(prob, 1,0,2,corners, res)
aay.GetPlaneData(path_output)
aay.vmax=0.5
aay.PlotData(path_output)

aaz=VisualizationTool(prob, 2,0,1, corners, res)
aaz.GetPlaneData(path_output)
aaz.vmax=0.5
aaz.PlotData(path_output)

#aax2=VisualizationTool(prob, 0,2,1, np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]]), res)
#aax2.GetPlaneData(path_output)
#aax2.PlotData(path_output)


#%% - Coupled Problem

ind_coup=prob.Full_ind_array
ind_coup[:prob.F]-=M*mesh.h**3/10
sol_coup=dir_solve(prob.Full_linear_matrix, -ind_coup)


plt.plot(net.pos_s[:,0],sol_coup[-prob.S:], label='hybrid reaction')
plt.title("q(s) with C_v=1")
plt.legend()
plt.show()

prob.s=sol_coup[:-prob.S]
prob.q=sol_coup[-2*prob.S:-prob.S]
prob.Cv=sol_coup[-prob.S:]


from PrePostTemp import VisualizationTool
res=50
corners=np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]])*(1-1/cells_3D)+mesh.h/2
aax=VisualizationTool(prob, 0,1,2, corners, res)
aax.GetPlaneData(path_output)
aay=VisualizationTool(prob, 1,0,2,corners, res)
aay.GetPlaneData(path_output)
aaz=VisualizationTool(prob, 2,0,1, corners, res)
aaz.GetPlaneData(path_output)

#aax2=VisualizationTool(prob, 0,2,1, np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]]), res)
#aax2.GetPlaneData(path_output)

aax.PlotData(path_output)
aay.PlotData(path_output)
aaz.PlotData(path_output)

#aax2.PlotData(path_output)


#%% - Coupled Problem