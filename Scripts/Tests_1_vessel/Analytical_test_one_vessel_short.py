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

cells_3D=31
n=40
path_matrices = os.path.join(path_src, '../../linear_system/' + name_script)
path_matrices=os.path.join(path_matrices,"F{}_n{}".format(cells_3D, n))

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
#B_assembly_bool=True
#phi_bar_bool=True
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
BC_type=np.array([ "Dirichlet","Dirichlet", "Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])
#BC_value=np.array([0.1,0.1,0.1,0.1,0,0])



# - This is the validation of the 1D transport eq without reaction
U = np.array([0.1])
D = 1
K=1
L_vessel = cells_3D*2
L=np.array([1,1,1])*L_vessel
mesh=cart_mesh_3D(L,cells_3D)


mesh.AssemblyBoundaryVectors()
alpha=50
R=L[0]/alpha
cells_1D = np.array([1])
#cells_1D = np.array([30])

startVertex=np.array([0])
endVertex=np.array([1])

position_within=0.5
pos_vertex=np.array([[L_vessel/2-R,L_vessel/2,  L_vessel*position_within],[L_vessel/2+R, L_vessel/2, L_vessel*position_within]])
#pos_vertex=np.array([[R,L_vessel/2,  L_vessel*position_within],[L_vessel-R, L_vessel/2, L_vessel*position_within]])

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
prob.phi_bar_bool=phi_bar_bool
prob.B_assembly_bool=B_assembly_bool
prob.I_assembly_bool=I_assembly_bool
prob.AssemblyProblem(path_matrices)

print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/net.L)

C_v_array=np.ones(len(net.pos_s))

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
from post_processing import ReconstructionCoordinatesFast, GetProbArgs
res=50
x=np.linspace(1,L_vessel-1, res)
crds=np.vstack((L_vessel/2+ np.zeros(res), x,L_vessel/2*np.ones(res))).T
phi_line=ReconstructionCoordinatesFast(crds, GetProbArgs(prob), prob.s, prob.q)

crds=np.vstack((x, x,x)).T
phi_line_far=ReconstructionCoordinatesFast(crds, GetProbArgs(prob), prob.s, prob.q)
plt.plot(x, phi_line)
plt.plot(x, phi_line_far)
plt.show()

#%% - Analytical 
q=1/(1+(alpha-(4/3*np.pi)**(1/3))/2/np.pi/alpha)
R_sph=L_vessel/(4/3*np.pi)**(1/3)

phi_an=q*2*R/4/np.pi*(1/np.abs(x-L_vessel/2)-1/R_sph)
phi_bar_an=q*2*R/4/np.pi*(1/R-1/R_sph)

phi_an[np.abs(x-L_vessel/2)<R]=phi_bar_an

plt.plot(x, phi_line)
plt.plot(x, phi_line_far)
plt.plot(x, phi_an, label="an")
plt.legend()
plt.show()
#%%
from PrePostTemp import GetCoarsePhi
phi_coarse=GetCoarsePhi( prob.q, prob.Cv, prob.s, GetProbArgs(prob))

slic=mesh.GetYSlice(L_vessel/2).reshape(cells_3D, cells_3D)
phi_coarse_middle=phi_coarse[0][slic[int(cells_3D/2)]]

plt.plot(mesh.y, phi_coarse_middle)

#%%

import numpy as np
import math
from scipy import integrate

def DefineIntegralLimits(pos_source, pos_boundary_cell, normal, L, h):
    i, j = np.where(normal == 0)[0]
    pos_boundary = pos_boundary_cell + h * normal / 2
    L_theta = pos_boundary[i] - pos_source[i]
    L_gamma = pos_boundary[j] - pos_source[j]
    
    theta_lim = (
        np.arctan((L_theta + h / 2) / (L / 2)),
        np.arctan((L_theta - h / 2) / (L / 2))
    )
    gamma_lim = (
        np.arctan((L_gamma + h / 2) / (L / 2)),
        np.arctan((L_gamma - h / 2) / (L / 2))
    )
    pdb.set_trace()
    return theta_lim, gamma_lim


def Distance(theta, gamma):
    return np.sqrt(1 + np.arctan(theta) ** 2 + np.arctan(gamma) ** 2)



# Define the function to integrate
def integrand(theta, gamma):
    return 2 / Distance(theta, gamma) / L

# Perform the double integration
a,b=DefineIntegralLimits(np.array([0,0,0]), mesh.pos_cells[prob.mesh_3D.full_north[0]], np.array([0,0,1]), L_vessel, mesh.h)
result, error = integrate.dblquad(integrand, a[0], a[1], b[0], b[1])

print("Result:", result)
print("Error:", error)

#%%
from neighbourhood import GetNeighbourhood
from mesh_1D import KernelPointFast, KernelIntegralSurfaceFast

cell=mesh.full_north[35]
normal=np.array([0,0,1])
array=np.zeros(len(net.pos_s))
# =============================================================================
# for i in range(len(net.pos_s)):
#     pos_source=net.pos_s[i]
#     a,b=DefineIntegralLimits(net.pos_s[i], mesh.pos_cells[cell],normal , L_vessel, mesh.h)
#     result, error = integrate.dblquad(integrand, a[0], a[1], b[0], b[1])
#     
#     
#     
#     array[i]=result*2*mesh.h
#     print(error)
# =============================================================================
i, j = np.where(normal == 0)[0]
pos_boundary = mesh.pos_cells[cell] + mesh.h * normal / 2   
kernel_point=KernelPointFast(pos_boundary, GetNeighbourhood(n, mesh.cells_x, mesh.cells_y, mesh.cells_z, cell), 
                net.s_blocks, net.source_edge, net.tau, net.pos_s, net.h, net.R,1)

kernel_Simpson=KernelIntegralSurfaceFast(net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge,
                                      pos_boundary, normal,  GetNeighbourhood(n, mesh.cells_x, mesh.cells_y, mesh.cells_z, cell),
                                      'P', 1, mesh.h)

print(kernel_point[0]*2*mesh.h)
print(kernel_Simpson[0]*2*mesh.h)
print(prob.B_matrix.toarray()[cell])


#%%- Get Boundary Value BV
grad_s=-prob.A_matrix.dot(prob.s)
for k in mesh.full_bound:
    normals=np.array([[0,0,1],  #for north boundary
                      [0,0,-1], #for south boundary
                      [0,1,0],  #for east boundary
                      [0,-1,0], #for west boundary
                      [1,0,0],  #for top boundary 
                      [-1,0,0]])#for bottom boundary

    bounds=np.where(mesh.full_full_boundary==k)[0]
    for b in bounds:
        normal=normals[b]
        pos_boundary=mesh.pos_cells[k]+normal*mesh.h/2
        print(pos_boundary)
        
        KPF=KernelPointFast(pos_boundary, net.s_blocks, net.s_blocks, net.source_edge, net.tau, net.pos_s, net.h, net.R,D)
        KIS=KernelIntegralSurfaceFast(net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge,
                                              pos_boundary, normal,  net.s_blocks,'P', 1, mesh.h)
        BV_point=prob.s[k]+grad_s[k]/mesh.h/2+KPF[0].dot(prob.q)
        BV_surf=prob.s[k]+grad_s[k]/mesh.h/2+KIS[0].dot(prob.q)
        print("\nFor cell {}, point_value={}".format(k, BV_point))
        print("For cell {}, point_value={}".format(k, BV_surf))
        
        KPF_2=KernelPointFast(mesh.pos_cells[k], net.s_blocks, net.s_blocks, net.source_edge, net.tau, net.pos_s, net.h, net.R,D)
        cell_point=prob.s[k]+KPF_2[0].dot(prob.q)
        print("For cell {}, point_value={}".format(k, cell_point))
        #Mass balance
        print("mass balance", -grad_s[k]+2*mesh.h*(-KIS[0].dot(prob.q)-prob.s[k]))
        
#%%
PP=AssemblyDiffusion3DInterior(mesh)
PP=AssemblyDiffusion3DBoundaries(mesh_object, BC_type, BC_value)

#%%
from PrePostTemp import VisualizationTool

#corners=np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]])*(1-1/cells_3D)+mesh.h/2
corners=np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]])
aax=VisualizationTool(prob, 0,1,2, corners, res)
aax.GetPlaneData(path_output)
aax.PlotData(path_output)


aay=VisualizationTool(prob, 1,0,2,corners, res)
aay.GetPlaneData(path_output)
aay.PlotData(path_output)

aaz=VisualizationTool(prob, 2,0,1, corners, res)
aaz.GetPlaneData(path_output)
aaz.PlotData(path_output)

#aax2=VisualizationTool(prob, 0,2,1, np.array([[0,0],[0,L_vessel],[L_vessel,0],[L_vessel,L_vessel]]), res)
#aax2.GetPlaneData(path_output)
#aax2.PlotData(path_output)



