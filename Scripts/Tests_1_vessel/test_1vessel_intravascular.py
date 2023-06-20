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



from assembly_1D import FullAdvectionDiffusion1D





#%% - This is the validation of the 1D transport eq without reaction
U = np.array([0.1])*2
D = 1
K=0.2
L = np.array([10])
R=np.array([L[0]/50])
cells_1D = 10

s = np.linspace(L[0]/2/cells_1D, L[0]-L[0]/2/cells_1D, cells_1D)

Pe = U*L/D

analytical = (np.exp(Pe*s/L)-np.exp(Pe))/(1-np.exp(Pe))

plt.plot(s, analytical)

startVertex=np.array([0])
endVertex=np.array([1])
vertex_to_edge=np.array([[0],[0]])
BCs=np.array([[0,1],
              [1,0]])

#aa, ind_array, DoF=FullAdvectionDiffusion1D(U, D, L/cells_1D, np.array([cells_1D]), startVertex, vertex_to_edge, R, BCs, "zero_flux")
aa, ind_array, DoF=FullAdvectionDiffusion1D(U, D, L/cells_1D, np.array([cells_1D]), startVertex, vertex_to_edge, R, BCs)

A = sp.sparse.csc_matrix((aa[0], (aa[1], aa[2])), shape=(cells_1D, cells_1D))


sol = sp.sparse.linalg.spsolve(A, -ind_array)

#sol = np.hstack((np.array((1)), sol, np.array((0))))



k=K/np.pi/R[0]**2
A[np.arange(cells_1D), np.arange(cells_1D)]+=k*L/cells_1D #out flux
sol_reac = sp.sparse.linalg.spsolve(A, -ind_array)

#sol = np.hstack((np.array((1)), sol, np.array((0))))

n=np.exp(np.sqrt(U**2+4*k*D)*L/2/D)
A=1/(1-n**2)
B=1/(1-1/n**2)

analytical_reac=np.exp(U[0]*s/2/D)*(A*np.exp(np.sqrt(U[0]**2+4*k*D)*s/2/D)+B*np.exp(-np.sqrt(U[0]**2+4*k*D)*s/2/D))


# Create the figure and axes objects
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

# Plot the first subplot
axes[0].plot(s,sol, label='numerical')
axes[0].plot(s, analytical, label='analytical')
box_string='Pe={}'.format(U*L/D)
axes[0].text(2,0.50, box_string, fontsize = 40, 
 bbox = dict(facecolor = 'white', alpha = 0.5))
axes[0].set_title('Advection - diffusion')

# Plot the second subplot
# Plot the first subplot
axes[1].plot(s,sol_reac, label='numerical')
axes[1].plot(s, analytical_reac, label='analytical')
axes[1].legend()
box_string='Da={:.2f}'.format(k*L[0]/D)
axes[1].text(2,0.50, box_string, fontsize = 40, 
 bbox = dict(facecolor = 'white', alpha = 0.5))
axes[1].set_title('Advection - diffusion - reaction')

# Set the overall title for the figure
fig.suptitle('Reference solution for weak couplings \n\n')

# Show the plot
plt.show()


#%% - 

from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
import matplotlib.pyplot as plt

import math

from mesh_1D import mesh_1D
from GreenFast import GetSourcePotential
import pdb

from hybridFast import hybrid_set_up

from neighbourhood import GetNeighbourhood, GetUncommon
#%



#BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet", "Neumann","Neumann","Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

cells=9
n=10
L_3D=np.array([10,10,10])
mesh=cart_mesh_3D(L_3D,cells)

mesh.AssemblyBoundaryVectors()

#%%

startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L_3D[0]/2, 0.01, L_3D[0]/2],[L_3D[0]/2, L_3D[0]-0.01,L_3D[0]/2]])
vertex_to_edge=[[0],[0]]
diameters=2*R
h=np.array([L_3D[0]])/cells_1D

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.PositionalArraysFast(mesh)

prob=hybrid_set_up(mesh, net, BC_type, BC_value, n, 1, np.zeros(len(diameters))+K, BCs)
mesh.GetOrderedConnectivityMatrix()
prob.B_assembly_bool=os.path.exists(path_matrices + '/B_matrix.npz')

prob.intra_exit_BC=None
prob.AssemblyProblem(path_matrices)


#%% - Validation of the H, I, F matrices:
# First: Advection diffusion, no reaction
#Also, try to make sure all the h coefficients that have to be added are properly done and written in the notebook!
#In the notebook write what each assembly function does and where do you multiply by h and why!
I=prob.I_matrix
sol = sp.sparse.linalg.spsolve(I, -prob.III_ind_array)

plt.scatter(net.pos_s[:,1],sol, label='hybrid')
plt.plot(s, analytical, label='analytical', linewidth=4)
plt.legend()
plt.show()

#%% Advection - diffusion - reaction, weak couplings

new_E=sp.sparse.identity(len(net.pos_s))/(K)
H=prob.H_matrix

F=prob.F_matrix
ind=np.concatenate((np.zeros(len(net.pos_s)), prob.III_ind_array))

L1=sp.sparse.hstack((new_E,F))
L2=sp.sparse.hstack((H,I))

Li=sp.sparse.vstack((L1,L2))

sol=dir_solve(Li, -ind)


plt.plot(net.pos_s[:,1],sol[cells_1D:], label='hybrid reaction')
plt.plot(s, analytical_reac, label='analytical')
plt.legend()
plt.show()

#End of the validation of the 1D transport model, that is G, H, I matrices

#%% - Test with no diffusive flux at the end:
prob.intra_exit_BC="zero_flux"
prob.AssemblyI(path_matrices)
I=prob.I_matrix
sol = sp.sparse.linalg.spsolve(I, -prob.III_ind_array)

plt.scatter(net.pos_s[:,1],sol, label='hybrid')
plt.plot(s, analytical, label='analytical', linewidth=4)
plt.legend()
plt.show()

#%% Advection - diffusion - reaction, weak couplings

new_E=sp.sparse.identity(len(net.pos_s))/(K)
H=prob.H_matrix

F=prob.F_matrix
ind=np.concatenate((np.zeros(len(net.pos_s)), prob.III_ind_array))

L1=sp.sparse.hstack((new_E,F))
L2=sp.sparse.hstack((H,I))

Li=sp.sparse.vstack((L1,L2))

sol=dir_solve(Li, -ind)


plt.plot(net.pos_s[:,1],sol[cells_1D:], label='hybrid reaction')
plt.plot(s, analytical_reac, label='analytical')
plt.legend()
plt.show()
