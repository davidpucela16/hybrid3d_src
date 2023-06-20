#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:22:23 2023

@author: pdavid
"""
import os 
path=os.path.dirname(__file__)
os.chdir(path)
import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from neighbourhood import GetNeighbourhood, GetUncommon
from small_functions import TrilinearInterpolation, auto_TrilinearInterpolation
from small_functions import FromBoundaryGetNormal, AppendSparse, GetBoundaryStatus
from scipy.sparse.linalg import spsolve as dir_solve
from assembly import AssemblyDiffusion3DBoundaries, AssemblyDiffusion3DInterior
from Second_eq_functions import node, InterpolateFast,GetInterpolationKernelFast,GetI1Fast, InterpolatePhiBarBlock, RetrievePhiBar

import scipy as sp
from scipy.sparse import csc_matrix

import multiprocessing
from multiprocessing import Pool
from assembly_1D import FullAdvectionDiffusion1D

from numba.typed import List

from mesh import GetID, Get8Closest
from mesh_1D import KernelIntegralSurfaceFast, KernelPointFast
from small_functions import in1D
from GreenFast import SimpsonSurface

from numba import njit, prange
from numba.experimental import jitclass
from numba import int64, float64
from numba import int64, types, typed
import numba as nb
import matplotlib.pylab as pylab

import pdb

plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15,15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)

import time


class hybrid_set_up():
    def __init__(self, mesh_3D, mesh_1D,  BC_type, BC_value,n, D, K, BCs_1D):
        
        self.BCs_1D=BCs_1D
        self.K=K
        
        self.mesh_3D=mesh_3D
        self.h=mesh_3D.h
        
        self.mesh_1D=mesh_1D
        self.R=mesh_1D.R
        self.D=D
        self.BC_type=BC_type
        self.BC_value=BC_value
        
        self.n=n
        
        self.mesh_3D.GetOrderedConnectivityMatrix()
        
        self.F=self.mesh_3D.size_mesh
        self.S=len(mesh_1D.pos_s)
        
        self.DetermineMaxSize()
        
        self.B_assembly_bool=False #True if the matrix has already been computed
        self.phi_bar_bool=False #True if the matrix has already been computed
        self.I_assembly_bool=False
        
        self.intra_exit_BC="zero_flux"
        
        return
    
    def DetermineMaxSize(self):
        """When assembling the relevant arrays for the sparse matrices, normally I use np.concatenate to 
        sequentially append the new values of the kernels. However, this is a very inneficient method, it 
        is better to declare a size of the arrays, and then fill them up sequentially. For that task, it is 
        usefull to know the maximum amount of sources influencing a given FV mesh"""
        s_blocks_unique=self.mesh_1D.uni_s_blocks #each FV cell that contains a source
        s_blocks_count=self.mesh_1D.counts #how many sources each block has
        
        max_size=np.sum(np.sort(s_blocks_count)[::-1][:(self.n*2+1)**3]) #Adds up all the sources in the n**2 most populated blocks
        #At no point will an array comming fror a KernelPoint or kernel_surface contain more than max_size entries
        self.max_size=max_size
        return 
    
    def AssemblyABC(self, mat_path):
        A_matrix=self.AssemblyA()
        self.A_matrix=A_matrix
        if not self.B_assembly_bool:
            B_matrix=self.Assembly_BFast()
            #Notice how the B matrix that we are saving does not have the boundaries included
            sp.sparse.save_npz(mat_path + "/B_matrix",B_matrix)
            
        else:
            print("loading matrix from: ", mat_path)
            B_matrix=sp.sparse.load_npz(os.path.join( mat_path , "B_matrix.npz"))
            print("Finished loading matrix")
        B_matrix=self.AssemblyBBoundaries(B_matrix)
        self.B_matrix=B_matrix
        
        C_matrix=csc_matrix((self.mesh_3D.size_mesh, len(self.mesh_1D.pos_s)))
        self.C_matrix=C_matrix
        
        A_B_C=sp.sparse.hstack((A_matrix, B_matrix, C_matrix))
        self.A_B_C_matrix=A_B_C
        return(A_B_C)
    
    def AssemblyA(self):
        """An h is missing somewhere to be consistent"""
        size=self.mesh_3D.size_mesh
        
        a=AssemblyDiffusion3DInterior(self.mesh_3D)
        b=AssemblyDiffusion3DBoundaries(self.mesh_3D, self.BC_type, self.BC_value)
        
        # =============================================================================
        #         NOTICE HERE, THIS IS WHERE WE MULTIPLY BY h so to make it dimensionally consistent relative to the FV integration
        A_matrix=csc_matrix((a[2]*self.mesh_3D.h, (a[0], a[1])), shape=(size,size)) + csc_matrix((b[2]*self.mesh_3D.h, (b[0], b[1])), shape=(size, size))
        #       We only multiply the non boundary part of the matrix by h because in the boundaries assembly we need to include the h due to the difference
        #       between the Neumann and Dirichlet boundary conditions. In short AssemblyDiffusion3DInterior returns data that needs to be multiplied by h 
        #       while AssemblyDiffusion3DBoundaries is already dimensionally consistent
        # =============================================================================
        self.I_ind_array=b[3]*self.mesh_3D.h
        
        
        return A_matrix
    
    def Assembly_BFast(self):
        
        mesh=self.mesh_3D
        nmb_ordered_connect_matrix = List(mesh.ordered_connect_matrix)
        net=self.mesh_1D
        B_data, B_row, B_col=AssemblyBArraysFast(nmb_ordered_connect_matrix, mesh.size_mesh, self.n, self.D,
                                    mesh.cells_x, mesh.cells_y, mesh.cells_z, mesh.pos_cells, mesh.h, 
                                    net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge)
        #HERE IS WHERE WE MULTIPLY BY H!!!!!
        self.B_matrix=csc_matrix((B_data*mesh.h, (B_row, B_col)), (mesh.size_mesh, len(self.mesh_1D.s_blocks)))
        return self.B_matrix
    
    def AssemblyBBoundaries(self, B):

        B=B.tolil()
        
        normals=np.array([[0,0,1],  #for north boundary
                          [0,0,-1], #for south boundary
                          [0,1,0],  #for east boundary
                          [0,-1,0], #for west boundary
                          [1,0,0],  #for top boundary 
                          [-1,0,0]])#for bottom boundary
        mesh=self.mesh_3D
        net=self.mesh_1D
        h=mesh.h
        c=0
        for bound in self.mesh_3D.full_full_boundary:
        #This loop goes through each of the boundary cells, and it goes repeatedly 
        #through the edges and corners accordingly
            
            for k in bound: #Make sure this is the correct boundary variable
                k_neigh=GetNeighbourhood(self.n, mesh.cells_x, mesh.cells_y, mesh.cells_z, k)
                
                pos_k=mesh.GetCoords(k)
                normal=normals[c]
                pos_boundary=pos_k+normal*h/2
                if self.BC_type[c]=="Dirichlet":
                    r_k=KernelIntegralSurfaceFast(net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge,
                                                          pos_boundary, normal,  k_neigh, 'P', self.D, self.mesh_3D.h)
                    
                if self.BC_type[c]=="Neumann":
                    r_k=KernelIntegralSurfaceFast(net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge,
                                                          pos_boundary, normal,  k_neigh, 'G', self.D, self.mesh_3D.h)
                
                kernel=csc_matrix((r_k[0]*h**2,(np.zeros(len(r_k[0])),r_k[1])), shape=(1,len(self.mesh_1D.s_blocks)))
                B[k,:]-=kernel
            c+=1
            
        return(B)
    
# =============================================================================
#     def AssemblyDEF(self):
#         """Deprecated, use the fast version instead"""
#         print()
#         print("Deprecated, use the fast version instead")
#         print()
#         D=np.zeros([3,0])
#         E=np.zeros([3,0])
#         F=np.zeros([3,0])
#         
#         #The following are important matrices that are interesting to keep separated
#         #Afterwards they can be used to assemble the D, E, F matrices
#         G_ij=np.zeros([3,0])
#         #H_ij=np.zeros([3,0])
#         Permeability=np.zeros([3,0])
#         
#         for j in range(len(self.mesh_1D.s_blocks)):
#             print("Assembling D_E_F slow, source: ", j)
#             kernel_s,col_s,kernel_q, col_q=self.Interpolate(self.mesh_1D.pos_s[j])
#             D=AppendSparse(D, kernel_s,np.zeros(len(col_s))+j, col_s)
#             E=AppendSparse(E, kernel_q,np.zeros(len(col_q))+j, col_q)
#             
#             G_ij=AppendSparse(G_ij, kernel_q,np.zeros(len(col_q))+j, col_q)
#             
#             
#             E=AppendSparse(E, 1/self.K[self.mesh_1D.source_edge[j]] , j, j)
#             Permeability=AppendSparse(Permeability, 1/self.K[self.mesh_1D.source_edge[j]] , j, j)
#         F=AppendSparse(F, -np.ones(len(self.mesh_1D.s_blocks)) , np.arange(len(self.mesh_1D.s_blocks)), np.arange(len(self.mesh_1D.s_blocks)))
#             
#         self.D_matrix_slow=csc_matrix((D[0], (D[1], D[2])), shape=(len(self.mesh_1D.pos_s), self.mesh_3D.size_mesh))
#         self.E_matrix_slow=csc_matrix((E[0], (E[1], E[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
#         self.F_matrix_slow=csc_matrix((F[0], (F[1], F[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
#         
#         self.G_ij_slow=csc_matrix((G_ij[0], (G_ij[1], G_ij[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
#         #self.H_ij=csc_matrix((H_ij[0], (H_ij[1], H_ij[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
#         self.Permeability=csc_matrix((Permeability[0], (Permeability[1], Permeability[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
#         
#         return sp.sparse.hstack((self.D_matrix_slow, self.E_matrix_slow, self.F_matrix_slow))
# =============================================================================
    
    def AssemblyDEFFast(self, path_phi_bar, mat_path):
        if self.phi_bar_bool:
            print("Began laoding phi_bar from ", mat_path)
            self.phi_bar_s=sp.sparse.load_npz(mat_path + '/phi_bar_s.npz')
            self.Gij=sp.sparse.load_npz(mat_path + '/phi_bar_q.npz')
            print("Finished laoding phi_bar from ", mat_path)
        else:
            if not os.path.exists(path_phi_bar):
                os.mkdir(path_phi_bar)
            print("Calculating phi_bar")
            self.InterpolatePhiFullFast(path_phi_bar, 1)
            self.phi_bar_s, self.Gij=RetrievePhiBar(mat_path, path_phi_bar,self.S, self.mesh_3D.size_mesh, self.mesh_1D.uni_s_blocks)
        self.D_matrix=self.phi_bar_s
        
        q_portion_diagonal=np.repeat(1/self.K, self.mesh_1D.cells) #q_j/K_j
        
        self.q_portion=sp.sparse.diags(q_portion_diagonal)
        #self.Gij+=self.q_portion
        self.F_matrix=-sp.sparse.diags(np.ones(len(self.mesh_1D.s_blocks)))
        self.D_E_F_matrix=sp.sparse.hstack((self.Gij+self.q_portion, self.F_matrix))
        #self.Gij=None
        self.D_E_F_matrix=sp.sparse.hstack((self.D_matrix, self.D_E_F_matrix))
        
        return self.D_E_F_matrix
    
    def GetEMatrix(self):
        return self.q_portion+self.Gij
        
# =============================================================================
#     def InterpolatePhiFullFast(self, path, num_processes):
#         
#         args=path,self.n, self.mesh_3D.cells_x, self.mesh_3D.cells_y, self.mesh_3D.cells_z, self.mesh_3D.h,self.mesh_3D.pos_cells,self.mesh_1D.s_blocks, self.mesh_1D.source_edge,self.mesh_1D.tau, self.mesh_1D.pos_s, self.mesh_1D.h, self.mesh_1D.R, self.D,self.mesh_1D.sources_per_block, self.mesh_1D.quant_sources_per_block
#         a=[]
#         for i in self.mesh_1D.uni_s_blocks:
#             a.append(PhiBarHelper((i, args)))
# 
#         dask.compute(*a)
#         return 
# =============================================================================
    def InterpolatePhiFullFast(self, path, num_processes):
        
        args=path,self.n, self.mesh_3D.cells_x, self.mesh_3D.cells_y, self.mesh_3D.cells_z, self.mesh_3D.h,self.mesh_3D.pos_cells,self.mesh_1D.s_blocks, self.mesh_1D.source_edge,self.mesh_1D.tau, self.mesh_1D.pos_s, self.mesh_1D.h, self.mesh_1D.R, self.D,self.mesh_1D.sources_per_block, self.mesh_1D.quant_sources_per_block
        for i in self.mesh_1D.uni_s_blocks:
            PhiBarHelper((i, args))

        return 


# =============================================================================
#     def RetrievePhiBar(self, path):
#         kernel_q=np.zeros(0, dtype=np.float64)
#         kernel_s=np.zeros(0, dtype=np.float64)
#         #The kernels with the positions
#         col_s=np.zeros(0, dtype=np.int64)
#         col_q=np.zeros(0, dtype=np.int64)
#         
#         row_s=np.zeros(0, dtype=np.int64)
#         row_q=np.zeros(0, dtype=np.int64)
#         
#         for i in self.mesh_1D.uni_s_blocks:
#             kernel_s=np.concatenate((kernel_s, np.load(path + '/{}_kernel_s.npy'.format(i))))
#             kernel_q=np.concatenate((kernel_q, np.load(path + '/{}_kernel_s.npy'.format(i))))
#             col_s=np.concatenate((col_s, np.load(path + '/{}_kernel_s.npy'.format(i))))
#             col_q=np.concatenate((col_q, np.load(path + '/{}_kernel_s.npy'.format(i))))
#             row_s=np.concatenate((row_s, np.load(path + '/{}_kernel_s.npy'.format(i))))
#             row_q=np.concatenate((row_q, np.load(path + '/{}_kernel_s.npy'.format(i))))
#             
#             
#         return kernel_s,row_s, col_s,kernel_q, row_q, col_q
# =============================================================================

    def AssemblyGHI(self, path_I):
        I_matrix=self.AssemblyI(path_I)
        
        #WILL CHANGE WHEN CONSIDERING MORE THAN 1 VESSEL
        aux_arr=np.zeros(len(self.mesh_1D.pos_s))
        #H matrix for multiple vessels
        for ed in range(len(self.R)): #Loop through every vessel
            DoFs=np.arange(np.sum(self.mesh_1D.cells[:ed]),np.sum(self.mesh_1D.cells[:ed])+np.sum(self.mesh_1D.cells[ed])) #DoFs belonging to this vessel
            aux_arr[DoFs]=self.mesh_1D.h[ed]/(np.pi*self.mesh_1D.R[ed]**2)
            self.aux_arr=aux_arr #I should check if I can do the previous operation with np.repeat
        
        H_matrix=sp.sparse.diags(aux_arr, 0)
        self.H_matrix=H_matrix
        
        #Matrix full of zeros:
        G_matrix=csc_matrix(( len(self.mesh_1D.pos_s),self.mesh_3D.size_mesh))
        self.G_matrix=G_matrix
        self.G_H_I_matrix=sp.sparse.hstack((G_matrix, H_matrix, I_matrix))
        
        return(self.G_H_I_matrix)
    
    def AssemblyI(self, path_I):
        """Models intravascular transport. Advection-diffusion equation
        
        FOR NOW IT ONLY HANDLES A SINGLE VESSEL"""
        
        D=self.mesh_1D.D
        U=self.mesh_1D.U
        
        L=self.mesh_1D.L
        
        
        if not self.I_assembly_bool:
            aa, ind_array, DoF=FullAdvectionDiffusion1D(U, D, self.mesh_1D.h, self.mesh_1D.cells, self.mesh_1D.startVertex, self.mesh_1D.vertex_to_edge, self.R, self.BCs_1D, self.intra_exit_BC)
            self.III_ind_array=ind_array
            I = csc_matrix((aa[0], (aa[1], aa[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
            sp.sparse.save_npz(path_I + "/I_matrix", I)
            np.save(path_I + "/III_ind_array", ind_array)
        else:
            print("Began laoding I and III_ind_array from ", path_I)
            I=sp.sparse.load_npz(path_I + "/I_matrix.npz")
            self.III_ind_array=np.load(path_I + "/III_ind_array.npy")
            print("Finished laoding I and III_ind_array from ", path_I)
        self.I_matrix=I
        
        return I
    
    def AssemblyProblem(self, path_matrices):
        Full_linear_matrix=sp.sparse.vstack((self.AssemblyABC(path_matrices),
                                             self.AssemblyDEFFast(path_matrices + "/E_portion", path_matrices),
                                             self.AssemblyGHI(path_matrices)))
        self.Full_linear_matrix=Full_linear_matrix
        
        self.Full_ind_array=np.concatenate((self.I_ind_array, np.zeros(len(self.mesh_1D.pos_s)), self.III_ind_array))
        return
    
    def GetIndependentArray(self):
        self.Full_ind_array=np.concatenate((self.I_ind_array, np.zeros(len(self.mesh_1D.pos_s)), self.III_ind_array))
        return
    
    def ReAssemblyMatrices(self):
        Upper=sp.sparse.hstack((self.A_matrix, self.B_matrix, self.C_matrix))
        Middle=sp.sparse.hstack((self.D_matrix, self.E_matrix, self.F_matrix))
        Down=sp.sparse.hstack((self.G_matrix, self.H_matrix, self.I_matrix))
        
        self.Middle=Middle
        
        Full_linear_matrix=sp.sparse.vstack((Upper,
                                             Middle,
                                             Down))
        
        return Full_linear_matrix
    

    def SolveProblem(self):
        sol=dir_solve(self.Full_linear_matrix, -self.Full_ind_array)
        self.s=sol[:self.mesh_3D.size_mesh]
        self.q=sol[self.mesh_3D.size_mesh:-self.S]
        self.Cv=sol[-self.S:]
        return
        
    def GetBoundaryStatus(self, coords):
        """Take good care of never touching the boundary!!"""
        bound_status=GetBoundaryStatus(coords, self.mesh_3D.h, self.mesh_3D.cells_x,self.mesh_3D.cells_y, self.mesh_3D.cells_z )
        return bound_status
    
    def GetPointValuePost(self, coords, rec,k):
        a,b,c,d,e,f=self.Interpolate(coords)
        rec[k]=a.dot(self.s[b])+c.dot(self.q[d])
        return 
    
    def Interpolate(self, x):
        """Function just to call InterpolateFast without having to list all the arguments"""
        return InterpolateFast(x, self.n, self.mesh_3D.cells_x, self.mesh_3D.cells_y,
                              self.mesh_3D.cells_z, self.mesh_3D.h, self.GetBoundaryStatus(x), 
                              self.mesh_3D.pos_cells,self.mesh_1D.s_blocks, self.mesh_1D.source_edge,
                              self.mesh_1D.tau, self.mesh_1D.pos_s, self.mesh_1D.h, self.mesh_1D.R, self.D)


    


@njit
def GetInterfaceKernelsFast(k,m,pos_k, pos_m,h_3D, n, cells_x, cells_y, cells_z,
                                    s_blocks, tau, h_1D, pos_s, source_edge,D):
    k_neigh=GetNeighbourhood(n, cells_x, cells_y, cells_z, k)
    m_neigh=GetNeighbourhood(n, cells_x, cells_y, cells_z, m)
    
    normal=(pos_m-pos_k)/h_3D
    
# =============================================================================
#     #if np.linalg.norm(normal) > 1: print('ERROR!!!!!!!!!!!!!!!!!')
#     if np.linalg.norm(normal) > 1.0000001: pdb.set_trace()
# =============================================================================
    
    
    sources_k_m=np.arange(len(s_blocks), dtype=np.int64)[in1D(s_blocks, GetUncommon(k_neigh, m_neigh))]
    sources_m_k=np.arange(len(s_blocks), dtype=np.int64)[in1D(s_blocks, GetUncommon(m_neigh, k_neigh))]
    #sources=in1D(s_blocks, neighbourhood)
    r_k_m=np.zeros(len(sources_k_m), dtype=np.float64)
    r_m_k=np.zeros(len(sources_m_k), dtype=np.float64)
    grad_r_k_m=np.zeros(len(sources_k_m), dtype=np.float64)
    grad_r_m_k=np.zeros(len(sources_m_k), dtype=np.float64)
    center=pos_m/2+pos_k/2
    
    c=0
    for i in sources_k_m:
        ed=source_edge[i]
        a,b= pos_s[i]-tau[ed]*h_1D[ed]/2, pos_s[i]+tau[ed]*h_1D[ed]/2
        r_k_m[c]=SimpsonSurface(a,b,'P', center,h_3D, normal,D)
        grad_r_k_m[c]=SimpsonSurface(a,b,'G', center,h_3D, normal,D)
        c+=1
        
    c=0
    for i in sources_m_k:
        ed=source_edge[i]
        a,b= pos_s[i]-tau[ed]*h_1D[ed]/2, pos_s[i]+tau[ed]*h_1D[ed]/2
        r_m_k[c]=SimpsonSurface(a,b,'P', center,h_3D, normal,D)
        grad_r_m_k[c]=SimpsonSurface(a,b,'G', center,h_3D, normal,D)
        c+=1
    

    return(sources_k_m, r_k_m, grad_r_k_m, sources_m_k, r_m_k, grad_r_m_k)


@njit
def AssemblyBArraysFast(nmb_ordered_connect_matrix, size_mesh, n,D,
                                cells_x, cells_y, cells_z, pos_cells, h_3D, 
                                s_blocks, tau, h_1D, pos_s, source_edge):
    
    B_data=np.zeros(0, dtype=np.float64)
    B_row=np.zeros(0,dtype=np.int64)
    B_col=np.zeros(0,dtype=np.int64)
    for k in range(size_mesh):
        if not k%20: print("Assembling B, FV cell: ", k)
        N_k=nmb_ordered_connect_matrix[k] #Set with all the neighbours
        
        for m in N_k:
            sources_k_m, r_k_m, grad_r_k_m, sources_m_k, r_m_k, grad_r_m_k=GetInterfaceKernelsFast(k,m,pos_cells[k], pos_cells[m],h_3D, n, cells_x, cells_y, cells_z,
                                                s_blocks, tau, h_1D, pos_s, source_edge,D )
            B_data=np.concatenate((B_data, -r_k_m-grad_r_k_m/2*h_3D, r_m_k+grad_r_m_k/2*h_3D))
            B_row=np.concatenate((B_row, k*np.ones(len(sources_k_m)+len(sources_m_k),dtype=np.int64)))
            B_col=np.concatenate((B_col, sources_k_m, sources_m_k))

    return B_data, B_row, B_col

                      
       

#@dask.delayed
def PhiBarHelper(args):
    block, lst=args
    path,n, cells_x, cells_y, cells_z, h_3D,pos_cells,s_blocks, source_edge,tau, pos_s, h_1D, R, D,sources_per_block, quant_sources_per_block=lst
    print("block", block)
    kernel_s,row_s, col_s,kernel_q, row_q, col_q=InterpolatePhiBarBlock(block,n, cells_x, cells_y, cells_z, h_3D, 
                                 pos_cells,s_blocks, source_edge,tau, pos_s, h_1D, R, D, 
                                 sources_per_block, quant_sources_per_block)
    
    np.save(path + '/{}_kernel_s'.format(block), kernel_s)
    np.save(path + '/{}_row_s'.format(block), row_s)
    np.save(path + '/{}_col_s'.format(block), col_s)
    
    np.save(path + '/{}_kernel_q'.format(block), kernel_q)
    np.save(path + '/{}_row_q'.format(block), row_q)
    np.save(path + '/{}_col_q'.format(block), col_q)
    
    return 
    #return kernel_s,row_s, col_s,kernel_q, row_q, col_q

    
