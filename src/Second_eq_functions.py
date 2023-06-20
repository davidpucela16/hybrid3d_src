#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:31:58 2023

@author: pdavid
"""

import numpy as np 
import pdb 
from neighbourhood import GetNeighbourhood, GetUncommon
from small_functions import TrilinearInterpolation, auto_TrilinearInterpolation
from small_functions import FromBoundaryGetNormal, AppendSparse
from scipy.sparse.linalg import spsolve as dir_solve
from assembly import AssemblyDiffusion3DBoundaries, AssemblyDiffusion3DInterior
from small_functions import GetBoundaryStatus

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
from neighbourhood import GetNeighbourhood

from numba import njit, prange
from numba.experimental import jitclass
from numba import int64, float64
from numba import int64, types, typed
import numba as nb

import shutil

freq_kernel_change=30

spec = [
    ('bound', int64[:]),               # a simple scalar field
    ('coords', float64[:]),
    ('ID', int64),        
    ('BC_value', int64),        
    ('kernel_q', float64[:]),        
    ('kernel_s', float64[:]),        
    ('col_s', int64[:]),
    ('col_q', int64[:]), 
    ('weight', float64), 
    ('neigh', int64[:]), 
    ('block_3D', int64)]
@jitclass(spec)
class node(): 
    def __init__(self, coords, local_ID):
        self.bound=np.zeros(0, dtype=np.int64) #meant to store -1 if it is not a boundary node and
                                #the number of the boundary if it is
        self.coords=coords
        self.ID=local_ID
        
        self.BC_value=0 #Variable of the independent term caused by the BC. Only not 0 when in a boundary
        #The kernels to multiply the unknowns to obtain the value of the slow term at 
        #the node 
        self.kernel_q=np.zeros(0, dtype=np.float64)
        self.kernel_s=np.zeros(0, dtype=np.float64)
        #The kernels with the positions
        self.col_s=np.zeros(0, dtype=np.int64)
        self.col_q=np.zeros(0, dtype=np.int64)
        
    def multiply_by_value(self, value):
        """This function is used when we need to multiply the value of the node 
        but when working in kernel form """
        self.weight=value
        self.kernel_q*=value
        self.kernel_s*=value
        return
    
    def kernels_append(self,arrays_to_append):
        """Function that simplifies a lot the append process"""
        
        a,b,c,d,e,f=arrays_to_append
        
        self.kernel_s=np.concatenate((self.kernel_s,a))
        self.col_s=np.concatenate((self.col_s, b))
        self.kernel_q=np.concatenate((self.kernel_q, c))
        self.col_q=np.concatenate((self.col_q, d))
        return

@njit
def InterpolateFast(x, n, cells_x, cells_y, cells_z, h_3D, bound_status, pos_cells,s_blocks, source_edge, tau_array, pos_s, h_1D, R, D):
    """returns the kernels to obtain an interpolation on the point x. 
    In total it will be 6 kernels, 3 for columns and 3 with data for s, q, and
    C_v respectively"""
    if len(bound_status): #boundary interpolation
        nodes=List([node(x, 0)])
        nodes[0].neigh=GetNeighbourhood(n, cells_x, 
                                         cells_y, 
                                         cells_z, 
                                         GetID(h_3D,cells_x, cells_y, cells_z,x))
        nodes[0].block_3D=GetID(h_3D,cells_x, cells_y, cells_z,nodes[0].coords)
# =============================================================================
#         nodes[0].dual_neigh=nodes[0].neigh
#         dual_neigh=nodes[0].dual_neigh
# =============================================================================
        dual_neigh=nodes[0].neigh
        blocks=np.array([nodes[0].block_3D])
        
    else: #no boundary node (standard interpolation)
        
        blocks=Get8Closest(h_3D,cells_x, cells_y, cells_z,x)
        nodes=List([node(pos_cells[blocks[0]], 0)])
        nodes[0].block_3D=GetID(h_3D,cells_x, cells_y, cells_z,nodes[0].coords)
        nodes[0].neigh=GetNeighbourhood(n, cells_x, 
                                         cells_y, 
                                         cells_z, 
                                         blocks[0])
        dual_neigh=nodes[0].neigh
        
        c=1
        for i in blocks[1:]:
            #Create node object
            
            nodes.append(node(pos_cells[i], c))
            #Append the neighbourhood to the neigh object variable
            nodes[c].neigh=GetNeighbourhood(n, cells_x, 
                                             cells_y, 
                                             cells_z, 
                                             i)
            nodes[c].block_3D=GetID(h_3D,cells_x, cells_y, cells_z,nodes[c].coords)
            dual_neigh=np.concatenate((dual_neigh, nodes[c].neigh))
            c+=1
    kernel_s,col_s,kernel_q, col_q=GetInterpolationKernelFast(x,List(nodes), np.unique(dual_neigh), n, cells_x,
                                       cells_y, cells_z, h_3D, s_blocks, source_edge, tau_array, pos_s, h_1D, R, D)
    return kernel_s,col_s,kernel_q, col_q

@njit
def GetInterpolationKernelFast(x, nodes: types.ListType(node.class_type.instance_type), dual_neigh,n, cells_x, cells_y, cells_z, h_3D, s_blocks, source_edge, tau_array, pos_s, h_1D, R, D):
    """For some reason, when the discretization size of the network is too small, 
    this function provides artifacts. I have not solved this yet"""
    #From the nodes coordinates, their neighbourhoods and their kernels we do the interpolation
    #Therefore, the kernels represent the slow term, the corrected rapid term must be calculated
    #INTERPOLATED PART:
    kernel_s,col_s,kernel_q, col_q=GetI1Fast(x,List(nodes), dual_neigh,D,h_3D,s_blocks, source_edge, tau_array, pos_s, h_1D, R)
    #RAPID (NON INTERPOLATED) PART
    q,sources=KernelPointFast(x, dual_neigh,s_blocks, source_edge, tau_array, pos_s, h_1D,R ,D)
    kernel_q=np.concatenate((kernel_q, q))
    col_q=np.concatenate((col_q, sources))
    
    return kernel_s,col_s,kernel_q, col_q


@njit
def GetI1Fast(x,nodes: types.ListType(node.class_type.instance_type), dual_neigh, D, h_3D, s_blocks, source_edge, tau_array, pos_s, h_1D, R):
    """Returns the kernels of the already Interpolated part, il reste juste ajouter
    le term rapide corrigé
        - x is the point where the concentration is Interpolated"""
    if len(nodes)==8:
        weights=TrilinearInterpolation(x, np.array([h_3D, h_3D, h_3D]))
        kernel_q=np.zeros(0, dtype=np.float64)
        kernel_s=np.zeros(0, dtype=np.float64)
        #The kernels with the positions
        col_s=np.zeros(0, dtype=np.int64)
        col_q=np.zeros(0, dtype=np.int64)
        for i in range(8): #Loop through each of the nodes
           
            uncommon=GetUncommon(dual_neigh, nodes[i].neigh) 
            #The following variable will contain the data kernel for q, the data kernel
            #for C_v and the col kernel i.e. the sources 
            a=KernelPointFast(x, uncommon, s_blocks, source_edge, tau_array, pos_s, h_1D, R, D)
            ######## Changed sign on a[0] 26 mars 18:08
            #I think the negative sign arises from the fact that this is the corrected part of the rapid term so it needs to be subtracted
            
            nodes[i].kernel_q=np.concatenate((nodes[i].kernel_q, -a[0]))
            
            nodes[i].col_q=np.concatenate((nodes[i].col_q, a[1]))
            
            nodes[i].kernel_s=np.array([1], dtype=np.float64)
            nodes[i].col_s=np.array([nodes[i].block_3D])
            
            #This operation is a bit redundant
            nodes[i].multiply_by_value(weights[i])
            kernel_q=np.concatenate((kernel_q, nodes[i].kernel_q))
            kernel_s=np.concatenate((kernel_s, nodes[i].kernel_s))
            
            col_q=np.concatenate((col_q, nodes[i].col_q))
            col_s=np.concatenate((col_s, nodes[i].col_s))
            
        
    
    else: #There are not 8 nodes cause it lies in the boundary so there is no interpolation
        kernel_s=np.array([1], dtype=np.float64) 
        col_s=np.array([nodes[0].block_3D], dtype=np.int64)
        kernel_q=np.zeros(0, dtype=np.float64)
        col_q=np.zeros(0, dtype=np.int64)
    
    return kernel_s,col_s,kernel_q, col_q



@njit
def InterpolatePhiBarBlock(block,n, cells_x, cells_y, cells_z, h_3D, 
                             pos_cells,s_blocks, source_edge,tau_array, pos_s, h_1D, R, D, sources_per_block, quant):
    """Perform the loop in C that loops over the whole network"""
    
    #I multiply by 50 as an estimation, maybe it should be increased
    size_arr=np.sum(quant[GetNeighbourhood(n, cells_x, cells_y, cells_z, block)]) #total amount of sources in the neighbourhood
    kernel_s_array=np.empty(len(s_blocks)*8, dtype=np.float64)
    kernel_q_array=np.empty(size_arr*80*quant[block], dtype=np.float64)
    
    col_s_array=np.empty(len(s_blocks)*8, dtype=np.int64)
    col_q_array=np.empty(size_arr*80*quant[block], dtype=np.int64)
    
    row_s_array=np.zeros(0, dtype=np.int64)
    row_q_array=np.zeros(0, dtype=np.int64)
    
    c_s=0 #counter for s
    c_q=0 #counter for q 
    
    kk=0
    
    #for j in range(len(s_blocks)):
    for j in sources_per_block[block]:

        kernel_s,col_s,kernel_q, col_q=InterpolateFast(pos_s[j], n, cells_x, cells_y,
                              cells_z, h_3D, GetBoundaryStatus(pos_s[j],h_3D, cells_x, cells_y, cells_z), pos_cells,s_blocks, source_edge,
                              tau_array, pos_s, h_1D, R, D)
        
        kernel_s_array[c_s:c_s+len(kernel_s)]=kernel_s
        col_s_array[c_s:c_s+len(kernel_s)]=col_s
        row_s_array=np.concatenate((row_s_array, np.zeros(len(col_s),dtype=np.int64)+j))
        c_s+=len(kernel_s)                    
        
        kernel_q_array[c_q: c_q+len(kernel_q)]=kernel_q
        col_q_array[c_q:c_q+len(kernel_q)]=col_q
        row_q_array=np.concatenate((row_q_array, np.zeros(len(col_q),dtype=np.int64)+j))
        c_q+=len(kernel_q)
        
        kk+=1
    #Now we eliminate the left over space
    kernel_s_array=kernel_s_array[:c_s]
    col_s_array=col_s_array[:c_s]
    
    kernel_q_array=kernel_q_array[:c_q]
    col_q_array=col_q_array[:c_q]
    
    return(kernel_s_array, row_s_array, col_s_array, 
           kernel_q_array, row_q_array,col_q_array)

def RetrievePhiBar(mat_path, phi_bar_path, S, size_mesh, uni_s_blocks):
    
    kernel_q=np.zeros(0, dtype=np.float64)
    kernel_s=np.zeros(0, dtype=np.float64)
    #The kernels with the positions
    col_s=np.zeros(0, dtype=np.int64)
    col_q=np.zeros(0, dtype=np.int64)

    row_s=np.zeros(0, dtype=np.int64)
    row_q=np.zeros(0, dtype=np.int64)
    
    phi_bar_s=sp.sparse.csc_matrix((S, size_mesh))
    phi_bar_q=sp.sparse.csc_matrix((S, S))
    
    d=0
    c=0
    for i in uni_s_blocks:
        print("Retrieve block: ", i)
        c+=1
        print(c)
        list_of_kernels=RetrieveBlockPhiBar(phi_bar_path,i)
        
        kernel_s=np.concatenate((kernel_s, list_of_kernels[0]))
        kernel_q=np.concatenate((kernel_q, list_of_kernels[3]))
        
        col_s=np.concatenate((col_s, list_of_kernels[2]))
        col_q=np.concatenate((col_q, list_of_kernels[5]))
        
        row_s=np.concatenate((row_s, list_of_kernels[1]))
        row_q=np.concatenate((row_q, list_of_kernels[4]))

        if c%freq_kernel_change==0:
            phi_bar_s+=sp.sparse.csc_matrix((kernel_s,(row_s, col_s)), shape=(S, size_mesh))
            phi_bar_q+=sp.sparse.csc_matrix((kernel_q,(row_q, col_q)), shape=(S, S))
            kernel_q=np.zeros(0, dtype=np.float64)
            kernel_s=np.zeros(0, dtype=np.float64)
            #The kernels with the positions
            col_s=np.zeros(0, dtype=np.int64)
            col_q=np.zeros(0, dtype=np.int64)
    
            row_s=np.zeros(0, dtype=np.int64)
            row_q=np.zeros(0, dtype=np.int64)
            
        if c==len(uni_s_blocks):
            #pdb.set_trace()
            phi_bar_s+=sp.sparse.csc_matrix((kernel_s,(row_s, col_s)), shape=(S, size_mesh))
            phi_bar_q+=sp.sparse.csc_matrix((kernel_q,(row_q, col_q)), shape=(S, S))
            
            sp.sparse.save_npz(mat_path + '/phi_bar_s', phi_bar_s)
            sp.sparse.save_npz(mat_path + '/phi_bar_q', phi_bar_q)
            shutil.rmtree(phi_bar_path)
            print(f"Folder '{phi_bar_path}' deleted successfully.")
            
    return phi_bar_s, phi_bar_q
   
def RetrieveBlockPhiBar(path, block):
    kernel_s=np.load(path + '/{}_kernel_s.npy'.format(block))
    kernel_q=np.load(path + '/{}_kernel_q.npy'.format(block))
    col_s=np.load(path + '/{}_col_s.npy'.format(block))
    col_q=np.load(path + '/{}_col_q.npy'.format(block))
    row_s=np.load(path + '/{}_row_s.npy'.format(block))
    row_q=np.load(path + '/{}_row_q.npy'.format(block))
    return (kernel_s, row_s,col_s,
            kernel_q,  row_q, col_q  )
    
    
