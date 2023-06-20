#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:17:50 2023

@author: pdavid
"""
import numpy as np 
import scipy as sp 
import pdb
from small_functions import AppendSparse


def AssemblyDiffusion3DInterior(mesh_object):
    #arrays containing the non-zero entries of the matrix
    row_array=np.array([]) #Contains the row values of each entry
    col_array=np.array([]) #Contains the colume values of each entry
    data_array=np.array([]) #Contains the entry value
    
    mesh_object.AssemblyBoundaryVectors()
    
    for k in mesh_object.int: #Loop that goes through each non-boundary cell
        rows=np.zeros(7)+k
        cols=mesh_object.GetDiffStencil(k)
        data=np.array([-6,1,1,1,1,1,1])
        
        row_array=np.concatenate((row_array, rows))
        col_array=np.concatenate((col_array, cols))
        data_array=np.concatenate((data_array, data))
        
    c=0
    for k in mesh_object.full_boundary: 
        neighs=np.squeeze(np.array([mesh_object.connect_list[c]]))
        
        rows=np.zeros(len(neighs)+1)+k
        cols=np.concatenate((np.array([k]) , neighs))
        data=np.ones(len(neighs)+1)
        data[0]=-len(neighs)
    
        row_array=np.concatenate((row_array, rows))
        col_array=np.concatenate((col_array, cols))
        data_array=np.concatenate((data_array, data))
        c+=1
        
        # =============================================================================
        #       We only multiply the non boundary part of the matrix by h because in the boundaries assembly we need to include the h due to the difference
        #       between the Neumann and Dirichlet boundary conditions. In short AssemblyDiffusion3DInterior returns data that needs to be multiplied by h 
        #       while AssemblyDiffusion3DBoundaries is already dimensionally consistent
        # =============================================================================

    return(np.array([row_array, col_array, data_array]))

        
def AssemblyDiffusion3DBoundaries(mesh_object, BC_type, BC_value):
    """For now, only a single value for the gradient or Dirichlet BC can be given
    and this function will transform the already assembled matrix to include the 
    boundary ocnditions"""
    row_array=np.array([]) #Contains the row values of each entry
    col_array=np.array([]) #Contains the colume values of each entry
    data_array=np.array([]) #Contains the entry value
    
    BC_array=np.zeros(mesh_object.size_mesh) #The array with the BC values 
    c=0
    for bound in mesh_object.full_full_boundary:
    #This loop goes through each of the boundary cells, and it goes repeatedly 
    #through the edges and corners accordingly
        for k in bound: #Make sure this is the correct boundary variable
            if BC_type[c]=="Dirichlet":
                row_array=np.append(row_array, k)
                col_array=np.append(col_array, k)
                data_array=np.append(data_array, -2*mesh_object.h)
                BC_array[k]=2*BC_value[c]*mesh_object.h
                
            if BC_type[c]=="Neumann":
                BC_array[k]=BC_value[c]*mesh_object.h**2
        c+=1
        
        # =============================================================================
        #       We only multiply the non boundary part of the matrix by h because in the boundaries assembly we need to include the h due to the difference
        #       between the Neumann and Dirichlet boundary conditions. In short AssemblyDiffusion3DInterior returns data that needs to be multiplied by h 
        #       while AssemblyDiffusion3DBoundaries is already dimensionally consistent
        # =============================================================================
    return(row_array, col_array, data_array, BC_array)      


    
