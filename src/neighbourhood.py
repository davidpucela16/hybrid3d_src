#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:17:20 2023

@author: pdavid
"""

import numpy as np
import pdb
from numba import njit
from small_functions import in1D
# =============================================================================
# def GetNeighbourhood(n, cells_x, cells_y, cells_z, block_ID):
#     """Will return the neighbourhood of a given block for a given n
#     in a mesh made of square cells
# 
#     It will assume cells_x=ylen"""
#     
#     step_y=cells_z
#     step_x=cells_z*cells_y
#     
#     pad_x = np.concatenate((np.zeros((n)) - 1, np.arange(cells_x), np.zeros((n)) - 1)).astype(int)
#     pad_y = np.concatenate((np.zeros((n)) - 1, np.arange(cells_y), np.zeros((n)) - 1)).astype(int)
#     pad_z = np.concatenate((np.zeros((n)) - 1, np.arange(cells_z), np.zeros((n)) - 1)).astype(int)
#     pos_x, pos_y, pos_z =int( block_ID/(cells_y*cells_z)), int(int(block_ID%step_x)/cells_z), int(block_ID%cells_z)
#     loc_x = pad_x[pos_x : pos_x + 2 * n + 1]
#     loc_x = loc_x[np.where(loc_x >= 0)]
#     loc_y = pad_y[pos_y : pos_y + 2 * n + 1]
#     loc_y = loc_y[np.where(loc_y >= 0)]
#     loc_z = pad_z[pos_z : pos_z + 2 * n + 1]
#     loc_z = loc_z[np.where(loc_z >= 0)]
# 
#     cube = np.zeros((len(loc_z),len(loc_y), len(loc_x)), dtype=int)
#     
#     d=0
#     for i in loc_x:
#         c=0
#         for j in loc_y:
#             
#             cube[:,c,d] = loc_z + step_x*i + step_y*j
#             c+=1
#         d+=1
#     # print("the neighbourhood", square)
#     return np.ndarray.flatten(cube)
# =============================================================================


def GetMultipleNeigh(n, cells_x, cells_y, cells_z, array_of_blocks):
    """This function will call the GetNeighbourhood function for multiple blocks to
    return the ensemble of the neighbourhood for all the blocks"""
    full_neigh = set()
    for i in array_of_blocks:
        full_neigh = full_neigh | set(GetNeighbourhood(n, cells_x, i))
    return np.array(list(full_neigh), dtype=int)

@njit
def GetUncommon(k_neigh, n_neigh):
    """returns the cells of the first neighbourhood that has not in common with the
    second neighbourhood"""

    neigh_k_unc = k_neigh[np.invert(in1D(k_neigh, n_neigh))]
    return neigh_k_unc


@njit
def GetNeighbourhood(n, cells_x, cells_y, cells_z, block_ID):
    """Will return the neighbourhood of a given block for a given n
    in a mesh made of square cells

    It will assume cells_x=ylen
    
    50 times faster than the non optimized"""
    
    step_y=cells_z
    step_x=cells_z*cells_y
    pad_x = np.concatenate((np.zeros(n)-1,  np.arange(cells_x), np.zeros(n)-1))
    pad_y = np.concatenate((np.zeros(n)-1,  np.arange(cells_y), np.zeros(n)-1))
    pad_z = np.concatenate((np.zeros(n)-1,  np.arange(cells_z), np.zeros(n)-1))
    pos_x, pos_y, pos_z =int( block_ID/(cells_y*cells_z)), int(int(block_ID%step_x)/cells_z), int(block_ID%cells_z)
    
    loc_x = pad_x[pos_x : pos_x + 2 * n + 1]
    loc_x = loc_x[np.where(loc_x >= 0)]
    loc_y = pad_y[pos_y : pos_y + 2 * n + 1]
    loc_y = loc_y[np.where(loc_y >= 0)]
    loc_z = pad_z[pos_z : pos_z + 2 * n + 1]
    loc_z = loc_z[np.where(loc_z >= 0)]

    cube = np.zeros((len(loc_z)*len(loc_y)*len(loc_x)), dtype=np.int64)
    
    c=0
    for i in loc_x:
        for j in loc_y:
            cube[c*len(loc_z): (c+1)*len(loc_z)] = loc_z + step_x*i + step_y*j
            c+=1
        
    return cube