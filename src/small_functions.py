#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:54:57 2023

@author: pdavid
"""
import numpy as np
from numba import njit

@njit
def GetBoundaryStatus(coords, h_3D, cells_x, cells_y, cells_z):
    """I need this to be out of the class to be able to be called by the InterpolatePhiBar"""
    bound_status=np.zeros(0, dtype=np.int64) #array that contains the boundaries that lie less than h/2 from the point
    if int(coords[0]/(h_3D/2))==0: bound_status=np.append(bound_status, 5) #down
    elif int(coords[0]/(h_3D/2))==2*cells_x-1: bound_status=np.append(bound_status, 4) #top
    if int(coords[1]/(h_3D/2))==0: bound_status=np.append(bound_status, 3) #west
    elif int(coords[1]/(h_3D/2))==2*cells_y-1: bound_status=np.append(bound_status, 2) #east
    if int(coords[2]/(h_3D/2))==0: bound_status=np.append(bound_status, 1) #south
    elif int(coords[2]/(h_3D/2))==2*cells_z-1: bound_status=np.append(bound_status, 0) #north
    return bound_status



@njit
def in1D(arr1, arr2):
    """Return a boolean array indicating which elements of `arr1` are in `arr2`."""
    arr2_sorted = np.sort(arr2)
    indices = np.searchsorted(arr2_sorted, arr1)
    return (arr2_sorted[indices] == arr1)

def AppendSparse(arr, d, r, c):
    data_arr=np.append(arr[0], d)
    row_arr=np.append(arr[1], r)
    col_arr=np.append(arr[2], c)
    return(data_arr, row_arr, col_arr)
@njit
def TrilinearInterpolation(pos, h):
    """"""
    #The following is the equivalent of the operator % but I wanna make sure it works
    pos=pos-h/2
    pos_x=pos[0]-np.floor(pos[0]/h[0])*h[0]
    pos_y=pos[1]-np.floor(pos[1]/h[1])*h[1]
    pos_z=pos[2]-np.floor(pos[2]/h[2])*h[2]
    
    x,y,z=pos_x/h[0], pos_y/h[1], pos_z/h[2] #to obtain a relative position
    
    A=np.array([[(1-x)*(1-y)*(1-z)],
                [(1-x)*(1-y)*z],
                [(1-x)*y*(1-z)],
                [(1-x)*y*z],
                [x*(1-y)*(1-z)],
                [x*(1-y)*z],
                [x*y*(1-z)],
                [x*y*z]], dtype=np.float64)
    
    return A


def auto_TrilinearInterpolation(x, nodes):
    """Before calling TrilinearInterpolation, this function takes the nodes 
    as arguments and reorganizes them in the correct order"""
    
    coords=np.zeros((0,3))
    for i in nodes:
        coords=np.vstack((coords, i.coords))
    
    hx=np.max(coords[:,0])-np.min(coords[:,0])
    hy=np.max(coords[:,1])-np.min(coords[:,1])
    hz=np.max(coords[:,2])-np.min(coords[:,2])
    
    order=np.zeros(8)
    zeroth=np.argmin(np.sum(coords, axis=1))[0]
    order[np.where(coords[0,:])==coords[zeroth]+hx]+=4
    order[np.where(coords[1,:])==coords[zeroth]+hy]+=2
    order[np.where(coords[2,:])==coords[zeroth]+hz]+=1
    
    return TrilinearInterpolation(x, np.array([hx,hy, hz]))[order]
    


def FromBoundaryGetNormal(bound_num):
    normal=np.array([[0,0,1],
                     [0,0,-1],
                     [0,1,0],
                     [0,-1,0],
                     [1,0,0],
                     [-1,0,0]])
    return normal[bound_num]
    
    
