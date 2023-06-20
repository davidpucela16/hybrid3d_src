#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:35:37 2023

@author: pdavid
"""

import numpy as np 
import pdb

import os 
directory_script = os.path.dirname(__file__)
os.chdir(directory_script)

from numba import njit

#%%
# =============================================================================
# def GetCorners(step_y, step_x, size_mesh):
#     """Retrunst the IDs of the corners"""
#     corners=np.array([0,step_y-1, step_x-step_y, step_x-1])
#     corners=np.concatenate((corners, corners + size_mesh - step_x))
#     return(corners)
# 
# def GetZEdges(corners):
#     """Returns the IDs of the edges parallel to the z axis without the corners"""
#     a=np.arange(corners[0], corners[1])
#     b=np.arange(corners[4], corners[5])
#     c=np.arange(corners[6], corners[7])
#     d=np.arange(corners[2], corners[3])
#     z_edges=np.concatenate(([a[1:]],[b[1:]],[c[1:]],[d[1:]]), axis=0)
#     return(z_edges)
# 
# def GetYEdges(corners, step_y):
#     """Returns the IDs of the edges parallel to the y axis without the corners"""
#     a=np.arange(corners[0], corners[2], step_y)
#     b=np.arange(corners[1], corners[3], step_y)
#     c=np.arange(corners[5], corners[7], step_y)
#     d=np.arange(corners[4], corners[6], step_y)
#     y_edges=np.concatenate(([a[1:]],[b[1:]],[c[1:]],[d[1:]]), axis=0)
#     return(y_edges)
# 
# def GetXEdges(corners, step_x):
#     """Returns the IDs of the edges parallel to the x axis without the corners"""
#     a=np.arange(corners[0], corners[4], step_x)
#     b=np.arange(corners[2], corners[6], step_x)
#     c=np.arange(corners[3], corners[7], step_x)
#     d=np.arange(corners[1], corners[5], step_x)
#     x_edges=np.concatenate(([a[1:]],[b[1:]],[c[1:]],[d[1:]]), axis=0)
#     return(x_edges)
# 
# def GetBoundaries(corners, z_edges, y_edges, x_edges, cells_x,cells_y, cells_z):
#     z_e=z_edges
#     y_e=y_edges
#     x_e=x_edges
#     step_y, step_x, size_mesh=cells_z, cells_z*cells_y, cells_x*cells_y*cells_z
#     
#     #Array that contains all the IDs of the cells belonging to an edge or a corner:
#     edges_plus_corners=np.concatenate((corners, z_e, y_e, x_e))
#     
#     #All included cells in the north boundary
#     north=np.arange(corners[1], corners[7]+1, step_y)
#     #All included cells in the south boundary
#     south=np.arange(corners[0], corners[6]+1, step_y)
# 
#     east=np.array([], dtype=int)
#     west=np.array([], dtype=int)
#     for i in range(cells_x):
#         east=np.concatenate((east, np.arange(step_y)+i*step_x))
#         west=np.concatenate((west, np.arange(step_y)+i*step_x + corners[2]))
#     
#     top=np.arange(corners[4], corners[7]+1)
#     down=np.arange(corners[0], corners[3]+1)
#     
#     #We save the cells in the boundaries including corners and edges
#     full_north=north
#     full_south=south
#     full_east=east
#     full_west=west
#     full_top=top
#     full_down=down
#     
#     #Array with repeated entries for corners and edges
#     full_full_boundary=np.array([full_north, 
#                                       full_south, 
#                                       full_east, 
#                                       full_west, 
#                                       full_top, 
#                                       full_down])
#     
#     #The cells in the boundaries excluding corners and edges:
#     north=np.delete(north, np.where(np.in1d(north, edges_plus_corners)))
#     south=np.delete(south, np.where(np.in1d(south, edges_plus_corners)))
#     east=np.delete(east, np.where(np.in1d(east, edges_plus_corners)))
#     west=np.delete(west, np.where(np.in1d(west, edges_plus_corners)))
#     top=np.delete(top, np.where(np.in1d(top, edges_plus_corners)))
#     down=np.delete(down, np.where(np.in1d(down, edges_plus_corners)))
#     
#     #array that contains the boundary cells without repetition
#     full_bound=np.concatenate((north, south, east, west, top, down, edges_plus_corners))
#     
#     interior=np.delete(np.arange(size_mesh), np.where(np.in1d(np.arange(size_mesh), full_bound)))
# =============================================================================

#%%
class cart_mesh_3D():
    def __init__(self, L, cells_x):
        """We only allow one h for all directions (cubic cells). Therefore cells_x
        is an integer"""
        h=L[0]/cells_x
        cells_y, cells_z=int(np.ceil(L[1]/h)), int(np.ceil(L[2]/h))
        
        self.cells_x, self.cells_y, self.cells_z=cells_x, cells_y, cells_z
        
        L[1], L[2]=cells_y*h, cells_z*h
        
        Lx, Ly, Lz=L
        self.x=np.linspace(h/2, Lx- h/2, cells_x)
        self.y=np.linspace(h/2, Ly-h/2, cells_y)
        self.z=np.linspace(h/2, Lz-h/2, cells_z)
        
        self.mesh=[self.x,self.y,self.z]
        
        self.cells=np.array([cells_x, cells_y, cells_z])
        self.h=h
        self.L=L
        
        self.ids_array=np.arange(cells_x*cells_y*cells_z)

        self.step_y=cells_z
        self.step_x=cells_z*cells_y
        self.size_mesh=cells_x*cells_y*cells_z
        #self.corners=GetCorners(self.step_x, self.step_y, self.size_mesh)
        self.GetCorners()
        
        self.pos_cells=np.zeros((self.size_mesh, 3), dtype=np.float64)
        self.AssemblyBoundaryVectors()
        
        return
    
    def AssemblyBoundaryVectors(self):
        """Assembles all the necessary boundary arrays:
            - self.int
            - self.full_boundary
            - self.connect_matrix
            - ..."""
        self.GetBoundaries()
        self.AssemblyListBoundaryStencils()
        self.GetConnectivityMatrix()
        return
        
    def GetCoords(self, k):
        """Returns the coordinate of the cell center of Id=k
        k must be an int"""
        
        c_x=int(k//(len(self.y)*len(self.z)))
        c_y=int(int(k%self.step_x)/len(self.z))
        c_z=int(k%len(self.z))
        
        return(np.array([self.x[c_x],self.y[c_y],self.z[c_z]]))
    
    
    def GetCorners(self):
        self.corners=np.array([0,self.step_y-1, self.step_x-self.step_y, self.step_x-1])
        self.corners=np.concatenate((self.corners, self.corners + self.size_mesh - self.step_x))
        return(self.corners)
    
    def GetZEdges(self):
        a=np.arange(self.corners[0], self.corners[1])
        b=np.arange(self.corners[4], self.corners[5])
        c=np.arange(self.corners[6], self.corners[7])
        d=np.arange(self.corners[2], self.corners[3])
        self.z_edges=np.concatenate(([a[1:]],[b[1:]],[c[1:]],[d[1:]]), axis=0)
        return(self.z_edges)
    
    def GetYEdges(self):
        a=np.arange(self.corners[0], self.corners[2], self.step_y)
        b=np.arange(self.corners[1], self.corners[3], self.step_y)
        c=np.arange(self.corners[5], self.corners[7], self.step_y)
        d=np.arange(self.corners[4], self.corners[6], self.step_y)
        self.y_edges=np.concatenate(([a[1:]],[b[1:]],[c[1:]],[d[1:]]), axis=0)
        return(self.y_edges)
    
    def GetXEdges(self):
        a=np.arange(self.corners[0], self.corners[4], self.step_x)
        b=np.arange(self.corners[2], self.corners[6], self.step_x)
        c=np.arange(self.corners[3], self.corners[7], self.step_x)
        d=np.arange(self.corners[1], self.corners[5], self.step_x)
        self.x_edges=np.concatenate(([a[1:]],[b[1:]],[c[1:]],[d[1:]]), axis=0)
        return(self.x_edges)
    
    def GetBoundaries(self):
        c=self.GetCorners()
        z_e=np.ndarray.flatten(self.GetZEdges())
        y_e=np.ndarray.flatten(self.GetYEdges())
        x_e=np.ndarray.flatten(self.GetXEdges())
        
        edges_plus_corners=np.concatenate((c, z_e, y_e, x_e))
        
        
        north=np.arange(self.corners[1], self.corners[7]+1, self.step_y)
        
        south=np.arange(self.corners[0], self.corners[6]+1, self.step_y)
    
        east=np.array([], dtype=int)
        west=np.array([], dtype=int)
        for i in range(len(self.x)):
            west=np.concatenate((west, np.arange(self.step_y)+i*self.step_x))
            east=np.concatenate((east, np.arange(self.step_y)+i*self.step_x + self.corners[2]))
        
        top=np.arange(self.corners[4], self.corners[7]+1)
        down=np.arange(self.corners[0], self.corners[3]+1)
        
        #We save the cells in the boundaries including corners and edges
        self.full_north=north
        self.full_south=south
        self.full_east=east
        self.full_west=west
        self.full_top=top
        self.full_down=down
        
        #Array with repeated entries for corners and edges
        self.full_full_boundary=np.array([self.full_north, 
                                          self.full_south, 
                                          self.full_east, 
                                          self.full_west, 
                                          self.full_top, 
                                          self.full_down])
        
        #The cells in the boundaries excluding corners and edges:
        self.north=np.delete(north, np.where(np.in1d(north, edges_plus_corners)))
        self.south=np.delete(south, np.where(np.in1d(south, edges_plus_corners)))
        self.east=np.delete(east, np.where(np.in1d(east, edges_plus_corners)))
        self.west=np.delete(west, np.where(np.in1d(west, edges_plus_corners)))
        self.top=np.delete(top, np.where(np.in1d(top, edges_plus_corners)))
        self.down=np.delete(down, np.where(np.in1d(down, edges_plus_corners)))
        
        #array that contains the boundary cells without repetition
        self.full_bound=np.concatenate((self.north, self.south, self.east, self.west, self.top, self.down, edges_plus_corners))
        
        self.int=np.delete(np.arange(self.size_mesh), np.where(np.in1d(np.arange(self.size_mesh), self.full_bound)))
        
        
    
    def AssemblyListBoundaryStencils(self):
        """This function is meant to create a list with as many entries as boundary
        cells that provides the IDs of the neighbouring FVs, as well as the boundary 
        they belong to
        
        Boundaries: north, south, east, west, top, down = 0,1,2,3,4,5
        
        This function is not yet debugged
        
        Variables:
        type_boundary: a list where each entry corresponds to a boundary cell
            and where each entry contains one or several values corresponding to
            the boundary it lies in.
        
        connect_matrix: """
        step_x, step_y=self.step_x, self.step_y
        full_boundary=np.array([], dtype=int) #It will store the IDs of the boundary cells
        
        connect_matrix=[]
        type_boundary=[]  
        #corner 0
        type_boundary.append([1,3,5])
        #corner 1
        type_boundary.append([0,3,5])
        #corner 2
        type_boundary.append([1,2,5])
        #corner 3
        type_boundary.append([0,2,5])
        
        #corner 4
        type_boundary.append([1,3,4])
        #corner 5
        type_boundary.append([0,3,4])
        #corner 6
        type_boundary.append([1,2,4])
        #corner 7
        type_boundary.append([0,2,4])
        
        full_boundary=np.concatenate((full_boundary, self.corners))
        
        c=0
        for i in self.GetZEdges(): 
        #Loop through each of the edges parallel to the z axis 
        
            if c==0: #The boundaries are West and Down
                b1=3
                b2=5
            elif c==1: #The boundaries are West and Top
                b1=3
                b2=4
            elif c==2: #The boundaries are East and Top
                b1=2
                b2=4
            elif c==3:#The boundaries are East and Top
                b1=2
                b2=5 
            #The following is a list containing as many entries as i   
            b_list=np.array([b1+np.zeros(len(i)), b2+np.zeros(len(i))], dtype=int).T.tolist()
            type_boundary=type_boundary+b_list #Append the b_list that contains the IDs of the boundaries
                                                #of the cells in i.
            full_boundary=np.concatenate((full_boundary, i)) 
            c+=1
        c=0
        for i in self.GetYEdges():
            if c==0: #The boundaries are South and Down 
                b1=1
                b2=5
            elif c==1:#The boundaries are North and Down
                b1=0
                b2=5
            elif c==2: #The boundaries are North and Top
                b1=0
                b2=4
            elif c==3: #The boundaries are South and Top
                b1=1
                b2=4 
            b_list=np.array([b1+np.zeros(len(i)), b2+np.zeros(len(i))], dtype=int).T.tolist()
            type_boundary=type_boundary+b_list
            full_boundary=np.concatenate((full_boundary, i))
            c+=1
            
        c=0
        for i in self.GetXEdges():
            if c==0:
                b1=1
                b2=3
            elif c==1:
                b1=1
                b2=2
            elif c==2:
                b1=0
                b2=2
            elif c==3:
                b1=0
                b2=3 
            b_list=np.array([b1+np.zeros(len(i)), b2+np.zeros(len(i))], dtype=int).T.tolist()
            type_boundary=type_boundary+b_list
            full_boundary=np.concatenate((full_boundary, i))
            c+=1

        c=0
        for i in [self.north, self.south, self.east, self.west, self.top, self.down]:
            type_boundary=type_boundary +[c]*len(i)
            full_boundary=np.concatenate((full_boundary, i))
            c+=1
        self.full_boundary=full_boundary
        self.type_boundary=type_boundary
    
    def GetDiffStencil(self, k):
        """Returns the star stensil"""
# =============================================================================
#         This needs to be continued when I think of the better way to assemble
#         the diffusion matrix
# =============================================================================
        return(np.array([k,k+1, k-1, k+self.step_y, k-self.step_y, k+self.step_x, k-self.step_x]))
    
    def GetConnectivityMatrix(self):
        """ ????????????????????????????????????????????????????????????????????????????????
        I think both this and the previous function work well, but I have not debugged them yet
        ???????????????????????????????????????????????????????????????????????????????????
        
        This function will provide a list self.connec_list where each entry corresponds
        with a boundary cell and contains the IDs of the real neighbouring blocks
        """
        stencil=np.array([+1,-1,self.step_y, -self.step_y, self.step_x, -self.step_x])
        connect_list=[]
        for i in range(len(self.full_boundary)):
            new_stencil=np.delete(stencil , self.type_boundary[i])+self.full_boundary[i]
            connect_list=connect_list+[new_stencil.tolist()]
        self.connect_list=connect_list
        return 
    def GetOrderedConnectivityMatrix(self):
        ordered_connect_matrix=[]
        for k in range(self.size_mesh):
            self.pos_cells[k]=self.GetCoords(k)
            if k in self.full_boundary:
                ordered_connect_matrix = ordered_connect_matrix + [self.connect_list[np.where(self.full_boundary==k)[0][0]]]
            else:
                ordered_connect_matrix = ordered_connect_matrix + [self.GetDiffStencil(k).tolist()]
        self.ordered_connect_matrix = ordered_connect_matrix
        return ordered_connect_matrix
    
    def GetXSlice(self, crds):
        """Returns an array with the IDs of the cells of an slice perpendicular 
        to the x axis closest to the coordinates given in crds
        
        crds is the coordinate along the relevant axis"""
        k=GetID(self.h,self.cells_x, self.cells_y, self.cells_z,np.array([crds, 0,0])) #This is the lowest ID of the slice 
        
        array=np.array([], dtype=int)
        c=0
        for j in range(self.cells[1]): #Loop throught 
            array=np.concatenate((array, np.arange(self.cells[-1])+self.step_y*c))
            c+=1
        return array+k
    
    def GetYSlice(self,crds):
        """Returns an array with the IDs of the cells of an slice perpendicular 
        to the y axis closest to the coordinates given in crds
        
                crds is the coordinate along the relevant axis"""
        k=GetID(self.h,self.cells_x, self.cells_y, self.cells_z,np.array([0,crds, 0])) #This is the lowest ID of the slice 
        
        array=np.array([], dtype=int)
        c=0
        for j in range(self.cells[0]): #Loop throught 
            array=np.concatenate((array, np.arange(self.cells[0])+self.step_x*c))
            c+=1
        return array+k
    
    def GetZSlice(self, crds):
        """ crds is the coordinate along the relevant axis"""
        k=GetID(self.h,self.cells_x, self.cells_y, self.cells_z,np.array([0,0, crds])) #This is the lowest ID of the slice 
        
        array=np.arange(0, self.size_mesh, self.cells[-1]).astype(int)
        
        return array+k
    
    
    def dual_mesh(self):
        self.dual_x=np.concatenate(0,self.x, self.L[0])
        self.dual_y=np.concatenate(0,self.y, self.L[1])
        self.dual_z=np.concatenate(0,self.z, self.L[2])
        
        self.dual_mesh=[self.daul_x,self.daul_y,self.dual_z]
        
        self.dual_step_y=len(self.dual_z)
        self.dual_step_x=len(self.dual_z)*len(self.dual_y)
        
    
@njit
def GetID(h,cells_x, cells_y, cells_z,crds):
    """pos is an np.array with the values of the three coordinates
    The function returns the position along the mesh cells array"""
    pos_x=((crds[0])//(h/2))//2
    pos_y=((crds[1])//(h/2))//2
    pos_z=((crds[2])//(h/2))//2
    
    return(int(pos_x*cells_z*cells_y+pos_y*cells_z+pos_z))

@njit
def Get8Closest(h,cells_x, cells_y, cells_z,crds):
    """This function returns the (8) closest Cartesian grid centers
    - x is the position (array of length=3)
    - h is the discretization size (float)
    """
    step_y=cells_z
    step_x=cells_z*cells_y
    arr=np.zeros(0, dtype=np.int64)
    for i in crds: #Loop through each of the axis 
        b=int(int(i//(h/2))%2)*2-1
        arr=np.append(arr, b)
    ID=GetID(h,cells_x, cells_y, cells_z,crds) #ID of the containing block
    
    blocks=np.array([ID, ID+arr[2]], dtype=np.int64)
    blocks=np.append(blocks, blocks+arr[1]*step_y)
    blocks=np.append(blocks, blocks+arr[0]*step_x)
    
    return(np.sort(blocks))

#%%        
        

                
        
        