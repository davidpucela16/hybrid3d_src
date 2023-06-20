#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:56:05 2023

@author: pdavid
"""
from GreenFast import GradPoint, LogLine, SimpsonSurface, GetSourcePotential,GetSelfInfluence
from neighbourhood import GetNeighbourhood, GetMultipleNeigh
from assembly_1D import flow
import numpy as np
import pdb
from numba import njit
from small_functions import in1D
from mesh import GetID
from numba.typed import List

class mesh_1D():
    def __init__(self, startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters,h_approx, D, *flow_vars):
        """Generates the 1D mesh of cylinders with their centers stored within pos_s       
        
            - startVertex contains the ID of the starting vertex for each edge
           - endVertex: same thing but with the end vertex
           - vertex_to_edge contains the edges each vertex is connected to
           - pos_vertex is the position in the three dimensions of each vertex
           - h is the discretization size of the 1D mesh
           - R is the array of radii for each edge
           """
           
        self.D=D
        L=np.sum((pos_vertex[endVertex] - pos_vertex[startVertex])**2, axis=1)**0.5
        self.L=L
        
        h=self.CalculateDiscretizatinSize(h_approx)
        self.tau=np.divide((pos_vertex[endVertex] - pos_vertex[startVertex]).T,L).T
        self.edges=np.arange(len(startVertex)) #Total number of edges in the network
        
        self.startVertex=startVertex
        self.endVertex=endVertex
        
        self.vertex_to_edge=vertex_to_edge
        pos_s=np.zeros((0,3))
        self.source_edge=np.array([], dtype=int) #Array that returns for each cyl source the edge it lies in
        for i in self.edges:
            local=np.linspace(h[i]/2, L[i]-h[i]/2, int(L[i]/h[i]))
            glob=np.multiply.outer(local, self.tau[i])+pos_vertex[startVertex[i]]
            pos_s=np.concatenate([pos_s, glob],axis=0)
            self.source_edge=np.append(self.source_edge, np.zeros(len(local), dtype=int)+i)
            
        self.pos_s=pos_s
        
        self.R=diameters/2 #One entry per edge
        
        if flow_vars:
            P,viscosity, bound_vertices=flow
            fl=flow(bound_vertices, P, L, diameters, startVertex, endVertex)
            fl.solver()
            self.U=fl.get_U()
        
        return
    def CalculateDiscretizatinSize(self, h_approx):
        """For the discretization size of the network, we propose a size h_approx and then 
        calculate based on that value the discretization size self.h on each vessel
        
        Since some vessels are too short to keep a discretization size close to h_approx, for
        those specific cases h_approx is divided by three to ensure every vessel is composed of
        at least 3 cylinders"""
        h=np.zeros(len(self.L), dtype=np.float64)
        for i in range(len(self.L)):
            if np.around(self.L[i]/h_approx) > 3:
                h[i]=self.L[i]/np.around(self.L[i]/h_approx)
            else:
                h[i]=self.L[i]/3
        self.cells=self.L/h
        self.cells=self.cells.astype(int)
        self.h=h
        return h
# =============================================================================
#     def PositionalArrays(self, mesh_3D):
#         """This function is the pre processing step. It is meant to create the s_blocks
#         and uni_s_blocks arrays which will be used extensively throughout. s_blocks represents
#         the block where each source is located, uni_s_blocks contains all the source blocks
#         in a given order that will be respected throughout the resolution
#         
#             - h_cart is the size of the cartesian mesh
#             
#         THIS IS DEPRECATED, THE FASTER FUNCTION WITH @NJIT IS DEFINED BELOW
#         
#         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         SHOULD BE DELETED SOON
#         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
#             
#         # pos_s will dictate the ID of the sources by the order they are kept in it!
#         print("THIS FUNCTION IS DEPRECATED!!")
#         
#         s_blocks = np.array([]).astype(int)
#         uni_s_blocks = np.array([], dtype=int)
#         self.a_array=np.zeros((0,3))
#         self.b_array=np.zeros((0,3))
#         for i in range(len(self.pos_s)):
#             ed=self.source_edge[i] #Current edge (int) the source lies on 
#             x_j=self.pos_s[i] #Center of the cylinder
#             u=np.array([x_j-self.tau[ed]*self.h[ed]/2 ,x_j+self.tau[ed]*self.h[ed]/2]) #a and b 
#             
#             self.a_array=np.vstack((self.a_array, u[0]))
#             self.b_array=np.vstack((self.b_array, u[1]))
#             
#             s_blocks=np.append(s_blocks, GetID(mesh_3D.h,mesh_3D.cells_x, mesh_3D.cells_y, mesh_3D.cells_z,x_j))
# # =============================================================================
# #             if s_blocks[-1] not in uni_s_blocks:
# #                 uni_s_blocks = np.append(uni_s_blocks, s_blocks[-1])
# # =============================================================================
#         self.s_blocks=s_blocks
#         self.uni_s_blocks, self.counts = np.unique(s_blocks, return_counts=True)
# 
#         total_sb = len(uni_s_blocks)  # total amount of source blocks
#         self.total_sb = total_sb
# =============================================================================
        
    def PositionalArraysFast(self, mesh_3D):
        self.s_blocks, self.sources_per_block, self.quant_sources_per_block=PositionalArraysFast(self.source_edge, self.pos_s, self.tau, self.h, 
                                                              mesh_3D.h, mesh_3D.cells_x, mesh_3D.cells_y, mesh_3D.cells_z)
        self.uni_s_blocks, self.counts = np.unique(self.s_blocks, return_counts=True)
        return 
    
# =============================================================================
#     def KernelPoint(self,x, neighbourhood, function, D):
#         #Maybe we should put this function independently so it can be jitted
#         """Returns the kernels to multiply the vectors of unknowns q and C_v
#         function K and D are useless"""
#         
#         sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, neighbourhood)]
#         q_array=np.array([])
#         
#         for i in sources:
#             ed=self.source_edge[i]
#             tau=self.tau[ed]
#             a,b= self.pos_s[i]-tau*self.h[ed]/2, self.pos_s[i]+tau*self.h[ed]/2
#             
#             dist_sq=self.R[ed]**2 if self.R[ed]<self.h[ed] else self.h[ed]**2/4+self.R[ed]**2
#             
#             #if (( np.dot(x-a, tau)>-self.R[ed] ) and ( np.dot(x-a, tau)**2<(self.h[ed]+self.R[ed])**2) and ( np.linalg.norm(np.cross(x-a, tau))<self.R[ed])):
#             #if (( np.dot(x-a, tau)>-self.R[ed] ) and ( np.dot(x-b, tau)>(self.R[ed])) and ( np.linalg.norm(np.cross(x-a, tau))<self.R[ed])):
#             if (np.sum((x-self.pos_s[i])**2) < dist_sq):
#                 q=GetSelfInfluence(self.R[ed], self.h[ed], D)
#                 
#             else:
#                 q=GetSourcePotential(a,b,x,D)
#                 ########################
#                 #Coefficients due to geometry are already added!
#                 ########################
#                 #if q>1: pdb.set_trace()
#                 
#             q_array=np.append(q_array, q)
#             
#         #Return q_kernel_data, C_v_kernel_data, the columns for both 
#         return q_array,  sources
# =============================================================================

@njit
def KernelPointFast(x, neighbourhood, s_blocks, source_edge, tau_array, pos_s, h_1D, R,D):
    #Maybe we should put this function independently so it can be jitted
    """Returns the kernels to multiply the vectors of unknowns q and C_v
    to obtain the rapid term"""
    
    sources=np.arange(len(s_blocks))[in1D(s_blocks, neighbourhood)]
    q_array=np.zeros(0, np.float64)
    for i in sources:
        ed=source_edge[i]
        tau=tau_array[ed]
        a,b=pos_s[i]-tau*h_1D[ed]/2, pos_s[i]+tau*h_1D[ed]/2
        
        projection=np.dot(x-a, tau)
        closest_point = a + projection * tau
        distance = np.linalg.norm(x - closest_point) #distance to centerline
        #if x[2]>59 and x[0]>16 : pdb.set_trace()
        #if (( np.dot(x-a, tau)>-self.R[ed] ) and ( np.dot(x-a, tau)**2<(self.h[ed]+self.R[ed])**2) and ( np.linalg.norm(np.cross(x-a, tau))<self.R[ed])):
        #if (( np.dot(x-a, tau)>-self.R[ed] ) and ( np.dot(x-b, tau)>(self.R[ed])) and ( np.linalg.norm(np.cross(x-a, tau))<self.R[ed])):
        
        #if (np.sum((x-pos_s[i])**2) < dist_sq):
        if distance < R[ed] and projection  < h_1D[ed]+R[ed] and projection>-R[ed]:
            q=GetSelfInfluence(R[ed], h_1D[ed], D)
            
        else:
            q=GetSourcePotential(a,b,x,D)
            ########################
            #Coefficients due to geometry are already added!
            ########################
           
        q_array=np.append(q_array, q)
        
    #Return q_kernel_data, C_v_kernel_data, the columns for both 
    return q_array,  sources
    

def is_point_inside_cylinder(point, start_point, end_point, radius):
    # Calculate the direction vector of the cylinder
    direction = end_point - start_point
    
    # Calculate the vector from the start point to the point of interest
    point_vector = point - start_point
    
    # Calculate the projection of point_vector onto the direction vector
    projection = np.dot(point_vector, direction) / np.dot(direction, direction)
    
    # Calculate the closest point on the centerline to the point of interest
    closest_point = start_point + projection * direction
    
    # Calculate the distance between the closest point and the point of interest
    distance = np.linalg.norm(point - closest_point)
    
    # Check if the distance is less than or equal to the radius
    if distance <= radius:
        return True
    else:
        return False
# =============================================================================
#     def KernelIntegralSurface(self, center,normal, neighbourhood,function, K,D,h):
#         """Returns the kernel that multiplied (scalar, dot) by the array of fluxes (q) returns
#         the integral of the rapid term over the surface
#         
#         Main function used to calculate J
#         
#         h must be the disc size of the mesh_3D
#         """
#         
#         sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, neighbourhood)]
#         
#         q_array, C_v_array=np.array([]), np.array([])
#         for i in sources:
#             ed=self.source_edge[i]
#             a,b= self.pos_s[i]-self.tau[ed]*self.h[ed]/2, self.pos_s[i]+self.tau[ed]*self.h[ed]/2
#             q=SimpsonSurface(a,b,function, center,h, normal,D)
# 
#             q_array=np.append(q_array, q)
#         
#         #Return q_kernel_data, C_v_kernel_dat
#         return q_array,  sources
# =============================================================================
@njit
def PositionalArraysFast(source_edge, pos_s, tau, h_1D, h_3D, cells_x, cells_y, cells_z, ):
    """This function is the pre processing step. It is meant to create the s_blocks
    and uni_s_blocks arrays which will be used extensively throughout. s_blocks represents
    the block where each source is located, uni_s_blocks contains all the source blocks
    in a given order that will be respected throughout the resolution
    
        - h_cart is the size of the cartesian mesh"""
        
    # pos_s will dictate the ID of the sources by the order they are kept in it!
    s_blocks = np.zeros(0, dtype=np.int64)
    uni_s_blocks = np.zeros(0, dtype=np.int64)
    for i in range(len(pos_s)):
        ed=source_edge[i] #Current edge (int) the source lies on 
        x_j=pos_s[i] #Center of the cylinder
        
        s_blocks=np.append(s_blocks, GetID(h_3D,cells_x, cells_y, cells_z,x_j))
    sources_per_block=List()   
    quant_sources_per_block=np.zeros(0, dtype=np.int64)
    for k in range(cells_x*cells_y*cells_z):
        arr=np.where(s_blocks==k)[0]
        sources_per_block.append(arr)
        quant_sources_per_block=np.append(quant_sources_per_block,len(arr))
    
    return s_blocks, sources_per_block, quant_sources_per_block
    
@njit
def KernelIntegralSurfaceFast(s_blocks, tau, h_net, pos_s, source_edge,center, normal, neighbourhood, function, D, h_3D):
    """Returns the kernel that multiplied (scalar, dot) by the array of fluxes (q) returns
    the integral of the rapid term over the surface
    
    Main function used to calculate J
    
    h must be the disc size of the mesh_3D
    """
    sources=np.arange(len(s_blocks))[in1D(s_blocks, neighbourhood)]
    #sources=in1D(s_blocks, neighbourhood)
    q_array=np.zeros(len(sources), dtype=np.float64)
    c=0
    for i in sources:
        ed=source_edge[i]
        a,b= pos_s[i]-tau[ed]*h_net[ed]/2, pos_s[i]+tau[ed]*h_net[ed]/2
        q_array[c]=SimpsonSurface(a,b,function, center,h_3D, normal,D)
        c+=1
    return q_array,  sources

@njit
def SimpsonVolume(block_ID, prob_args):
    n, h_3D,cells_x, cells_y, cells_z, pos_cells, s_blocks, source_edge, tau, pos_s, h_1D, R, D=prob_args
    neighbourhood=GetNeighbourhood(n, cells_x, cells_y, cells_z, block_ID)
    integral_kernel, sources=KernelPointFast(pos_cells[block_ID], neighbourhood, 
                                             s_blocks, source_edge, tau, pos_s, h_1D, 
                                             R,D)
    integral_kernel*=64
    h_3D
    points_edge=np.array([[0,h_3D/2,h_3D/2],
                         [0,h_3D/2,-h_3D/2],
                         [0,-h_3D/2,h_3D/2],
                         [0,-h_3D/2,-h_3D/2],
                         
                         [h_3D/2,0,h_3D/2],
                         [h_3D/2,0,-h_3D/2],
                         [-h_3D/2,0,h_3D/2],
                         [-h_3D/2,0,-h_3D/2],
                         
                         [h_3D/2,h_3D/2,0],
                         [h_3D/2,-h_3D/2,0],
                         [-h_3D/2,h_3D/2,0],
                         [-h_3D/2,-h_3D/2,0]], dtype=np.float64)
    
    points_face=np.array([[0,0,h_3D/2],
                         [0,0,-h_3D/2],
                         [0,h_3D/2,0],
                         [0,-h_3D/2,0],
                         [h_3D/2,0,0],
                         [-h_3D/2,0,0]], dtype=np.float64)
    
    points_corners=np.array([[h_3D/2,h_3D/2,h_3D/2],
                             [h_3D/2,-h_3D/2,h_3D/2],
                             [-h_3D/2,h_3D/2,h_3D/2],
                             [-h_3D/2,-h_3D/2,h_3D/2],
                             [h_3D/2,h_3D/2,-h_3D/2],
                             [h_3D/2,-h_3D/2,-h_3D/2],
                             [-h_3D/2,h_3D/2,-h_3D/2],
                             [-h_3D/2,-h_3D/2,-h_3D/2]], dtype=np.float64)

    
    #This only works because it is a Cartesian grid and the normal has to be parallel to one of the axis
    #Other wise we would need another way to calculate this tangential vector
    #tau represents one of the prependicular vectors to normal 
    for i in points_edge:
        pos=pos_cells[block_ID]+i
        #The function returns two kernels that cannot be multiplied 
        kernel,_=KernelPointFast(pos, neighbourhood, 
                                 s_blocks, source_edge, tau, pos_s, h_1D, 
                                 R,D)
        integral_kernel+=kernel*4
        
    for i in points_face:
        pos=pos_cells[block_ID]+i
        #The function returns two kernels that cannot be multiplied 
        kernel,_=KernelPointFast(pos, neighbourhood, 
                                 s_blocks, source_edge, tau, pos_s, h_1D, 
                                 R,D)
        integral_kernel+=kernel*16
    
    for i in points_corners:
        pos=pos_cells[block_ID]+i
        #The function returns two kernels that cannot be multiplied 
        kernel,_=KernelPointFast(pos, neighbourhood, 
                                 s_blocks, source_edge, tau, pos_s, h_1D, 
                                 R,D)
        integral_kernel+=kernel
        
    integral_kernel/=216
    return integral_kernel, sources

def test_kernel_integral():
    
    from Green import unit_test_Simpson
    h_mesh=10 #size of the cube
    
    #One source and one block
    startVertex=np.array([0])
    endVertex=np.array([1])
    vertex_to_edge=np.array([[0]])
    
    L_vessel=1
    
    pos_vertex=np.array([[-L_vessel/2,0,0],[L_vessel/2,0,0]])
    diameters=np.array([0.1]) #on s'en fiche 
    h=np.array([L_vessel/10])
    D=1 
    
    a=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,D)
    
    a.s_blocks=np.zeros(10)
    
    pp=a.KernelIntegralSurface(np.array([0,0,h_mesh/2]), np.array([0,0,1]), np.zeros(10), 'G', np.array([1]),1,h_mesh)
    
    print("If h_mesh is sufficiently greater than L_vessel this should be around -0.198: ", np.sum(pp[0])*h_mesh**2/L_vessel)
    
    unit_test_Simpson(h_mesh, D)  



      
