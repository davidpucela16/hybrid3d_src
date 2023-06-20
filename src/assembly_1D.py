#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:09:23 2023

@author: pdavid
"""
import numpy as np 
import scipy as sp 
import pdb
from small_functions import AppendSparse
from scipy.sparse import csc_matrix
from numba import njit

from scipy.sparse.linalg import spsolve as dir_solve


def AssembleVertexToEdge(vertices, edges):
    """I will use np where because I don't know how to optimize it other wise"""
    vertex_to_edge=[]
    for i in range(len(vertices)):
        a=np.where(edges[:,0]==i)[0]
        b=np.where(edges[:,1]==i)[0]
        temp=list(np.concatenate((a,b)))
        vertex_to_edge.append(temp)
    return vertex_to_edge

@njit
def PreProcessingNetwork(init, end, flow_rate):
    """Pre processes the edges so blood flows always from init to end"""
    for i in np.where(flow_rate<0)[0]:
        temp=init[i]
        init[i]=end[i]
        end[i]=temp
    return init, end

def SetArtificialBCs(vertex_to_edge, entry_concentration, exit_concentration, init, end):
    """Assembles the BCs_1D array with concentration=entry_concentration for the init vertices and 
    concentration=exit_concentration for exiting vertices
    
    Remember to have preprocessed the init and end arrays for the velocity to always be positive"""
    BCs_1D=np.zeros(2, dtype=np.int64)
    c=0
    for i in vertex_to_edge: #Loop over all the vertices
    #i contains the edges the vertex c is in contact with
        if len(i)==1:
            vertex=i[0]
            if np.in1d(i, init):
                BCs_1D=np.vstack((BCs_1D, np.array([c, entry_concentration])))
            else:
                BCs_1D=np.vstack((BCs_1D, np.array([c, exit_concentration])))
        c+=1
    return(BCs_1D)

def CheckLocalConservativenessFlowRate(init, end, vertex_to_edge, flow_rate):
    """Checks if mass is conserved at the bifurcations"""
    vertex=0
    non_conservation=0
    total_flux=0
    print("Running conservative check on bifurcations")
    for i in vertex_to_edge:
        if len(i)>2:
            a=np.zeros(len(i)) #to store whether the edges are entering or exiting
            c=0
            for j in i: #Goes through each edge of the bifurcation
                a[c]=1 if vertex==init[j] else -1  #Vessel exiting
                c+=1
                total_flux+=np.average(np.abs(flow_rate[i]))
            #print("Conservative Check", np.dot(flow_rate[i], a))
            non_conservation+=np.abs(np.dot(flow_rate[i], a))
        vertex+=1
    print("normalized lack of mass conservation: ", non_conservation/total_flux)
    return
        

# =============================================================================
# def CheckLocalConservativenessVelocity(init, end, vertex_to_edge, flow_rate, R):
#     """Checks if mass is conserved at the bifurcations"""
#     vertex=0
#     for i in vertex_to_edge:
#         
#         if len(i)>2:
#             a=np.zeros(len(i)) #to store whether the edges are entering or exiting
#             c=0
#             for j in i: #Goes through each edge of the bifurcation
#                 a[c]=1 if vertex==init[j] else -1  #Vessel exiting
#                 c+=1
#                 
#             print(np.dot(flow_rate[i], a))
#         vertex+=1
#     return
# =============================================================================

    


def FullAdvectionDiffusion1D(U, D, h, cells, init, vertex_to_edge, R, BCs, *zero_flux):
    sparse_arrs=AssemblyTransport1DFast(U, D, h, cells )
    sparse_arrs, ind_array, DoF_list=AssemblyVertices(U, D, h, cells, sparse_arrs, vertex_to_edge, R, init, BCs,*zero_flux)
    
    return sparse_arrs, ind_array, DoF_list

@njit
def AssemblyTransport1D(U, D, h, cells):
    """Assembles the linear matrix for convection-dispersion for a network only for the inner DoFs"""
    data=np.zeros(0, dtype=np.float64) 
    row=np.zeros(0, dtype=np.int64)
    col=np.zeros(0, dtype=np.int64)
    for ed in range(len(cells)):
        print("Assembling inner cells intra transport, edge: ", ed)
        initial=np.sum(cells[:ed])
        for i in initial+np.arange(cells[ed]-2)+1:
            data=np.concatenate((data, np.array([-U[ed]-D/h[ed],U[ed]+ 2*D/h[ed], -D/h[ed]], dtype=np.float64)))
            row=np.concatenate((row, np.array([i,i,i])))
            col=np.concatenate((col, np.array([i-1,i,i+1], dtype=np.int64)))

    return data, row, col       
@njit
def AssemblyTransport1DFast(U, D, h, cells):
    """Assembles the linear matrix for convection-dispersion for a network only for the inner DoFs"""
    data=np.zeros(np.sum(cells)*3, dtype=np.float64) #times 3 because the stensil has three entries
    row=np.zeros(np.sum(cells)*3, dtype=np.int64)    #They are over - dimensioned
    col=np.zeros(np.sum(cells)*3, dtype=np.int64)
    counter=0
    for ed in range(len(cells)):
        initial=np.sum(cells[:ed])
        for i in initial+np.arange(cells[ed]-2)+1:
            data[counter:counter+3]=np.array([-U[ed]-D/h[ed],U[ed]+ 2*D/h[ed], -D/h[ed]], dtype=np.float64)
            row[counter:counter+3]=np.array([i,i,i])
            col[counter:counter+3]=np.array([i-1,i,i+1], dtype=np.int64)
            counter+=3
    return data, row, col   

       
def AssemblyVertices(U, D, h, cells, sparse_arrs, vertex_to_edge, R, init, BCs, *exit_BC):
    """This function assembles the 1D transport equation for the bifurcations"""
    #BCs is a two dimensional array where the first entry is the vertex and the second the value of the Dirichlet BC
    
    #We create here the independent array to store the BCs:
    ind_array=np.zeros(np.sum(cells))
    DoF_list=[] #List that stores the DoF that are in contact with this bifurcation
    if exit_BC: exit_BC=exit_BC[0]
    vertex=0 #counter that indicates which vertex we are on
    for i in vertex_to_edge:
        if len(i)==1: #Boundary condition, only one edge attached to this vertex
            #Mount the boundary conditions here
            ed=i[0]
            if init[ed]!=vertex: #Then it must be the end Vertex of the edge 
                if exit_BC=="zero_flux":
                    current_DoF=np.sum(cells[:ed])+cells[ed]-1 #End vertex 
                    kk=-1
                    sparse_arrs=AppendSparse(sparse_arrs, np.array([-U[ed]-D/h[ed],U[ed]+D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF+kk,current_DoF]))
                    #print("no flux BC on intravascular")
                else:
                    print("Dirichlet BC on the intravascular system")
                    current_DoF=np.sum(cells[:ed])+cells[ed]-1 #End vertex 
                    kk=-1
                    sparse_arrs=AppendSparse(sparse_arrs, np.array([-U[ed]-D/h[ed],U[ed]+3*D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF+kk,current_DoF]))
                    value_Dirichlet=-2*D/h[ed]
                    ind_array[current_DoF]=BCs[np.where(BCs[:,0]==vertex)[0][0],1]*value_Dirichlet #assigns the value of the BC multiplied by the factor 
                   
            else: 
                #print("Initial vertex!!")
                current_DoF=np.sum(cells[:ed]) #initial vertex
                kk=1
# =============================================================================
#                 sparse_arrs=AppendSparse(sparse_arrs, np.array([-D/h[ed]-U[ed],U[ed]+3*D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF+kk,current_DoF]))
#                 value_Dirichlet=-2*D/h[ed]
# =============================================================================
                sparse_arrs=AppendSparse(sparse_arrs, np.array([-D/h[ed],U[ed]+3*D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF+kk,current_DoF]))
                value_Dirichlet=-2*D/h[ed]-U[ed]
                #pdb.set_trace()
                ind_array[current_DoF]=BCs[np.where(BCs[:,0]==vertex)[0][0],1]*value_Dirichlet #assigns the value of the BC multiplied by the factor 
                
                
        else: #Bifurcation between two or three vessels (or apparently more)
            #pdb.set_trace()
            den=0
            num=np.zeros(len(i))
            
            exiting=np.array([], dtype=int) #The edges for which the bifurcation is the initial DoF
            entering=np.array([], dtype=int)   #The edges for which the bifurcation is the final DoF
            
            #Loop to calculate the coefficients of gamma
            c=0
            DoF=np.zeros(len(i), dtype=int) #To store the actual DoF we are working on 
            #We figure out which edges are exiting and which are entering so we
            #can know which have a advective term on the numerator
            for ed in i: #Goes through each of the vessel in the bifurcation
                den+=2*D*np.pi*R[ed]**2/h[ed] #We always add the diffusive term 
                num[c]=2*D*np.pi*R[ed]**2/h[ed] #Same in the numerator
                if init[ed]==vertex: #The bifurcation is the initial vertex/exiting vessel
                    den+=U[ed]*np.pi*R[ed]**2
                    exiting=np.append(exiting, i[c])
                    DoF[c]=np.sum(cells[:ed])
                else:
                    num[c]+=U[ed]*np.pi*R[ed]**2
                    entering=np.append(entering, i[c])
                    DoF[c]=np.sum(cells[:ed])+cells[ed]-1
                
                c+=1
            
            gamma=num/den
            DoF=DoF.astype(int)
            DoF_list+=list(DoF)
            if np.sum(gamma)!=1: 
                print("Error in gamma estimation ", np.sum(gamma))
            for ed in exiting: #Exiting vessels
                local_ed=np.where(i==ed)[0][0] #position of the edge in list i
            
                current_gamma=gamma[local_ed]
                current_DoF=DoF[local_ed]
                DoF_w=np.delete(DoF, local_ed) #The other two DoFs
                #Exiting flux to the normal neighbouring cylinder 
                sparse_arrs =AppendSparse(sparse_arrs, np.array([D/h[ed]+U[ed], -D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF,current_DoF+1])) 
                #Bifurcation Flux
                sparse_arrs =AppendSparse(sparse_arrs, np.array([2*D/h[ed]*(1-current_gamma)-U[ed]*current_gamma]), np.array([current_DoF]), np.array([current_DoF]))
                
                for j in DoF_w:
                    vessel=np.where(DoF==j)[0][0] #same as local ed before but I wanted to give a different name. It is the local position of the vessel within the bifurcation (i)
                    sparse_arrs =AppendSparse(sparse_arrs, np.array([-2*D/h[ed]*gamma[vessel]-U[ed]*gamma[vessel]]), np.array([current_DoF]), np.array([j]))
                
            for ed in entering:
                local_ed=np.where(i==ed)[0][0]
                
                current_gamma=gamma[local_ed]
                current_DoF=DoF[local_ed]
                #Normal Cylinder
                sparse_arrs =AppendSparse(sparse_arrs, np.array([-U[ed]-D/h[ed],D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF-1,current_DoF]))
                DoF_w=np.delete(DoF, local_ed) #The other two DoFs
                
                #Bifurcation Flux
                sparse_arrs =AppendSparse(sparse_arrs, np.array([2*D/h[ed]*(1-current_gamma)+U[ed]]), np.array([current_DoF]), np.array([current_DoF]))
                
                for j in DoF_w:
                    vessel=np.where(DoF==j)[0][0]
                    sparse_arrs =AppendSparse(sparse_arrs, np.array([-2*D/h[ed]*gamma[vessel]]), np.array([current_DoF]), np.array([j]))
            
        vertex+=1
        
    return sparse_arrs, ind_array, DoF_list



#analytical=(np.exp(-Pe*x)-np.exp(-Pe))/(1-np.exp(-Pe))


class flow():
    """This class acts as a flow solver, if the veocities (or flow) are imported from another simulation
    this class is not neccesary"""    
    def __init__(self, uid_list, value_list, L, diameters, startVertex, endVertex):
        self.bc_uid=uid_list
        self.bc_value=value_list
        self.L=L
        self.d=diameters
        self.viscosity=0.0012
        self.start=startVertex
        self.end=endVertex
        self.n_vertices=np.max(np.array([np.max(self.start)+1, np.max(self.end)+1]))
        
        
    def solver(self):
        A=np.zeros([self.n_vertices,self.n_vertices])
        P=np.zeros([self.n_vertices])
        for i in range(len(self.start)): #Loop that goes through each edge assembling the pressure matrix
        
            if self.start[i] not in self.bc_uid:
                A[self.start[i],self.start[i]]-=self.d[i]**4/self.L[i]
                A[self.start[i],self.end[i]]+=self.d[i]**4/self.L[i]
            if self.end[i] not in self.bc_uid:
                A[self.end[i],self.end[i]]-=self.d[i]**4/self.L[i]
                A[self.end[i],self.start[i]]+=self.d[i]**4/self.L[i]
        A[self.bc_uid,self.bc_uid]=1
        P[self.bc_uid]=self.bc_value
        
        self.A=A
        self.P=P
        
        return(A)
    
    def get_U(self):
        """Computes and returns the speed from the pressure values that have been previously computed"""
        pressures=np.linalg.solve(self.solver(), self.P)
        U=np.array([])
        for i in range(len(self.start)):
            vel=self.d[i]**2*(pressures[self.start[i]]-pressures[self.end[i]])/(32*self.viscosity*self.L[i])
            U=np.append(U,vel)
        self.P=pressures
        return(U)

   
    
