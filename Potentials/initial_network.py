#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:20:01 2023

@author: pdavid
"""
import numpy as np
import matplotlib.pyplot as plt
simplissime=True
nDoFPerCyl=10

if simplissime:
    #DEFINE THE NETWORK (could be read from h5 file)
    diameters=np.array([1.14149,1.03756,1.21498,1.53213,1.27028])*1e-6
    endVertex=np.array([1,2,3,4,5], dtype=int)
    startVertex=np.array([0,1,1,3,3], dtype=int)
    nbPointsPerEdge=np.array([27,32,7,30,28])*nDoFPerCyl
    
    #Necessary to introduce the bifurcation manually 
    bif_list=[[1,3],[[0,1,2],[2,3,4]]] #the number of the vertex and the edges it touches
    
    #BOUNDARY CONDITIONS
    uid_list=np.array([0,2,4,5], dtype=int)
    value_list=np.array([10000,2000,2000,2000])
    value_list_conc=np.array([1,0,0,0])
    
    
    xVertex=np.array([17,9,25,9,21,5])
    yVertex=np.array([2,17,23,23,46,50])
    
else:
    #DEFINE THE NETWORK (could be read from h5 file)
    diameters=np.array([20,10,10])*1e-3
    endVertex=np.array([1,2,3], dtype=int)
    startVertex=np.array([0,1,1], dtype=int)
    nbPointsPerEdge=np.array([20,20,20])*5
    #Necessary to introduce the bifurcation manually 
    bif_list=[[1],[[0,1,2]]] #the number of the vertex and the edges it touches
    
    #BOUNDARY CONDITIONS
    uid_list=np.array([0,2,3], dtype=int)
    value_list=np.array([1000,4000,6000])
    value_list_conc=np.array([0,0.5,1])    

#It is important to know how many vertices there are in the simulation
vertices=np.unique(np.concatenate([endVertex, startVertex]))

L=np.sqrt((xVertex[startVertex]-xVertex[endVertex])**2+(yVertex[startVertex]-yVertex[endVertex])**2)
L=L*1e-3
h=L/nbPointsPerEdge


def plot_network(startVertex, endVertex, xVertex, yVertex, title):
    """Only purpose is to visualize the network if it is 2D"""
    networkx=np.array([])
    networky=np.array([])
    for i in range(len(startVertex)):
        networkx=np.append(networkx,xVertex[startVertex[i]])
        networkx=np.append(networkx,xVertex[endVertex[i]])
        
        networky=np.append(networky,yVertex[startVertex[i]])
        networky=np.append(networky,yVertex[endVertex[i]])
    plt.plot(networkx, networky)
    plt.ylabel("y ($\mu m$)")
    plt.xlabel("x ($\mu m$)")
    plt.title(title)
    
    
plot_network(startVertex, endVertex, xVertex, yVertex, 'title')



def PreProcessingNetwork(U, init, end, pos_vertex):
    vertex_to_edge=[]
    
    for i in range(len(pos_vertex)):
        if U[i]<0:
            temp=init[i].copy()
            init[i]=end[i].copy()
            end[i]=temp
    
    for i in range(len(pos_vertex)):
            
        a=np.arange(len(init))[init==i] #edges for which this vertex is the initial 
        b=np.arange(len(end))[end==i]  #edges for which this vertex is the end
        
        vertex_to_edge+=[a.tolist()+(-b).tolist()]
        
    return init, end, vertex_to_edge
        