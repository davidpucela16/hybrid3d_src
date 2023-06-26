#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:24:51 2023

@author: pdavid
"""
import os 
import numpy as np 
import scipy as sp 
from post_processing import ReconstructionCoordinatesFast, GetPlaneReconstructionFast,GetCoordsPlane, GetProbArgs
from neighbourhood import GetNeighbourhood
from mesh_1D import SimpsonVolume
import matplotlib.pyplot as plt
from dask import delayed
from numba import njit
import pdb
from scipy.sparse import csc_matrix

from dask.distributed import Client, LocalCluster

def SplitFile(filename, output_dir_network):
    with open(filename, 'r') as file:
        output_files = []
        current_output = None
        line_counter = 0

        for line in file:
            line_counter += 1

# =============================================================================
#             if line_counter < 25:
#                 continue
# =============================================================================

            if line.startswith('@'):
                if current_output:
                    current_output.close()
                output_filename = f"output_{len(output_files)}.txt"
                output_path = os.path.join(output_dir_network, output_filename)
                current_output = open(output_path, 'w')
                output_files.append(output_path)

            if current_output:
                current_output.write(line)

        if current_output:
            current_output.close()

        return output_files


def SetArtificialBCs(vertex_to_edge, entry_concentration, exit_concentration, init, end):
    """Assembles the BCs_1D array with concentration=entry_concentration for the init vertices and 
    concentration=exit_concentration for exiting vertices
    
    Remember to have preprocessed the init and end arrays for the velocity to always be positive"""
    BCs_1D=np.zeros(2, dtype=np.int64)
    c=0
    for i in vertex_to_edge: #Loop over all the vertices
    #i contains the edges the vertex c is in contact with
        if len(i)==1:
            edge=i[0]
            if np.in1d(c, init):
                BCs_1D=np.vstack((BCs_1D, np.array([c, entry_concentration])))
            else:
                BCs_1D=np.vstack((BCs_1D, np.array([c, exit_concentration])))
        c+=1
    return(BCs_1D)

def ClassifyVertices(vertex_to_edge, init):
    """Classifies each vertex as entering, exiting or bifurcation
    The flow must have already been pre processed so it is always positive, and the direction is given 
    by the edges array"""
    #BCs is a two dimensional array where the first entry is the vertex and the second the value of the Dirichlet BC
    entering=[]
    exiting=[]
    bifurcation=[]
    vertex=0 #counter that indicates which vertex we are on
    for i in vertex_to_edge:
        if len(i)==1: #Boundary condition
            #Mount the boundary conditions here
            if init[i]!=vertex: #Then it must be the end Vertex of the edge 
                exiting.append(vertex)
            else: 
                entering.append(vertex)
        else: #Bifurcation between two or three vessels (or apparently more)
            bifurcation.append(vertex)
        vertex+=1
    return entering, exiting, bifurcation

def get_phi_bar(phi_bar_path, s, q):
    phi_bar_s=sp.sparse.load_npz( phi_bar_path + "/phi_bar_s.npz")
    phi_bar_q=sp.sparse.load_npz( phi_bar_path + "/phi_bar_q.npz")
    
    phi_bar=phi_bar_s.dot(s)+phi_bar_q.dot(q)
    return phi_bar

def Get9Lines(ind_axis, resolution, L_3D, prob):
    
    #The other two axis are:
    others=np.delete(np.array([0,1,2]), ind_axis)
    
    points_a=np.array([1/6,3/6,5/6])*L_3D[others[0]]
    points_b=np.array([1/6,3/6,5/6])*L_3D[others[1]]
    
    crds=np.zeros([9,3,resolution])
    
    indep=np.linspace(0,L_3D[ind_axis], resolution)*0.98+L_3D[ind_axis]*0.01
    
    
    for i in range(3):
        for j in range(3):
            a=np.zeros(resolution)+points_a[i]
            b=np.zeros(resolution)+points_b[j]
            
            crds[j+3*i, ind_axis,:]=indep
            crds[j+3*i, others[0],:]=a
            crds[j+3*i, others[1],:]=b
        
    
    phi = []
    for i in range(9):
        phi.append(ReconstructionCoordinatesFast(crds[i, :, :].T, prob.n, prob.mesh_3D.cells_x, 
                                                          prob.mesh_3D.cells_y,prob.mesh_3D.cells_z, prob.mesh_3D.h,
                                                          prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, 
                                                          prob.mesh_1D.source_edge,prob.mesh_1D.tau, 
                                                          prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.mesh_1D.R, 
                                                          1, prob.s, prob.q))
    
    return(phi,crds, others, points_a, points_b)

class VisualizationTool():
    def __init__(self,prob,perp_axis, i_axis, j_axis, corners_2D, resolution):
        """perp_axis -> axis perpendicular to the cutting plane
        i_axis and j_axis -> the ones we want to appear horizontal and vertical in the plots
        the i_axis also acts as independent variable in the line plots
        
        corners are given as 4 bidimensional arrays in the form of:
            (0,0),(1,0),(0,1),(1,1)"""
        self.resolution=resolution
        self.prob=prob
        self.perp_axis=perp_axis
        self.i_axis=i_axis
        self.j_axis=j_axis
        self.corners_2D=corners_2D
        self.pos_array=np.array([0.2,0.4,0.6,0.8]) #Default values of the third axis
        self.line_data=1
        self.plot_phi_extra=False
        return
    
    def GetPlaneData(self, path):
        prob, resolution=self.prob, self.resolution
        perp_axis, i_axis, j_axis=self.perp_axis,self.i_axis, self.j_axis
        corners_2D=self.corners_2D
        L_3D=prob.mesh_3D.L       
        
        dirs=np.array(["x","y","z"])
        pos_array=self.pos_array

        corners_3D=np.zeros((4,3))
        self.vmax=0
        self.vmin=1
        for x in pos_array*prob.mesh_3D.L[perp_axis]:
            corners_3D[:,perp_axis]=x
            corners_3D[:,i_axis]=corners_2D[:,0]
            corners_3D[:,j_axis]=corners_2D[:,1]
            phi_intra_rec,a,b, phi_extra_rec, crds=GetPlaneReconstructionFast(x, perp_axis, i_axis, j_axis,corners_3D , resolution, prob, prob.Cv, path)
            crds_1D=crds.reshape(resolution, resolution,3)[np.array(pos_array*resolution,dtype=np.int32)]
            
            self.vmax=np.max(np.append(phi_extra_rec, self.vmax))
            self.vmin=np.min(np.append(phi_extra_rec, self.vmin))
            phi_1D=[]
            if self.line_data:
                for i in range(len(pos_array)):
                    phi_1D.append(ReconstructionCoordinatesFast(crds_1D[i], GetProbArgs(prob), prob.s, prob.q))
                np.save(os.path.join(path, "phi_1D_{}={:04g}_{}".format(dirs[perp_axis], int(x), dirs[i_axis])),phi_1D)
                np.save(os.path.join(path, "coordinates_{}={:04g}_{}".format(dirs[perp_axis], int(x), dirs[i_axis])),crds_1D)

    def GetVolumeData(self, chunks, process, perp_axis_res, path, shrink_factor):
        os.makedirs(path, exist_ok=True)
        self.line_data=0
        L_perp=self.prob.mesh_3D.L[self.perp_axis]
        perp_disc=(np.linspace(0,L_perp*(1-1/perp_axis_res), perp_axis_res)+1/(2*perp_axis_res))*shrink_factor + L_perp*(1-shrink_factor)/2
        chunk_size=int(np.ceil(perp_axis_res/chunks))
        initial_chunk=process*chunk_size
        disc_local=perp_disc[initial_chunk:initial_chunk+chunk_size]
        self.pos_array=disc_local/L_perp
        self.GetPlaneData(path)
        
        self.bounding_box=np.zeros((3,2))
        self.bounding_box[self.perp_axis]=perp_disc[[0,-1]]
        self.bounding_box[self.i_axis]=perp_disc[[0,-1]]
        self.bounding_box[self.j_axis]=perp_disc[[0,-1]]
    
    def PlotData(self, path):
        dirs=np.array(["x","y","z"])
        perp_axis, i_axis, j_axis= self.perp_axis, self.i_axis, self.j_axis
        prob=self.prob
        phi=[]
        coordinates=[]
        phi_extra=[]
        phi_1D_full=[]
        #pdb.set_trace()
        for i in self.pos_array:
            plane_coord=i*prob.mesh_3D.L[perp_axis]
            #We could put phi_extra, or phi_intra, or mask 
            phi.append(np.load(os.path.join(path, 'phi_intra_{}={:04g}_{}_{}.npy'.format(dirs[perp_axis], int(plane_coord), dirs[i_axis], dirs[j_axis]))))
            
            phi_extra.append(np.load(os.path.join(path, 'phi_extra_{}={:04g}_{}_{}.npy'.format(dirs[perp_axis], int(plane_coord), dirs[i_axis], dirs[j_axis]))))
            phi_1D_full.append(np.load(os.path.join(path, 'phi_1D_{}={:04g}_{}.npy'.format(dirs[perp_axis], int(plane_coord), dirs[i_axis]))))
            coordinates.append(np.load(os.path.join(path, "coordinates_{}={:04g}_{}.npy".format(dirs[perp_axis], int(plane_coord), dirs[i_axis]))))
        # Generate example matrices
        # Define the minimum and maximum values for the color scale
        vmin=self.vmin
        vmax=self.vmax
        
        ylim=(np.min([0,np.min(phi_1D_full)]),np.max(phi_1D_full))
        #################################################################
        #   In case I wanna plot phi_extra
        ##################################################################
        if self.plot_phi_extra:
            self.phi_intra=phi.copy()
            phi=phi_extra
            
        # Plot the matrices using imshow
        fig, axs = plt.subplots(2, 4, figsize=(30,16))
        im1 = axs[0, 0].imshow(phi[0], cmap='jet', vmin=vmin, vmax=vmax)
        axs[0, 0].set_xlabel(dirs[i_axis])
        axs[0, 0].set_ylabel(dirs[j_axis])
        axs[0, 1].plot(coordinates[0][0,:,i_axis],phi_1D_full[0].T, 
                       label=[dirs[j_axis] + '={:.1f}'.format(self.pos_array[0]*prob.mesh_3D.L[j_axis]),
                              dirs[j_axis] + '={:.1f}'.format(self.pos_array[1]*prob.mesh_3D.L[j_axis]),
                              dirs[j_axis] + '={:.1f}'.format(self.pos_array[2]*prob.mesh_3D.L[j_axis]),
                              dirs[j_axis] + '={:.1f}'.format(self.pos_array[3]*prob.mesh_3D.L[j_axis])])
        axs[0, 1].set_xlabel(dirs[i_axis])
        axs[0, 1].set_ylim(ylim)
        axs[0, 1].legend()

        im2 = axs[0, 2].imshow(phi[1], cmap='jet', vmin=vmin, vmax=vmax)
        axs[0,2].set_xlabel(dirs[i_axis])
        axs[0, 2].set_ylabel(dirs[j_axis])
        axs[0, 3].plot(coordinates[1][0,:,i_axis],phi_1D_full[1].T)
        axs[0, 3].set_xlabel(dirs[i_axis])
        axs[0, 3].set_ylim(ylim)
        
        im3 = axs[1, 0].imshow(phi[2], cmap='jet', vmin=vmin, vmax=vmax)
        axs[1, 0].set_xlabel(dirs[i_axis])
        axs[1, 0].set_ylabel(dirs[j_axis])
        axs[1, 1].plot(coordinates[0][0,:,i_axis],phi_1D_full[2].T)
        axs[1, 1].set_xlabel(dirs[i_axis])
        axs[1, 1].set_ylim(ylim)

        im4 = axs[1, 2].imshow(phi[3], cmap='jet', vmin=vmin, vmax=vmax)
        axs[1, 2].set_xlabel(dirs[i_axis])
        axs[1, 2].set_ylabel(dirs[j_axis])
        axs[1, 3].plot(coordinates[0][0,:,i_axis],phi_1D_full[3].T)
        axs[1,3].set_xlabel(dirs[i_axis])
        axs[1,3].set_ylim(ylim)
        
        L_3D=prob.mesh_3D.L    
        # Set titles for the subplots
        axs[0, 0].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[0]*L_3D[0]))
        axs[0, 1].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[0]*L_3D[0]))
        axs[0, 2].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[1]*L_3D[0]))
        axs[0, 3].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[1]*L_3D[0]))
        axs[1, 0].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[2]*L_3D[0]))
        axs[1, 1].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[2]*L_3D[0]))
        axs[1, 2].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[3]*L_3D[0]))
        axs[1, 3].set_title(dirs[perp_axis] + '={:.1f}'.format(self.pos_array[3]*L_3D[0]))
        
        
        
        # Adjust spacing between subplots
        fig.tight_layout()

        # Move the colorbar to the right of the subplots
        cbar = fig.colorbar(im1, ax=axs, orientation='vertical', shrink=0.8)
        cbar_ax = cbar.ax
        cbar_ax.set_position([0.83, 0.15, 0.03, 0.7])  # Adjust the position as needed

        # Show the plot
        plt.savefig(os.path.join(path, 'perp_axis_{}_res_{}.svg').format(perp_axis, self.resolution))
        plt.show()
        return
    
    def Plot9Lines(self):
        phi,crds, others, points_a, points_b=self.Get9Lines(self.perp_axis, self.resolution, self.L_3D, self.prob)
        for k in range(9):
            plt.plot(phi[k], label=str(np.array(["x","y","z"])[others]) + "={:.1f}, {:.1f}".format(points_a[k//3], points_b[k%3]))
        plt.xlabel(np.array(["x","y","z"])[i])
        plt.legend()
        plt.show()
        
    def Get9Lines(self,perp_axis, resolution, L_3D, prob):
        
        #The other two axis are:
        others=np.delete(np.array([0,1,2]), perp_axis)
        
        points_a=np.array([1/6,3/6,5/6])*L_3D[others[0]]
        points_b=np.array([1/6,3/6,5/6])*L_3D[others[1]]
        
        crds=np.zeros([9,3,resolution])
        
        indep=np.linspace(0,L_3D[perp_axis], resolution)*0.98+L_3D[perp_axis]*0.01
        
        
        for i in range(3):
            for j in range(3):
                a=np.zeros(resolution)+points_a[i]
                b=np.zeros(resolution)+points_b[j]
                
                crds[j+3*i, perp_axis,:]=indep
                crds[j+3*i, others[0],:]=a
                crds[j+3*i, others[1],:]=b
            
        
        phi = []
        for i in range(9):
            phi.append(ReconstructionCoordinatesFast(crds[i, :, :].T, prob.n, prob.mesh_3D.cells_x, 
                                                              prob.mesh_3D.cells_y,prob.mesh_3D.cells_z, prob.mesh_3D.h,
                                                              prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, 
                                                              prob.mesh_1D.source_edge,prob.mesh_1D.tau, 
                                                              prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.mesh_1D.R, 
                                                              1, prob.s, prob.q))
        
        return(phi,crds, others, points_a, points_b)

def Get1DCOMSOL(file_path):
    # Open the file for reading
    with open(file_path, "r") as file:
        # Skip the header lines
        for _ in range(10):
            next(file)
    
        # Read the remaining lines and extract the values
        values = []
        for line in file:
            # Split the line into columns using space as the separator
            columns = line.split()
    
            # Extract the values from the first two columns
            value1 = float(columns[0])
            value2 = float(columns[1])
    
            # Add the values to the list
            values.append((value1, value2))

    phi_bar_COMSOL = np.array(values)[:, 1]
    x_COMSOL = np.array(values)[:, 0]
    return(phi_bar_COMSOL, x_COMSOL)   

@njit
def GetCoarsePhi( q, Cv, s, prob_args):
    """Provides the average concentration per FV cell on the coarse mesh.
    It also returns the arrays to construct the average of the rapid term per cell"""
    phi=np.zeros(len(s), dtype=np.float64)
    kernel_array=np.zeros(0, dtype=np.float64)
    sources_array=np.zeros(0, dtype=np.int64)
    for i in range(len(s)):
        #print("FV Block: ", i)
        kernel_q,sources=SimpsonVolume(i, prob_args)
        phi[i]=kernel_q.dot(q[sources]) + s[i]
        
        kernel_array=np.concatenate((kernel_array, kernel_q))
        sources_array=np.concatenate((sources_array, sources))
    return phi, kernel_array, sources_array

@njit
def GetCoarsekernels(prob_args, path):
    kernel_array=np.zeros(0, dtype=np.float64)
    sources_array=np.zeros(0, dtype=np.int64)
    row_array=np.zeros(0, dtype=np.int64)
    for i in range(len(prob_args[5])):
        #print("FV Block: ", i)
        kernel_q,sources=SimpsonVolume(i, prob_args)
        kernel_array=np.concatenate((kernel_array, kernel_q))
        sources_array=np.concatenate((sources_array, sources))
        row_array=np.concatenate((row_array, np.zeros(len(sources), dtype=np.int64) + i))
    return kernel_array, row_array,sources_array

def GetInitialGuess(labels, prob):
    """Not useful for the moment"""
    label_per_source=np.repeat(labels, prob.mesh_1D.cells)
    Cv=np.zeros(prob.S)+0.5
    Cv[np.where(label_per_source==0)[0]]=1
    Cv[np.where(label_per_source==1)[0]]=0.4
    K_per_source=np.repeat(prob.K, prob.mesh_1D.cells)
    q=K_per_source*Cv*0.5
    prob_args=(prob.n, prob.mesh_3D.h, prob.mesh_3D.cells_x,  prob.mesh_3D.cells_y,  
               prob.mesh_3D.cells_z,  prob.mesh_3D.pos_cells,  prob.mesh_1D.s_blocks, 
               prob.mesh_1D.source_edge, prob.mesh_1D.tau, prob.mesh_1D.pos_s, 
               prob.mesh_1D.h, prob.mesh_1D.R, prob.D)
    neg_s,_,_=GetCoarsePhi(q, Cv, np.zeros(prob.F), prob_args)
    
    return -neg_s+0.9, q, Cv

############################################################################
#   Some useful data modification for the network, Post processing for the 1D results
############################################################################

def LabelVertexEdge(vertex_to_edge, init):
    """Assigns a label to each vertex and to each edge
        - 0 if normal edge
        - 1 if entering boundary edge
        - 2 if exiting boundary edge"""
    label_vertex=np.zeros(len(vertex_to_edge))
    label_edge=np.zeros(len(init))
    c=0
    for i in vertex_to_edge:
        if len(i)==1:
            ed=i[0]
            if c in init:
                #entering
                label_vertex[c]=1
                label_edge[ed]=1
            else:
                #exiting
                label_vertex[c]=2
                label_edge[ed]=2
        c+=1
    return(label_vertex, label_edge)

@njit
def GetSingleEdgeSources(cells_per_segment,  edge):
    """Returns the IDs of all the sources belonging to the edge in array form"""
    return np.arange(np.sum(cells_per_segment[:edge]),np.sum(cells_per_segment[:edge+1]))

def GetEdgesConcentration(cells_per_segment, prop):
    """Provides the average concentration per edge"""
    avg_edge_conc=np.zeros(len(cells_per_segment))
    for i in range(len(cells_per_segment)): #goes through every single source
        avg_edge_conc[i]=np.average(prop[GetSingleEdgeSources(cells_per_segment, i)])
    return avg_edge_conc
    

def GetBoundarySources(pos_s, pos_vertex, vertex_label, entry_exit):
    """Provides the IDs of the sources lying on the boundary"""
    if entry_exit=="entering":
        print("Entering segments")
        pos=pos_vertex[np.where(vertex_label==1)[0]]
    elif entry_exit=="exiting":
        print("Exiting segments")
        pos=pos_vertex[np.where(vertex_label==2)[0]]
    else:
        print("Non boundary segments")
        pos=pos_vertex[np.where(vertex_label==0)[0]]
        
    IDs=np.zeros(0, dtype=np.int64)
    for i in pos:
        source=np.argmin(np.sum((pos_s-i)**2, axis=1))
        if source>len(pos_s): pdb.set_trace()
        IDs=np.append(IDs, source)
    return IDs
    
    
def GetPointsAM(edges, pos_vertex, pos_s, cells_1D):
    """Returns the coordinates of all the points (including the vertices)
    as required by the .am format"""
    points_array=np.zeros(((0,3)), dtype=np.float64)
    ed=-1
    for i in edges:
        ed+=1
        init_pos=np.sum(cells_1D[:ed])
        end_pos=np.sum(cells_1D[:ed+1])
        local_arr=np.vstack((pos_vertex[i[0]], pos_s[init_pos:end_pos],pos_vertex[i[1]]))
        points_array=np.vstack((points_array, local_arr))
    return points_array

def GetConcentrationAM(edges, property_vertex, property_point, cells_1D):
    """Returns the concentration (or any other property) per source including the 
    vertices as required by the .am format"""
    points_array=np.zeros(0, dtype=np.float64)
    ed=-1
    for i in edges:
        ed+=1
        init_pos=np.sum(cells_1D[:ed])
        end_pos=np.sum(cells_1D[:ed+1])
        local_arr=np.append(property_vertex[i[0]],np.append(property_point[init_pos:end_pos],property_vertex[i[1]]))
        points_array=np.concatenate((points_array, local_arr))
    return points_array

def GetConcentrationVertices(vertex_to_edge, startVertex, cells_per_segment, property_sources):
    """Protivdes the value of a scalar field in the network at the vertices"""
    value_array=np.zeros(0)
    for i in range(len(vertex_to_edge)):
        for ed in vertex_to_edge[i]:
            value=0
            sources=GetSingleEdgeSources(cells_per_segment,  ed)
            if startVertex[ed]==i:
                value+=property_sources[sources[0]]
            else:
                value+=property_sources[sources[-1]]
        value/=len(vertex_to_edge[i])
        value_array=np.append(value_array, value)
    return value_array

    
    
    