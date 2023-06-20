#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:51:31 2023

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from scipy import integrate
import pdb


def Gjerde(x,a,b,R):
    """Returns the average value of the integral without the coefficient i.e. to the result one would have to multiply
    by the surface of the open cylinder (2 \pi R_j L_j)/(4*pi*D) to obtain a proper single layer potential
    
    DONT FORGET THE DIFUSSION COEFFICIENT"""
    ra=np.linalg.norm(x-a)
    rb=np.linalg.norm(x-b)
    
    L=np.linalg.norm(b-a)
    tau=(b-a)/L
    
    log=np.log((np.max([ra, rb]) + L/2 + np.abs(np.dot((a+b)/2-x,tau)))/(np.min([ra, rb]) - L/2 + np.abs(np.dot((a+b)/2-x,tau))))
    #log=np.log((rb+L+np.dot(tau, a-x))/(ra+np.dot(tau, a-x)))
    return log*R/2

def Fleischman(inc_s, R):
    x_j=np.array([0,0,0])
    x_i=np.array([inc_s, R, 0])
    return (np.linalg.norm(x_j-x_i)**-1*R/2)



#%%
class Classic():
    def __init__(self, L, R):
        self.L = L
        self.R = R

    # The first two functions are to calculate the cross-influence double layer
    # coefficient. Firstly, brute force, secondly, scipy integration

    def H_ij_analytical(self, inc_s):
        R = self.R
        #H = lambda theta: R*(1-np.cos(theta))*(inc_s**2 + (2*R*np.sin(theta/2))**2)**-1.5
        H = lambda theta: -R*(1-np.cos(theta))*(inc_s**2 + (2*R*np.sin(theta/2))**2)**-1.5
        self.H = H
        self.integral = integrate.quad(H, 0, 2*np.pi)
        return(self.integral[0]*R/(np.pi*4))

    def H_ij_numerical(self, inc_s, cells):
        R = self.R
        angle = np.linspace(0, 2*np.pi*(1-1/cells), cells)
        self.angle = angle
        x = np.array([0, R, 0])
        normal = np.array([0, 1, 0])

        integrand = np.array([])
        for theta in angle:
            x_star = np.array([inc_s, R*np.cos(theta), R*np.sin(theta)])
            d = np.linalg.norm(x-x_star)

            integrand = np.append(integrand, np.dot((x-x_star), normal)/d**3)

        self.integrand = integrand

        return(np.sum(self.integrand)*self.R/2/cells)

    # Now the functions for the single layer potential
    def G_ij_analytical(self, inc_s):
        """Calculates the single layer potential of a circumference of the cylinder
        over anoter point on the cylinder and outside of the circumference"""
        R = self.R
        G= lambda theta: (inc_s**2 + (2*R*np.sin(theta/2))**2)**-0.5
        self.G = G
        return(integrate.quad(G, 0, np.pi*2)[0]*R/(np.pi*4))

    def G_ij_elliptical(self, inc_s):
        R = self.R
        k = (inc_s**2/(4*R**2)+1)**-0.5

        return(sps.ellipk(k**2)*k/(2*np.pi))

    def G_ij_numerical(self, inc_s, cells):
        R = self.R
        angle = np.linspace(0, 2*np.pi*(1-1/cells), cells)
        self.angle = angle
        x = np.array([0, R, 0], dtype=np.float64)
        normal = np.array([0, 1, 0], dtype=np.float64)

        integrand = np.array([])
        for theta in angle:
            x_star = np.array([inc_s, R*np.cos(theta), R *
                              np.sin(theta)], dtype=np.float64)
            d = np.linalg.norm(x-x_star)

            integrand = np.append(integrand, 1/d)

        self.integrand = integrand
        return(np.sum(self.integrand)*self.R/2/cells)
    
    
    
    #Now the singular integrals i.e. the self-influence coefficients
    #Single layer
    def G_ii_non_singular(self,inc_s, phi):
        """Calculates the non singular part of the self coefficient"""
        R=self.R
        G = lambda theta :(inc_s**2 + 4*R**2*np.sin(theta/2)**2)**-0.5
        return(integrate.quad(G,phi/2,np.pi*2-phi/2)[0]*R/(4*np.pi))
    
    def G_ii_analytical(self,h):
        """Evaluation of the elliptic integral when the singularity is included """
        phi=np.arcsin(h/(2*self.R))
        self.phi=phi
        return(self.G_ii_non_singular(0, phi))
    
    #Double layer
    def H_ii_non_singular(self,h):
        """Calculates the non singular part of the double layer self coefficient"""
        phi=np.arcsin(h/(2*self.R))
        R=self.R

        #H = lambda theta :R*(1-np.cos(theta))*(4*R**2*np.sin(theta/2)**2)**-1.5
        H = lambda theta : -R*(1-np.cos(theta))*(4*R**2*np.sin(theta/2)**2)**-1.5
        return(integrate.quad(H,phi/2,np.pi*2-phi/2)[0]*R/(4*np.pi))
    
    def H_ii_numerical(self, cells):
        R = self.R
        angle = np.linspace(0, 2*np.pi*(1-1/cells), cells)
        self.angle = angle
        x = np.array([0, R, 0], dtype=np.float64)
        normal = np.array([0, 1, 0], dtype=np.float64)

        integrand = np.array([])
        for theta in angle[1:]:
            x_star = np.array([0, R*np.cos(theta), R *
                              np.sin(theta)], dtype=np.float64)
            L = np.linalg.norm(x-x_star)

            integrand = np.append(integrand, np.dot(x-x_star, normal)/L**3)

        self.integrand = integrand/(cells-1)
        return(np.sum(self.integrand))
    
    
    #Now to obtain the coefficients along a vessel:    
    def get_single_layer_point(self, axial_disc, point):
        """The influence of the whole vessel on the point. point is given between
        zero and one"""
        x=np.linspace(0,self.L*(1-1/axial_disc),axial_disc)+self.L/axial_disc/2
        self.x=x
        
        G=np.empty(len(x))
        
        #I calculate manually in which DoF does the point fall 
        j=int(np.around(axial_disc*point))
        
        for i in range(len(x)):
            inc_s=np.abs(x[i]-x[j])
            if i!=j:
                G[i]=self.G_ij_elliptical(inc_s)
            else:
                G[i]=self.G_ii_analytical(self.L/axial_disc)+0.28
        self.G=G
        return self.G*self.L/axial_disc
    
    
    def get_double_layer_point(self,axial_disc, point):
        """The influence of the whole vessel on the point. point is given between
        zero and one"""
        x=np.linspace(0,self.L*(1-1/axial_disc),axial_disc)+self.L/axial_disc/2
        self.x=x
        
        H=np.empty(len(x))
        
        #I calculate manually in which DoF does the point fall 
        j=int(np.around(axial_disc*point))
        
        for i in range(len(x)):
            inc_s=np.abs(x[i]-x[j])
            if i!=j:
                H[i]=self.H_ij_analytical(inc_s)*self.L/axial_disc
            else:
                H[i]=self.H_ii_non_singular(self.L/axial_disc)*self.L/axial_disc+0.5
        
        self.H=H
        return self.H
    
    
    def get_single_layer_vessel(self, axial_disc):
        """Computes the full matrix of single layer coefficients for a whole 
        straight vessel. Gij for all i and all j"""
        x=np.linspace(0,self.L*(1-1/axial_disc),axial_disc)+self.L/axial_disc/2
        self.x=x
        if self.L/axial_disc<self.R:
            G_hat=np.empty((len(x), len(x)))
            
            for i in range(len(x)):
                for j in range(len(x)):
                    if i!=j:
                        inc_s=np.abs(x[i]-x[j])
                        G_hat[i,j]=self.G_ij_elliptical(inc_s)
            
            G_self_hat=self.G_ii_analytical(self.L/axial_disc)+0.28
            
            G_hat[np.arange(len(x)),np.arange(len(x))]=G_self_hat
            self.G_hat=G_hat
            #Now, later we multiply by longitude
            return self.G_hat*self.L/axial_disc
        else: 
            ratio=int(2*self.L/self.R/axial_disc)
            return self.get_single_layer_vessel_coarse(axial_disc, ratio)
            
    
    def get_double_layer_vessel(self, axial_disc):
        """Computes the full matrix of double layer coefficients for a whole 
        straight vessel. Hij for all i and all j"""
        x=np.linspace(0,self.L*(1-1/axial_disc),axial_disc)+self.L/axial_disc/2
        self.x=x
        
        H=np.empty((len(x), len(x)))
        
        for i in range(len(x)):
            for j in range(len(x)):
                if i!=j:
                    inc_s=np.abs(x[i]-x[j])
                    H[i,j]=self.H_ij_analytical(inc_s)
        H*=self.L/axial_disc
        H_self=self.H_ii_non_singular(self.L/axial_disc)*self.L/axial_disc+0.5
        #H_self=self.H_ii_non_singular(self.L/axial_disc)*self.L/axial_disc
        H[np.arange(len(x)),np.arange(len(x))]=H_self
        self.H=H
        return self.H
    
    def get_double_layer_vessel_coarse(self, axial_disc, ratio):
        """Calculates the double layer influence coefficients for a single vessel. Since the treatment
        of the singularity requires the axial discretization to be small, here we compute the coefficients 
        for a fine discretization and then integrate them to provide an average cross influence coefficients 
        for a discretization...."""
        x=np.linspace(0,self.L*(1-1/axial_disc),axial_disc)+self.L*1/axial_disc/2
        fine_disc=axial_disc*ratio
        DL_vessel=np.zeros((axial_disc, axial_disc))
        
        for j in range(axial_disc):
            #Calculates the influence coefficients over this i.e. H_kj for all k
            point=x[j]
            intt=np.zeros(axial_disc) #array that will store the integrated values 
            DL_point=self.get_double_layer_point(fine_disc, point/self.L)
            for k in range(axial_disc):
                intt[k]=np.sum(DL_point[ratio*k:ratio*(1+k)])
            DL_vessel[j]=intt
        return(DL_vessel)
    
    def get_single_layer_vessel_coarse(self, axial_disc, ratio):
        """Calculates the double layer influence coefficients for a single vessel. Since the treatment
        of the singularity requires the axial discretization to be small, here we compute the coefficients 
        for a fine discretization and then integrate them to provide an average cross influence coefficients 
        for a discretization...."""
        x=np.linspace(0,self.L*(1-1/axial_disc),axial_disc)+self.L*1/axial_disc/2
        fine_disc=axial_disc*ratio
        SL_vessel=np.zeros((axial_disc, axial_disc))
        
        for j in range(axial_disc):
            #Calculates the influence coefficients over this i.e. H_kj for all k
            point=x[j]
            intt=np.zeros(axial_disc) #array that will store the integrated values 
            SL_point=self.get_single_layer_point(fine_disc, point/self.L)
            for k in range(axial_disc):
                intt[k]=np.sum(SL_point[ratio*k:ratio*(1+k)])
            SL_vessel[j]=intt
        return(SL_vessel)
    


class Alternatives():
    def __init__(self, L, R, axial_disc):
        self.L = L
        self.R = R
        x=np.linspace(0,self.L*(1-1/axial_disc),axial_disc)
        self.x=x
        self.h=self.L/axial_disc
        x+=self.h/2
        self.x=x
        
    def Gjerde_point(self, point):
        R=self.R
        h=self.h
        x=self.x
        G=np.empty(len(x))
        j=int(np.around(len(x)*point))
        x_j=np.array([x[j], R, 0])
        for i in range(len(x)):
            G[i]=Gjerde(x_j,
                          np.array([x[i]-h/2, 0,0]),
                             np.array([x[i]+h/2, 0, 0]), self.R)
        self.G=G
        return G
    
    def Fleischman_point(self, point):
        R=self.R
        h=self.h
        x=self.x
        G=np.empty(len(x))
        j=int(np.around(len(x)*point))
        x_j=np.array([x[j], R, 0])
        for i in range(len(x)):
            inc_s=np.abs(x[i]-x_j[0])
            G[i]=Fleischman(inc_s, R)*h
        self.G=G
        return G
    
    def get_my_H_ij(self, inc_s, h):
        R=self.R
        x_star=np.array([0,R,0])
        a=np.array([inc_s-h/2, 0,0])
        b=np.array([inc_s+h/2, 0,0])
        
        d=np.linalg.norm(x_star-np.array([inc_s, 0,0]))
        
        es=np.array([1,0,0])
        
        return (np.dot(x_star-a, es)-np.dot(x_star-b, es))/d**3
        
    def My_double_point(self,  point):
        R=self.R
        h=self.h
        x=self.x
        H=np.empty(len(x))
        j=int(np.around(len(x)*point))
        x_j=np.array([x[j], R, 0])
        es=np.array([1,0,0])
        
        self.j=j
        
        for i in range(len(x)):
            
            inc_s=x[i]-x[j]
            H[i]=self.get_my_H_ij(inc_s, h)
            
        self.H=H*R**2/np.pi
        return self.H

        



# =============================================================================
# #%% Fleischman vs Gjerde:
# cells=100
# L=20
# R=1
# h=L/cells
# x=np.linspace(0+h/2,L-h/2,cells)
# Fl=np.array(())
# Gj=np.array(())
# 
# x_j=np.array([L/3,R,0])
# 
# for i in x:
#     x_i=np.array([i,0,0])
# 
#     Fl=np.append(Fl, Fleischman(x_j[0]-i, R))
#     Gj=np.append(Gj, Gjerde(x_j,
#                             x_i-np.array([h/2,0,0]),
#                             x_i+np.array([h/2,0,0]),
#                             R))
# 
# plt.plot(x,Fl, label='Fl')
# plt.plot(x,Gj/h, label='Gj')
# 
# cells=int(cells/10)
# h=L/cells
# x=np.linspace(0+h/2,L-h/2,cells)
# Fl=np.array(())
# Gj=np.array(())
# for i in x:
#     x_i=np.array([i,0,0])
# 
#     Fl=np.append(Fl, Fleischman(x_j[0]-i, R))
#     Gj=np.append(Gj, Gjerde(x_j,
#                             x_i-np.array([h/2,0,0]),
#                             x_i+np.array([h/2,0,0]),
#                             R))
# 
# plt.plot(x,Fl, label='Fl - coarse')
# plt.plot(x,Gj/h, label='Gj - coarse')
# 
# 
# 
# plt.legend()
# plt.title("Comparison Fleischman vs Gjerde")
# plt.show()
# 
# 
# 
# #%%
# a = Classic(L, R)
# 
# #%% - Comparison Single Layer cross - influence
# inc_s=L/5
# print("Estimation G_ij with scipy integration", a.G_ij_analytical(inc_s))
# print("Estimation G_ij with elliptical integration", a.G_ij_elliptical(inc_s))
# print("Estimation G_ij brute force", a.G_ij_numerical(inc_s, cells))
# 
# plt.plot(a.angle, 1/a.G(a.angle), label='Scipy')
# plt.scatter(a.angle, 1/a.integrand, label='brute force')
# plt.legend()
# plt.show()
# 
# 
# 
# # %% - NOw the double layer ones
# 
# print("Estimation H_ij with scipy integration", a.H_ij_analytical(inc_s))
# print("Estimation G_ij brute force", a.H_ij_numerical(inc_s, cells))
# plt.plot(a.angle, a.H(a.angle), label='Scipy')
# plt.scatter(a.angle, a.integrand, label='brute force')
# plt.show()
# 
# #%% - Convergence of self double layer:
# 
# c=0
# for cells in np.array([10,20,50,100,500,700,1000]):
#     a = Classic(L,R)
#     H=a.get_double_layer_point(cells, 0.5)  
#     plt.scatter(a.x,H, label=c)
#     print("Sum of all H_ij for a single line with {} cells is= {}".format(cells, np.sum(H)))
#     c+=1
# plt.legend()
# plt.show()
# 
# #%% - Comparison of the three options to calculate the single layer potential. Notice the 
# 
# c=0
# for cells in np.array([10,100,1000]):
#     a = Classic(L,R)
#     G=a.get_single_layer_point(cells, 0.5)  
#     plt.plot(a.x,G*cells/L, label=c)
#     c+=1
#     
# b=Alternatives(L,R, int(cells))
# G_line=b.Gjerde_point(0.5)
# G_Fl=b.Fleischman_point(0.5)
# plt.plot(b.x,G_line*cells/L, label='Gjerde')
# plt.plot(b.x,G_Fl*cells/L, label='Fleischman')
# plt.legend()
# plt.show()
# 
# #%%
# 
# plt.plot(b.x,b.My_double_point(0.5), label='mine')
# plt.plot(a.x,a.get_double_layer_point(len(a.x), 0.5), label="Analyt")
# plt.title("My double layer")
# print(np.sum(b.H))
# 
# 
# #%% - Convergence of self single layer:
# 
# c=0
# for cells in np.array([10,20,50,100,500,1000]):
#     a = Classic(L,R)
#     G=a.get_single_layer_vessel(cells)  
#     plt.plot(a.x,G[int(cells/2)], label=c)
#     c+=1
# plt.legend()
# plt.show()
# 
# 
# # %% - NOw the double layer ones
# 
# print(a.H_ij_analytical(inc_s))
# print(a.H_ij_numerical(inc_s, cells))
# plt.plot(a.angle, a.H(a.angle), label='Scipy')
# plt.scatter(a.angle, a.integrand, label='brute force')
# plt.show()
# 
# #%% - Convergence of self double layer:
# 
# c=0
# for cells in np.array([10,20,50,100,500,700,1000]):
#     a = Classic(L,R)
#     H=a.get_double_layer_vessel(cells)  
#     plt.scatter(a.x,H[int(cells/2)], label=c)
#     print(np.sum(H[int(cells/2)]))
#     c+=1
# plt.legend()
# plt.show()
# 
# #%%
# 
# c=0
# for cells in np.array([10,20,50,100,500,1000]):
#     a = Classic(L,R)
#     Gj=a.get_single_layer_vessel(cells)  
#     plt.plot(a.x,G[int(cells/2)], label=c)
#     c+=1
# plt.legend()
# plt.show()
# 
# =============================================================================

    
