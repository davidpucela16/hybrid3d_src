#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:26:41 2023

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt 
import pdb

import sys
sys.path.append('../src/')


# =============================================================================
# def Green_line_orig(arg, D):
#     """Returns the average value of the integral without the coefficient i.e. to the result one would have to multiply
#     by the surface of the open cylinder (2 \pi R_j L_j)/(4*pi*D) to obtain a proper single layer potential
#     
#     DONT FORGET THE DIFUSSION COEFFICIENT
#     
#     THIS IS THE ORIGINAL FUNCTION RIGHT AFTER INTEGRATION WHICH WILL CONTAIN SINGULARITIES"""
#     x, a, b=arg
#     ra=np.linalg.norm(x-a)
#     rb=np.linalg.norm(x-b)
#     
#     L=np.linalg.norm(b-a)
#     tau=(b-a)/L
#     
#     num=ra + L/2 + np.dot(x-(a+b)/2,tau)
#     den=rb-L/2+np.dot(x-(a+b)/2,tau)
#     
#     return np.log(num/den)/(4*np.pi*D)
# =============================================================================
from numba import njit
@njit
def LogLine(x, a, b):
    """Returns the average value of the integral without the coefficient i.e. to the result one would have to multiply
    by the surface of the open cylinder (2 \pi R_j L_j)/(4*pi*D) to obtain a proper single layer potential
    
    DONT FORGET THE DIFUSSION COEFFICIENT"""
    ra=np.linalg.norm(x-a)
    rb=np.linalg.norm(x-b)
    
    L=np.linalg.norm(b-a)
    tau=(b-a)/L
    
    log=np.log((np.max(np.array([ra, rb])) + L/2 + np.abs(np.dot((a+b)/2-x,tau)))/(np.min(np.array([ra, rb])) - L/2 + np.abs(np.dot((a+b)/2-x,tau))))

    
    return log

@njit
def GradPoint(x,x_j):
    return -(x-x_j)/(np.linalg.norm(x-x_j)**3)
@njit
def GetSourcePotential(a,b,x,D):
    """Returns two arrays, one to multiply to q and another to multiply to Cv
    It includes both the single layer and double layer
        - center_source is self explanatory, center of the cylinder/segment
        - R is the radius of the cylinder
        - D is the diffusion coefficient
        - K is the effective diffusivity of the wall [m^2 s^-1]
        - x is the point where the potential is calculated
    """
    Sj=LogLine(x, a,b)/(4*np.pi*D)
    return(Sj)

@njit
def GetGradSourcePotential( a,b,x, D):
    #tau=(b-a)/np.linalg.norm(b-a)
    x_j=(a+b)/2
    #The second empty array that is returned is the corresponding to the C_v
    return np.linalg.norm(b-a)*GradPoint(x,x_j)/(4*np.pi*D)
    #I have multiplied the grad point by the length of the segment, Im pretty sure it needs to be done to be consistent dimensionally 

@njit
def SimpsonSurface(a,b, function, center, h, normal, D):
    """Assumes a square surface
    
    As always, the integral must be multiplied by the surface OUTSIDE of the
    function
    
    source must be given as a tuple. If the function is:
        - Green_3D
        - grad_Green_3D
        - GetGradSourcePotential
        
    h and normal define the square surface where the function is integrated 
        - source is the position of the center of the source 
        - function is the function to be integrated
        - center is the surface of integration
        - h is the size of the square surface
        - normal is the normal vector to the surface
        - D is the diffusion coefficient
 
    else: arg=x_j 
        """
    w_i=np.array([1.0,4.0,1.0,4.0,16.0,4.0,1.0,4.0,1.0])/36
    
    corr = np.array([[-1,-1,],
                [0, -1],
                [1, -1],
                [-1, 0],
                [0, 0],
                [1, 0],
                [-1, 1],
                [0, 1],
                [1, 1]], dtype=np.float32)*h/2
    #This only works because it is a Cartesian grid and the normal has to be parallel to one of the axis
    #Other wise we would need another way to calculate this tangential vector
    #tau represents one of the prependicular vectors to normal 
    tau_1, tau_2=np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    tau_1[np.where(normal==0)[0][0]]=1
    tau_2[np.where(normal==0)[0][1]]=1
    integral=0
    if function=='P':
        for i in range(len(w_i)):
            pos=center+corr[i,0]*tau_1+corr[i,1]*tau_2
            #The function returns two kernels that cannot be multiplied 
            w_integral=GetSourcePotential(a,b,pos, D)
            integral+=w_integral*w_i[i]
    elif function=='G':
        for i in range(len(w_i)):
            pos=center+corr[i,0]*tau_1+corr[i,1]*tau_2
            #The function returns two kernels that cannot be multiplied 
            grad=GetGradSourcePotential(a,b,pos, D)
            integral+=np.dot(grad,normal.astype(np.float64) )*w_i[i]

# =============================================================================
#     elif function==GetGradSourcePotential:
#         #I've realized we can calculate this analytically more easily, so here is 
#         #the attempt
#         d=center-(a+b)/2 #array pointing to the center of the tile 
#         integral=-np.dot(d, normal)/4/np.pi/np.linalg.norm(d)**3
# =============================================================================
            
    return integral




def unit_test_Simpson(h,D):
    aa,bb=np.array([0,-0.2,0]),np.array([0,0.2,0])
    a=SimpsonSurface(aa,bb, 'G', np.array([0,0,-h/2]),h, np.array([0,0,-1]), D)*h**2
    
    b=GradPoint(np.array([0,0,0]), np.array([0,h/2,0]))[1]*h**2*16/36
    
    b+=GradPoint(np.array([0,0,0]), np.array([-h/2,h/2,0]))[1]*h**2*4/36
    b+=GradPoint(np.array([0,0,0]), np.array([0,h/2,-h/2]))[1]*h**2*4/36
    b+=GradPoint(np.array([0,0,0]), np.array([0,h/2,h/2]))[1]*h**2*4/36
    b+=GradPoint(np.array([0,0,0]), np.array([-h/2,h/2,0]))[1]*h**2*4/36
    
    b+=GradPoint(np.array([0,0,0]), np.array([-h/2,h/2,-h/2]))[1]*h**2/36
    b+=GradPoint(np.array([0,0,0]), np.array([-h/2,h/2,h/2]))[1]*h**2/36
    b+=GradPoint(np.array([0,0,0]), np.array([h/2,h/2,-h/2]))[1]*h**2/36
    b+=GradPoint(np.array([0,0,0]), np.array([h/2,h/2,h/2]))[1]*h**2/36
    
    from scipy.integrate import dblquad
    def integrand(y, x):
        
        p_1=np.array([0,0,0])
        p_2=np.array([x,y,h/2])
        r = np.dot(GradPoint(p_2, p_1), np.array([0,0,1]))
        return r / (4*np.pi*D)
    
    scp, _ = dblquad(integrand, -h/2,h/2 , -h/2,  h/2)
    #The following is the exact result of the Simpson's integration done by hand (notebook)
    print("With scipy integration: ", scp)
    print("Manually with the grad point function", -b/4/np.pi/D)
    print("Analytical Simpson= ", -(27**-0.5+2**0.5+4)/(9*np.pi*D))
    #The following is the value returned by the function 
    print("Calculated Simpson= ", a/np.linalg.norm(aa-bb)) #We have to divide by the length of the source since it is included in the integral
    #The following is the analytical value of the integral
    print("Analytical (Gauss)= ", -1/(6*D))
    return

def another_unit_test_Simpson():
    """integral over the faces of a cube to make sure the gradient of the greens function
    over a closed surface is 0"""
    normal=np.array([[0,0,1],
                     [0,0,-1],
                     [0,1,0],
                     [0,-1,0],
                     [1,0,0],
                     [-1,0,0]])
    integral=0
    for h in np.array([1,2,3,9]): #size of the cube
        #pdb.set_trace()    
        a,b=np.array([0,-0.2,0]),np.array([0,0.2,0])
        L=0.4
        integral=0
        for i in range(6):
            no=normal[i]
            center=no*h/2
            integral+=SimpsonSurface(a,b, 'G', center, h, no, 1)*h**2/L
        print("This must be 1 due to properties of delta", integral)
    print("We expect around a 20% error")
        
        

def unit_test_single_layer(h, a):
    """h is the size of the square
    a is the separation from the square of the point where the source is located"""
    #Here we want to test the integration of the potential function
    from scipy.integrate import dblquad
    import time
    
    def integrand(y, x):
        r = np.sqrt(x**2 + y**2 + a**2)
        return 1 / (4*np.pi*r)
    
    t=np.zeros(4)
    
    t[0]=time.time()
    integral, _ = dblquad(integrand, -h/2,h/2 , -h/2,  h/2)
    t[1]=time.time()
    a,b=np.array([-0.1,0,0]),np.array([0.1,0,0])
    t[2]=time.time()
    mine=SimpsonSurface(a,b, 'P', np.array([0,0,a]), h, np.array([0,0,1]), 1)*h**2/np.linalg.norm(a-b)
    t[3]=time.time()
    
    print("Scipy integral: ", integral)
    print("Simpson integral", mine)
    
    return t[1]-t[0], t[3]-t[2]

@njit
def GetSelfInfluence(R,L, D):
    #####################################
    # REVIEW THE GEOMETRICAL COEFFICIENTS
    ####################################"
    a=np.array([0,0,0])
    b=np.array([0,0,L])
    x1=np.array([0,R,0])
    x2=np.array([0,R,L/2])
    x3=np.array([0,R,L])
    G1=LogLine(x1,a,b)
    G2=LogLine(x2,a,b)
    G3=LogLine(x3,a,b)
    return (G1+4*G2+G3)/6/(4*np.pi*D)


# =============================================================================
# a=np.array([0,0,0], dtype=np.float64)
# b=np.array([0,0,0.1], dtype=np.float64)
# x=np.array([0,10,0], dtype=np.float64)
# 
# unit_test_Simpson(4,2)
# another_unit_test_Simpson()
# =============================================================================
