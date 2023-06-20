#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:04:37 2023

@author: pdavid
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from scipy import integrate
import pdb

from Potentials_module import Classic, Alternatives, Gjerde, Fleischman
#%% Fleischman vs Gjerde:
cells=1100
L=20
R=1
h=L/cells
x=np.linspace(0+h/2,L-h/2,cells)
Fl=np.array(())
Gj=np.array(())

x_j=np.array([L/2,R,0])

for i in x:
    x_i=np.array([i,0,0])

    Fl=np.append(Fl, Fleischman(x_j[0]-i, R))
    Gj=np.append(Gj, Gjerde(x_j,
                            x_i-np.array([h/2,0,0]),
                            x_i+np.array([h/2,0,0]),
                            R))



a = Classic(L, R)
exact=a.get_single_layer_point(cells, 1/2)
plt.plot(x,Fl, label='Point source')
plt.plot(x,Gj/h, label='Line source')
plt.plot(a.x,exact/h, label='Exact')
plt.legend()
plt.show()



cells=int(cells/100)
h=L/cells
x=np.linspace(0+h/2,L-h/2,cells)
Fl=np.array(())
Gj=np.array(())
for i in x:
    x_i=np.array([i,0,0])

    Fl=np.append(Fl, Fleischman(x_j[0]-i, R))
    Gj=np.append(Gj, Gjerde(x_j,
                            x_i-np.array([h/2,0,0]),
                            x_i+np.array([h/2,0,0]),
                            R))
a = Classic(L, R)
exact=a.get_single_layer_point(cells, 1/2)
plt.plot(x,Fl, label='Point')
plt.plot(x,Gj/h, label='Line')
plt.plot(a.x,exact/h, label='Exact')
a = Classic(L, R)
exact_coarse=a.get_single_layer_point(cells, 1/2)
#plt.plot(a.x,exact_coarse/h, label='exact_coarse' )


plt.legend()
plt.title("Comparison Point vs Line")
plt.show()



#%%
a = Classic(L, R)

#%% - Comparison Single Layer cross - influence
cells=100
inc_s=L/5
print("Estimation G_ij with scipy integration", a.G_ij_analytical(inc_s))
print("Estimation G_ij with elliptical integration", a.G_ij_elliptical(inc_s))
print("Estimation G_ij brute force", a.G_ij_numerical(inc_s, cells))

plt.plot(a.angle, 1/a.G(a.angle), label='Scipy')
plt.scatter(a.angle, 1/a.integrand, label='brute force')
plt.legend()
plt.show()



# %% - NOw the double layer ones

print("Estimation H_ij with scipy integration", a.H_ij_analytical(inc_s))
print("Estimation G_ij brute force", a.H_ij_numerical(inc_s, cells))
plt.plot(a.angle, a.H(a.angle), label='Scipy')
plt.scatter(a.angle, a.integrand, label='brute force')
plt.show()

#%% - Convergence of self double layer:

c=0
for cells in np.array([10,20,50,100,500,700,1000]):
    a = Classic(L,R)
    H=a.get_double_layer_point(cells, 0.5)  
    plt.scatter(a.x,H, label=c)
    print("Sum of all H_ij for a single line with {} cells is= {}".format(cells, np.sum(H)))
    c+=1
plt.legend()
plt.show()

#%% - Comparison of the three options to calculate the single layer potential. Notice the 

c=0
for cells in np.array([10,100,1100]):
    a = Classic(L,R)
    G=a.get_single_layer_point(cells, 0.5)  
    plt.plot(a.x,G*cells/L, label=c)
    c+=1
    
b=Alternatives(L,R, int(cells))
G_line=b.Gjerde_point(0.5)
G_Fl=b.Fleischman_point(0.5)
plt.plot(b.x,G_line*cells/L, label='Gjerde')
plt.plot(b.x,G_Fl*cells/L, label='Fleischman')
plt.legend()
plt.show()

#%%
DL_point=a.get_double_layer_point(len(a.x), 0.5)
plt.plot(b.x,b.My_double_point(0.5), label='mine')
plt.plot(a.x,DL_point, label="Analyt")
plt.title("My double layer")
print(np.sum(b.H))



#%%
intt=np.zeros(11)
ratio=int(len(a.x)/len(intt))
for i in range(len(x)):
    intt[i]=np.sum(DL_point[ratio*i:ratio*(1+i)])
plt.plot(intt)
plt.title("shape of the double layer for L={} and R={}".format(L,R))

#%% - Convergence of self single layer:

c=0
for cells in np.array([10,20,50,100,500,1000]):
    a = Classic(L,R)
    G=a.get_single_layer_vessel(cells)  
    plt.plot(a.x,G[int(cells/2)], label=c)
    c+=1
plt.legend()
plt.show()


# %% - NOw the double layer ones

print(a.H_ij_analytical(inc_s))
print(a.H_ij_numerical(inc_s, cells))
plt.plot(a.angle, a.H(a.angle), label='Scipy')
plt.scatter(a.angle, a.integrand, label='brute force')
plt.show()

#%% - Convergence of self double layer:

c=0
for cells in np.array([10,20,50,100,500]):
    a = Classic(L,R)
    H=a.get_double_layer_vessel(cells)  
    plt.scatter(a.x,H[int(cells/2)], label=c)
    print(np.sum(H[int(cells/2)]))
    c+=1
plt.legend()
plt.show()

#%%

c=0
for cells in np.array([10,20,50,100,500]):
    a = Classic(L,R)
    Gj=a.get_single_layer_vessel(cells)  
    plt.plot(a.x,Gj[int(cells/2)], label=c)
    c+=1
plt.legend()
plt.show()
