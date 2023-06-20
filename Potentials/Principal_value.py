#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:55:18 2023

@author: pdavid
"""
import numpy as np 

disc=1000
theta=np.linspace(0,2*np.pi*(1-1/disc), disc)
R=20

x_star=np.array([R,0])*0.9
normal_star=np.array([1,0])

PV=0

#for i in theta[1:]:
for i in theta:
    x=np.array([np.cos(i), np.sin(i)])*R
    d=x-x_star
    normal=np.array([np.cos(i), np.sin(i)])
    PV+=np.dot(d, normal)/np.linalg.norm(d)**2

PV*=R/(disc-1)
print(PV)