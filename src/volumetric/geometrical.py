#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file was developed by Martin Bishop
"""
Created on Wed Jul 27 09:40:26 2022

@author: sq20
"""
import pandas as pd
import glob
import math
from math import acos,asin,cos,sin,sqrt
import random
import numpy as np


#########################################
# Function to rotate one vector to be aligned with another (Marina Strocchi)
#########################################
def computeRotationMatrix(v_target, v_rotate):
    
    ax = np.cross(v_target,v_rotate)

    ax = ax/np.linalg.norm(ax)

    cos_theta = np.dot(v_target,v_rotate)

    theta = -acos(cos_theta)

    R = np.zeros((3,3))  
    R[0,0] = ax[0]**2 + cos(theta) * (1 - ax[0]**2);
    R[0,1] = (1 - cos(theta)) * ax[0] * ax[1] - ax[2] * sin(theta);
    R[0,2] = (1 - cos(theta)) * ax[0] * ax[2] + ax[1] * sin(theta);
    R[1,0] = (1 - cos(theta)) * ax[0] * ax[1] + ax[2] * sin(theta);
    R[1,1] = ax[1]**2 + cos(theta) * (1 - ax[1]**2);
    R[1,2] = ( 1 - cos(theta)) * ax[1] * ax[2] - ax[0] * sin(theta);
    R[2,0] = ( 1 - cos(theta)) * ax[0] * ax[2] - ax[1] * sin(theta);
    R[2,1] = ( 1 - cos(theta)) * ax[1] * ax[2] + ax[0] * sin(theta);
    R[2,2] = ax[2]**2 + cos(theta) * (1 - ax[2]**2);

    return R
    
#########################################
# Function to rotate one vector to be aligned with another (Marina Strocchi)
#########################################
def computeAxisRotationMatrix(u, theta):
    
    R11 = cos(theta) + u[0]**2*(1-cos(theta))
    R12 = u[0]*u[1]*(1-cos(theta)) - u[2]*sin(theta)
    R13 = u[0]*u[2]*(1-cos(theta)) + u[1]*sin(theta)
    R21 = u[1]*u[0]*(1-cos(theta)) + u[2]*sin(theta)
    R22 = cos(theta) + u[1]**2*(1-cos(theta))
    R23 = u[1]*u[2]*(1-cos(theta)) - u[0]*sin(theta)
    R31 = u[2]*u[0]*(1-cos(theta)) - u[1]*sin(theta)
    R32 = u[2]*u[1]*(1-cos(theta)) + u[0]*sin(theta)
    R33 = cos(theta) + u[2]**2*(1-cos(theta))

    R = [[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]]

    return R

#########################################
# Function to compute area of triangle from vertices
#########################################
def computeTriangleArea(a, b, c):
    
    ab = b-a
    ac = c-a
    
    area = 0.5*np.cross(ab,ac)
    
    return area
    
#########################################
# Function to compute normal to plane given three points in plane
#########################################
def computeNormalToPlane(a, b, c):

    # defines two vectors linking point A with point B, and point A with point C
    v_ab = b-a
    v_ac = c-a
    
    # computes the normal (in a consistent manner)
    normal = np.cross(v_ab,v_ac)
    
    normal = normal/np.linalg.norm(normal)
    
    return normal
