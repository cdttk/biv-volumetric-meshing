#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:22:29 2022

@author: sq20
"""

import pandas as pd
import glob
import logging
import math
import random
import numpy as np
import os

logger = logging.getLogger(__name__)

def Laplace_solver(MPIEXEC,OPTS,CARP,parfile,mesh,SIMID,*argv):
    
    # LV,RV_septum and RV Laplace solver
    cmd=MPIEXEC+" -np 8 "+CARP+" -experiment 2 +F "+parfile+" -meshname "+mesh+" -simID "+SIMID+" -ellip_options_file "+ OPTS+" "
    if not argv==None:
        for arg in argv:
            cmd=cmd+arg+" "
    logger.debug(cmd)
    return os.system(cmd) == 0
    
def igb_todata(IGBEXTRACT,phieGrad_path):    
    cmd="%s -o ascii -O %s/phieGrad.dat %s/phie.igb " % (IGBEXTRACT,phieGrad_path,phieGrad_path)
    logger.debug(cmd)
    os.system(cmd)
    gradMag=np.genfromtxt(phieGrad_path+'/phieGrad.dat')
    #gradMag=gradMag[0,:]
    return gradMag

def GlGradient_data(GlGradient,MESH,datatype,inputdata,outputdata):    
    #data types: elem_ctr,vtx
    cmd = "%s extract gradient -msh %s -idat %s -odat %s" % (GlGradient, MESH, inputdata, outputdata)
    logger.debug(cmd)
    return os.system(cmd) == 0
    
    
