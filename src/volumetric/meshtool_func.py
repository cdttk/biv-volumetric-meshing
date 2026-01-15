#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 11:46:28 2022

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

def merge_mesh(meshtoolloc,msh1,msh2,outmsh,ifmt=None,ofmt=None):
    #merge two meshes
    if ifmt == None:
        ifmt="carp_txt"
    
    if ofmt == None:
        ofmt="carp_txt"
    
    cmd=meshtoolloc+" merge meshes -msh1="+msh1+" -msh2="+msh2+" -outmsh="+outmsh+" -ifmt="+ifmt+" -ofmt="+ofmt

    logger.debug(cmd)
    return os.system(cmd) == 0

def convert_mesh(meshtoolloc,imsh,omsh,ifmt=None,ofmt=None):
    #merge two meshes
    if ifmt == None:
        ifmt="carp_txt"
    
    if ofmt == None:
        ofmt="carp_txt"
    
    cmd=meshtoolloc+" convert -imsh="+imsh+" -omsh="+omsh+" -ifmt="+ifmt+" -ofmt="+ofmt

    logger.debug(cmd)
    return os.system(cmd) == 0

def generate_mesh(meshtoolloc,surf,outmsh,*argv,ifmt=None,ofmt=None): 
    if ifmt == None:
        ifmt="carp_txt"
    
    if ofmt == None:
        ofmt="carp_txt"
    cmd="timeout 30s "+meshtoolloc+" generate mesh -surf="+surf+" -outmsh="+ outmsh +" -ifmt="+ifmt+" -ofmt="+ofmt+" "
    
    if not argv==None:
        for arg in argv:
            cmd=cmd+arg+" "

    logger.debug(cmd)
    return os.system(cmd) == 0


def resample_mesh(meshtoolloc,msh,min,max,outmsh,ifmt=None,ofmt=None): 
    if ifmt == None:
        ifmt="carp_txt"
    
    if ofmt == None:
        ofmt="carp_txt"
        
    cmd=meshtoolloc+" resample mesh -msh="+msh+" -min="+min+" -max="+max+" -outmsh="+outmsh+" -ifmt="+ifmt+" -ofmt="+ofmt+" " 
    print ("Running simulation...")

    logger.debug(cmd)
    return os.system(cmd) == 0

def query_edge(meshtoolloc,msh,ifmt=None,file="edgeinfo"):
    if ifmt == None:
        ifmt="carp_txt"
    cmd =meshtoolloc+" query edges -msh="+msh+" -ifmt="+ifmt
# run sim

    logger.debug(cmd)
    pipe = os.popen(cmd)

    # saving the output
    output = pipe.read()
    with open(file+'.txt', 'w+') as f:
        f.write(output)
        f.close() 

    return pipe.close() == None
    
def extract_surf(meshtoolloc,msh,surf,op,*argv):   
    
    cmd=meshtoolloc+" extract surface -msh="+msh+" -surf="+surf+" -op="+op+"  "
    if not argv==None:
        for arg in argv:
            cmd=cmd+arg+" "

    logger.debug(cmd)
    return os.system(cmd) == 0
    
def extract_mesh(meshtoolloc,msh,tags,submsh,ifmt=None,ofmt=None):   
    
    if ifmt == None:
        ifmt="carp_txt"
    
    if ofmt == None:
        ofmt="carp_txt"

    cmd=meshtoolloc+" extract mesh -msh="+msh+" -tags="+tags+" -submsh="+submsh+" -ifmt="+ifmt+" -ofmt="+ofmt+" " 

    logger.debug(cmd)
    return os.system(cmd) == 0

def map_file(meshtoolloc,submsh,file,outdir,mode=None):  
#default is mesh to submesh     
    if mode == None:
        mode="m2s"
    cmd=meshtoolloc+" map -submsh="+submsh+" -files="+file+"  -outdir="+outdir+" -mode="+mode

    logger.debug(cmd)
    return os.system(cmd) == 0
    
def insert_submesh(meshtoolloc,submsh,msh,outmsh):   
    
    cmd=meshtoolloc+" insert submesh -submsh="+submsh+" -msh="+msh+" -outmsh="+outmsh

    logger.debug(cmd)
    return os.system(cmd) == 0

def generate_dfield(meshtoolloc,msh,ssurf,odat,*argv):
    cmd=meshtoolloc+" generate distancefield -msh="+msh+" -ssurf="+ssurf+" -odat="+odat+" "
    if not argv==None:
        for arg in argv:
            cmd=cmd+arg+" "
    
    logger.debug(cmd)
    return os.system(cmd) == 0
    
def insert_data(meshtoolloc,submsh,submsh_data,msh,odat,mode): 
    #insert data: data defined on a submesh is inserted back into a mesh
    #-mode=<int>		 (optional) Data mode. 0 = nodal, 1 = element. Default is 0.
    cmd=meshtoolloc+" insert data -submsh="+submsh+" -submsh_data="+submsh_data+" -msh="+msh+" -odat="+odat+" -mode="+mode
    logger.debug(cmd)
    return os.system(cmd) == 0
    
def extract_data(meshtoolloc,submsh,msh_data,submsh_data,mode):   
    
    cmd=meshtoolloc+" extract data -submsh="+submsh+" -submsh_data="+submsh_data+" -msh_data="+msh_data+" -mode="+mode
    logger.debug(cmd)
    return os.system(cmd) == 0
    
def insert_fibers(meshtoolloc,submsh,submsh_data,msh,odat,mode): 
    #insert data: data defined on a submesh is inserted back into a mesh
    #-mode=<int>		 (optional) Data mode. 0 = nodal, 1 = element. Default is 0.
    cmd=meshtoolloc+" insert data -submsh="+submsh+" -submsh_data="+submsh_data+" -msh="+msh+" -odat="+odat+" -mode="+mode
    logger.debug(cmd)
    return os.system(cmd) == 0

def extract_gradient(meshtoolloc,msh,idat,odat,mode,ifmt=None):   
    #mode=<int>	 (optional) output mode. 0 == nodal output, 1 == element output. 0 is default.
    if ifmt == None:
        ifmt="carp_txt"

    cmd=meshtoolloc+" extract gradient -msh="+msh+" -idat="+idat+" -odat="+odat+" -mode="+mode+" -ifmt="+ifmt
    print(cmd)
    os.system(cmd)
    return os.system(cmd) == 0


def interpolate_elemdata(meshtoolloc,imsh,idat,omsh,odat):   
    #-omsh=<path>	 (input) path to basename of the mesh we interpolate to
    #-imsh=<path>	 (input) path to basename of the mesh we interpolate from

    cmd=meshtoolloc+" interpolate elemdata -imsh="+imsh+" -idat="+idat+" -omsh="+omsh+" -odat="+odat
    print(cmd)
    os.system(cmd)
    return os.system(cmd) == 0

def collect_nodal(meshtoolloc, imsh, ifmt, omsh, ofmt, nod):
    cmd = '{} collect -imsh={} -ifmt={} -omsh={} -ofmt={} -nod={}'.format(meshtoolloc, imsh, ifmt, omsh, ofmt, nod)
    logger.debug(cmd)
    return os.system(cmd) == 0
