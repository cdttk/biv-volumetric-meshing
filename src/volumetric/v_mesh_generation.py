#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:31:59 2022

@author: sq20
"""

import pandas as pd
import numpy as np
import os
import glob
import math
import random
import shutil

from . import carpfunc
from . import meshIO
from . import meshtool_func as meshtool

def generate_valve(meshtoolloc,input_dir,folder,Valvename,endoname,outmsh,bdry_step,ptindex):
    #Valvename: the valve for generating mesh, str
    #endoname: the endo where the target valve is located,str
    #outmsh: the name of valve mesh,str
    #bdry_step: the distance of the lower plane from the original valve plane, typically 0.2-0.5
    #the element tag value for the valve mesh
    #ptindex: the index of the valve points in original .pts(totally 5810)
    if not os.path.exists(folder+'/'+outmsh):
        os.mkdir(folder+'/'+outmsh)
    os.chdir(folder+'/'+outmsh)
    
    if not meshtool.convert_mesh(meshtoolloc,input_dir+"/"+Valvename,Valvename,ifmt="vtk",ofmt=None):
        return False
    pts = meshIO.read_pts(basename=Valvename, file_pts=None)
    
    #to find the middle point in the lower plane of valve
    volumetricname=Valvename+"_volumetricmesh"
    if not meshtool.generate_mesh(meshtoolloc,Valvename,volumetricname,'-bdry_layers=-1','-bdry_step='+bdry_step):
        return False
   
    
    ptsvolume = meshIO.read_pts(basename=volumetricname, file_pts=None)
    indexofnotnan=np.empty((0,1),int)
    for i in range(len(pts.values),len(ptsvolume.values)):
        if not math.isnan(ptsvolume.values[i,1]):
            indexofnotnan=np.append(indexofnotnan,[i])
            
    #endo mesh reading in:
    if not meshtool.convert_mesh(meshtoolloc,input_dir+"/"+endoname,endoname,ifmt="vtk",ofmt=None):
        return False
    
    #add the middle point in the lower plane of valve to the original .pts file
    pts_LVendo=meshIO.read_pts(basename=endoname)
    
    pts_LVendo1= np.append(pts_LVendo.values,[ptsvolume.iloc[indexofnotnan[len(indexofnotnan)-1],:]],axis=0)
    #add middle point in the lower plane of valve to the .pts
    
    meshIO.write_pts(ptsFilename="LV_endo_with_lowplane"+Valvename, pts=pd.DataFrame(pts_LVendo1))
    
    et_vertex_start_end = np.array(
        [[0, 1499], [1500, 2164], [2165, 3223], [3224, 5581],
          [5582, 5630], [5631, 5655], [5656, 5696], [5697, 5729],
          [5730, 5809]])
    '''Class constant, surface index limits for vertices `et_pos`. 
        Surfaces are defined in the following order:
       
        LV_ENDOCARDIAL = 0 
        RV_SEPTUM = 1
        RV_FREEWALL = 2
        EPICARDIAL =3
        MITRAL_VALVE =4
        AORTA_VALVE = 5
        TRICUSPID_VALVE = 6
        PULMONARY_VALVE = 7
        RV_INSERT = 8
        '''
    elem_LVendo=meshIO.read_elem(basename=endoname,file_elem=None)
    
    start_idx= et_vertex_start_end[ptindex][0] #mitral valve points
    end_idx= et_vertex_start_end[ptindex][1]
    index=list(range(start_idx,end_idx))
    N=np.where(np.isin(elem_LVendo,index))[0] #find index for elements connected to valve plane
    unique, counts = np.unique(N, return_counts=True) #unique is the value and counts is how many times the value appears
    for i in range(len(unique)):
        if counts[i]==1:
            if  np.isin(elem_LVendo.iloc[unique[i],1],index):
                elem_LVendo.iloc[unique[i],1]=5810
            if  np.isin(elem_LVendo.iloc[unique[i],2],index):
                elem_LVendo.iloc[unique[i],2]=5810
            if  np.isin(elem_LVendo.iloc[unique[i],3],index):
                elem_LVendo.iloc[unique[i],3]=5810
    
    volumetricname=Valvename+"_volumetricmesh"
    src_basename = os.path.join(Valvename, volumetricname)
    
    meshIO.write_elem("LV_endo_with_lowplane"+Valvename,elem_LVendo,shapes="Tr")
    
    if not meshtool.convert_mesh(meshtoolloc,"LV_endo_with_lowplane"+Valvename,"LV_endowithlowplane"+Valvename,ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc,"LV_endowithlowplane"+Valvename,input_dir+"/"+Valvename,"merge1",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc,"merge1",input_dir+"/"+endoname,"merge"+Valvename,ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.generate_mesh(meshtoolloc,"merge"+Valvename,outmsh,ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.convert_mesh(meshtoolloc,outmsh,outmsh,ifmt="vtk"):
        return False
    
    #retag the valve mesh to new tag
    elem=meshIO.read_elems(basename=outmsh,file_elem=None)
    elem.insert(5,'tag',ptindex)
    #print (elem.head)
    elem = elem.drop(5, axis=1) # axis 1 drops columns, 0 will drop rows that match index value in labels
    meshIO.write_elem(outmsh, pd.DataFrame(elem),shapes="Tt")
    
    if not meshtool.convert_mesh(meshtoolloc,outmsh,outmsh,ofmt="vtk"):
        return False

    print("new mesh is saved as ", outmsh)
    os.chdir(folder)
          
    return True

def merge_resample(meshtoolloc,input_dir,folder,LV_endo,RV_FW,RV_septum,epi,Valves,min,max,outmsh):
    if not os.path.exists(outmsh):
        os.mkdir(outmsh)
    os.chdir(outmsh)
    #merge all myo surfaces and generate mesh
    if not meshtool.merge_mesh(meshtoolloc,input_dir+"/"+LV_endo,input_dir+"/"+RV_FW,"M1",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc,"M1",input_dir+"/"+RV_septum,"M2",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc,"M2",input_dir+"/"+epi,"M3",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.generate_mesh(meshtoolloc,"M3","M3_volume",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.convert_mesh(meshtoolloc,"M3_volume","M3_volume",ifmt="vtk"):
        return False
    
    #merge myo mesh with valve meshes
    if not meshtool.merge_mesh(meshtoolloc,"M3_volume",folder+'/'+Valves[0]+'/'+Valves[0],"M_valve1",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc,"M_valve1",folder+'/'+Valves[1]+'/'+Valves[1],"M_valve12",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc,"M_valve12",folder+'/'+Valves[2]+'/'+Valves[2],"M_valve123",ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.merge_mesh(meshtoolloc,"M_valve123",folder+'/'+Valves[3]+'/'+Valves[3],"M_valve1234",ifmt="vtk",ofmt="vtk"):
        return False
    
    #resample all mesh and save edge lengths in "edgeinfo.txt"
    if not meshtool.resample_mesh(meshtoolloc,"M_valve1234",min,max,outmsh,ifmt="vtk",ofmt="vtk"):
        return False
    if not meshtool.convert_mesh(meshtoolloc,outmsh,outmsh,ifmt="vtk"):
        return False
    if not meshtool.query_edge(meshtoolloc,outmsh,file="edgeinfo"):
        return False
    
    if not meshtool.extract_mesh(meshtoolloc,outmsh,"0",outmsh+"_i"):
        return False
    
    #rewrite the lon file to (1,0,0) or the Laplace solver wont run
    lon=meshIO.read_fibres(basename=outmsh+"_i",file_lon=None)
    lon.insert(0,'x_axis',1)
    #print (elem.head)
    lonnew = lon.drop(2, axis=1) # axis 1 drops columns, 0 will drop rows that match index value in labels
    meshIO.write_lon(outmsh+"_i", pd.DataFrame(lonnew))
    if not meshtool.convert_mesh(meshtoolloc,outmsh+"_i",outmsh+"_i",ofmt="vtk"):
        return False
    
    os.chdir(folder)
    
    return True


    
def extract_surfacenolabel(meshtoolloc,mesh, pts5810,surfmesh,input_dir,RV_septum):
    #input mesh with labels as RV and LV:0, valves:4,5,6,7
    #input the original pts file for finding coords in LV_endo, RV_endo and RV_septum
    #output is 3 surf: LV_endo, RV_endo and RV_septum
    if not os.path.exists(surfmesh):
        os.mkdir(surfmesh)
    os.chdir(surfmesh)
    
    et_vertex_start_end = np.array(
        [[0, 1499], [1500, 2164], [2165, 3223], [3224, 5581],
          [5582, 5630], [5631, 5655], [5656, 5696], [5697, 5729],
          [5730, 5809]])
    '''Class constant, surface index limits for vertices `et_pos`. 
        Surfaces are defined in the following order:
       
        LV_ENDOCARDIAL = 0 
        RV_SEPTUM = 1
        RV_FREEWALL = 2
        EPICARDIAL =3
        MITRAL_VALVE =4
        AORTA_VALVE = 5
        TRICUSPID_VALVE = 6
        PULMONARY_VALVE = 7
        RV_INSERT = 8
        '''
       
    LVendo=et_vertex_start_end[0][0]
    LVendocoord=str(pts5810.iloc[LVendo,0])+","+str(pts5810.iloc[LVendo,1])+","+str(pts5810.iloc[LVendo,2])
    RVendo=et_vertex_start_end[2][0]
    RVendocoord=str(pts5810.iloc[RVendo,0])+","+str(pts5810.iloc[RVendo,1])+","+str(pts5810.iloc[RVendo,2])
    
    
    #extract LV_endo and RV_endo. For Rv_endo, both carp txt and .surf are written out
    meshtool.extract_surf(meshtoolloc,mesh,"LV_endo","0-4,5,6,7","-coord="+LVendocoord)
    meshtool.extract_surf(meshtoolloc,mesh,"RV_endo","0-4,5,6,7","-coord="+RVendocoord,'-ofmt=carp_txt')
    
    
    # RV_septum mapping from low res to high res. Credit to Martin Bishop.
    #create RV_septum from original surface mesh in vtk to carp_txt
    meshtool.convert_mesh(meshtoolloc, input_dir+'/'+RV_septum, RV_septum,ifmt='vtk')
    
    #generate a dummy element data file of ones for this septal surface mesh
    elems=meshIO.read_elem(RV_septum)
    data=np.ones([len(elems),1])
    np.savetxt('RV_septum_elemdata.dat',data)
    
    
    #interpolates elemdata across to high res mesh
    omsh='RV_endo.surfmesh'
    meshtool.interpolate_elemdata(meshtoolloc,RV_septum,'RV_septum_elemdata.dat',omsh,'rv_endo_septdata.dat')
    
    
    # #thresholds interpolated datafield to define septal surface on highres mesh
    #extract surface_fromdata
    
    surf=meshIO.read_surf('RV_endo')
    
    data= np.loadtxt('rv_endo_septdata.dat')
    thr=0.9
    Septum_surf_hre=surf[data>thr]
    meshIO.write_surf('RV_septumfinal',Septum_surf_hre)
    surf_vtx=np.unique(Septum_surf_hre)
    meshIO.write_vtx_File(vtxFilename='RV_septumfinal.surf', vtx=pd.DataFrame(surf_vtx))
    
    # #extract RV free wall as RV endo - RV septum
    meshtool.extract_surf(meshtoolloc,mesh,"RV_endo_FW","RV_endo-RV_septumfinal")
    
    biv_surf_i="surf_i"
    if not os.path.exists(biv_surf_i):
        os.mkdir(biv_surf_i)
    if not meshtool.map_file(meshtoolloc,mesh+"_i","*.surf",biv_surf_i):
        return None
    if not meshtool.map_file(meshtoolloc,mesh+"_i","*.vtx",biv_surf_i):
        return None
    biv_surf_i="/surf_i/"
    
    return [surfmesh+biv_surf_i+"/LV_endo",surfmesh+biv_surf_i+"/RV_septumfinal",surfmesh+biv_surf_i+"/RV_endo_FW"]
    
def split_RVLV(meshtoolloc,MPIEXEC,OPTS,CARP,parfile,mesh_nolabel,SIMID,surf_endo,IGBEXTRACT,thres,outmsh):
    
    #input SIMID:result from Laplace
    #mesh_nolabel: resampled mesh dir plus the name
    #thres: define LV and RV boundary, typically 0.7
    #pathtopar: the location of the .par for Laplace solver
    #pathtosurface: the location of the surfaces in the resampled mesh
    #output the new labeled .elem file. 
    #Labels: RV:0; LV:1; Valves:10-13
    
    args="-stimulus[0].vtx_file"+" "+surf_endo[0]+".surf -stimulus[1].vtx_file"+" "+surf_endo[1]+".surf -stimulus[2].vtx_file"+" "+surf_endo[2]+".surf"
    carpfunc.Laplace_solver(MPIEXEC,OPTS,CARP,parfile,mesh_nolabel,SIMID,args)
    
    gradMag=carpfunc.igb_todata(IGBEXTRACT,SIMID)
    
    elem=meshIO.read_elems(basename=mesh_nolabel)
    #thres=0.51
    print("Start spliting...")
    split = elem[[1, 2, 3, 4]].applymap(lambda x: x < len(gradMag) and gradMag[x] < thres)
    elem[5] = split.any(axis=1).map(int)
    #elem.loc[elem[5]==0,5]=0     #RV:0
    outmsh=mesh_nolabel+"_split"
    meshIO.write_elem(elemFilename=outmsh, elem=elem,shapes="Tt")
    print("End spliting!")
    shutil.copyfile(mesh_nolabel+".lon", outmsh+".lon")
    shutil.copyfile(mesh_nolabel+".pts", outmsh+".pts")
    shutil.copyfile(mesh_nolabel+".nod", outmsh+".nod")
    shutil.copyfile(mesh_nolabel+".eidx", outmsh+".eidx")
    
    if not meshtool.convert_mesh(meshtoolloc,outmsh,outmsh,ofmt="vtk"):
        return None
    
    print("Split LV and RV mesh is"+ outmsh)
    
    return outmsh
