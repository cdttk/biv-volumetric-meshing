#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:36:35 2022

@author: sq20
"""


import pandas as pd
import numpy as np
import os
import glob
import math
import random
import shutil

from . import meshIO
from . import meshtool_func as meshtool
from . import carpfunc
from . import geometrical

def coord_z(folder,meshtoolloc,mesh,surfloc,UVCfolder): 
    #"folder" is where the labeled mesh stored,"mesh" is the mesh name
    #"surfloc" is where to store surfaces
    #find apex point in LV only mesh
    #output is the x,y,z coord of the apex point
    os.chdir(folder)
    
    if not meshtool.extract_mesh(meshtoolloc,mesh,"1,4,5","LV_all"):
        return None
    
    if not os.path.exists(surfloc):
        os.mkdir(surfloc)
    
    if not meshtool.extract_surf(meshtoolloc,"LV_all",surfloc+"/mitral","4"):
        return None
    d_from_mitral="d_mitraltoLVall.dat"
    if not meshtool.generate_dfield(meshtoolloc,"LV_all",surfloc+"/mitral",d_from_mitral):
        return None
    
    d_LVtomitral=np.genfromtxt(d_from_mitral)
    #find the index of the furthest point from the mitral valve as apex point
    index=np.where(d_LVtomitral==np.amax(d_LVtomitral))[0]
    if not meshtool.extract_mesh(meshtoolloc,mesh,"0,1",mesh+"_i"):
        return None
    
    pts_LV=np.array(meshIO.read_pts(basename="LV_all"))  
    pts_biv=np.array(meshIO.read_pts(basename=mesh+"_i"))
    
    #find the apex point' index in biv mesh
    ptsindex_biv=np.where(np.all(pts_biv==pts_LV[index],axis=1))[0]
    
    # extract all surface in biv mesh (only LV and RV) to store in folder surfloc/biv_i
    if not os.path.exists(surfloc+"/biv_i/"):
        os.mkdir(surfloc+"/biv_i/")
    
    if not meshtool.extract_surf(meshtoolloc,mesh+"_i",surfloc+"/biv_i/myoall","0,1"):
        return None
    
    surf=meshIO.read_surf(basename=surfloc+"/biv_i/myoall")
    # find the surface including the apex point and write to mesh+"_i_apex" in both .surf and .vtx
    surfnew=surf.loc[(surf[1]==int(ptsindex_biv))|(surf[2]==int(ptsindex_biv))|(surf[3]==int(ptsindex_biv)),:]
    surf_vtx=np.unique(surfnew.loc[:,1:3])
    
    Filename=surfloc+"/biv_i/apex_i"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    
    #extract base surf in myo only mesh which is the intersection region of valves with myo
    #first extract the "biv_i_all-biv_allvalves" in the whole mesh and map it to biv myo only mesh
    if not meshtool.extract_surf(meshtoolloc,mesh,surfloc+"/intersection","0,1:4,5,6,7"):
        return None
    if not meshtool.map_file(meshtoolloc,mesh+"_i",surfloc+"/intersection.surf",surfloc+"/biv_i"):
        return None
    if not meshtool.map_file(meshtoolloc,mesh+"_i",surfloc+"/intersection.surf.vtx",surfloc+"/biv_i"):
        return None

    # calculate the distance from base to apex and store the results in 
    apexsurf=surfloc+"/biv_i/apex_i"
    basesurf=surfloc+"/biv_i/intersection"
    outputFilename=UVCfolder+"/COORDS_Z.dat"
    
    if not meshtool.generate_dfield(meshtoolloc,mesh+"_i",apexsurf,outputFilename,"-esurf="+basesurf):
        return None
    
    return outputFilename

def transmural(folder,meshtoolloc,mesh,pts5810,surfloc,MPIEXEC,OPTS,CARP,parfile,SIMID,IGBEXTRACT,UVCfolder):
    #
    os.chdir(folder)
    
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
    RVseptum=et_vertex_start_end[1][0]
    RVseptumcoord=str(pts5810.iloc[RVseptum,0])+","+str(pts5810.iloc[RVseptum,1])+","+str(pts5810.iloc[RVseptum,2])
    epi=et_vertex_start_end[3][0]
    epicoord=str(pts5810.iloc[epi,0])+","+str(pts5810.iloc[epi,1])+","+str(pts5810.iloc[epi,2])
    
    surf_iloc=surfloc+"/biv_i/"
    op=surf_iloc+"myoall-"+surf_iloc+"intersection"
    
    LV_endo = surf_iloc + 'LV_endo'
    RV_septum = surf_iloc + 'RV_septum'
    RV_endo = surf_iloc + 'RV_endo'
    epi = surf_iloc + 'epi'
    
    if not meshtool.extract_surf(meshtoolloc,mesh+"_i",surf_iloc+"LV_endo",op,"-coord="+LVendocoord):
        return None
    if not meshtool.extract_surf(meshtoolloc,mesh+"_i",surf_iloc+"RV_septum",surf_iloc+'myoall_insertion',"-coord="+RVseptumcoord,"-edge=10"):
        return None
    if not meshtool.extract_surf(meshtoolloc,mesh+"_i",surf_iloc+"RV_endo1",op,"-coord="+RVendocoord,"-edge=10"):
        return None
    if not meshtool.extract_surf(meshtoolloc,mesh+"_i",surf_iloc+"RV_endo",surf_iloc+"RV_endo1-"+surf_iloc+'RV_septum',"-coord="+RVendocoord):
        return None
    if not meshtool.extract_surf(meshtoolloc,mesh+"_i",surf_iloc+"epi",op,"-coord="+epicoord):
        return None
    
    args="-stimulus[0].vtx_file "+LV_endo+".surf "
    args=args+" -stimulus[1].vtx_file "+RV_endo+".surf "
    args=args+" -stimulus[2].vtx_file "+epi+".surf "
    args=args+" -stimulus[3].vtx_file "+RV_septum+".surf "
    # LV,RV_septum, RV_endo and epi Laplace solver
    if not carpfunc.Laplace_solver(MPIEXEC,OPTS,CARP,parfile,mesh+"_i",SIMID,args):
        return None
    
    gradMag=carpfunc.igb_todata(IGBEXTRACT,SIMID)
    file=UVCfolder+"/PHO.dat"
    meshIO.write_data(dataFilename=file, data=gradMag)
    # print (max(gradMag))
    # print (min(gradMag))
    return file
    
def find_two_RV_insertion_pts(folder,meshname, surfloc,meshtoolloc,coordzdir):
    #this fuc is to find two pts in RV_insertion by 
    #(1)finding the anterior point which is the closest point to the pulmonary valve
    #(2)finding the interior point which is the furtherest point to ANT in RV insertion surf and also have same z
    #ANT and INF is wrong. it needs to switch for later steps!!!
    os.chdir(folder)
    LVmyosurf=surfloc+"/LVmyo"
    if not os.path.exists(LVmyosurf):
        os.mkdir(LVmyosurf)

    LV = surfloc + '/LV'
    if not meshtool.extract_surf(meshtoolloc,meshname,surfloc+"/LV","1"):
        return False
    
    RVinsertion = surfloc + '/RVinsertion'
    if not meshtool.extract_surf(meshtoolloc,meshname,surfloc+"/RVinsertion","0:1"):
        return False
    if not meshtool.extract_surf(meshtoolloc,meshname+'_i',surfloc+"/biv_i/RVinsertion","0:1"):
        return False
    # if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf",surfloc+"/LVmyo"):
    #     return False
    # if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf.vtx",surfloc+"/LVmyo"):
    #     return False
    
    
    LVvtx=meshIO.read_vtx_File(surfloc+"/LV.surf.vtx")
    LVsurf=meshIO.read_surf(surfloc+"/LV")
    pts=meshIO.read_pts(meshname)
    coord_LV= np.zeros((len(LVvtx),3))
    for i in range(len(LVvtx)):
        coord_LV[i]=pts.loc[int(LVvtx[i])]
    #######################
    #find the centre point of Pulmonary valve
    
    if not meshtool.extract_mesh(meshtoolloc,meshname,"7","pulmonarymesh"):
        return False
    pulmonary_pts=meshIO.read_pts(folder+"/pulmonarymesh")
    pulmonary_centre=np.array(np.mean(pulmonary_pts, axis = 0))
    ####################
    
    distances = np.linalg.norm(coord_LV-pulmonary_centre, axis=1)
    NodeANT= int(LVvtx[np.argmin(distances)])
    ######################################
    #print this out for visualizing but not necesseary
    surfnew=LVsurf.loc[(LVsurf[1]==NodeANT)|(LVsurf[2]==NodeANT)|(LVsurf[3]==NodeANT),:]
    surf_vtx=np.unique(surfnew.loc[:,1:3])
    Filename=surfloc+"/Node_ANT"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    ######################################
    
    
    #################################
    #Node_INF is identified by 1) find all surfaces in RV insertion having similar coordz as Node_ANT
    # 2) In those surfaces, find the furtherest point from the Node_ANT
    #note coordz.dat is for biv_i mesh, so suface need to be mapped
    
    #map to biv_i mesh
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/Node_ANT.vtx",surfloc+"/biv_i"):
        return False
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/Node_ANT.surf",surfloc+"/biv_i"):
        return False

    RVinsertion = surfloc + '/RVinsertion'
    if not meshtool.extract_surf(meshtoolloc,meshname,surfloc+"/RVinsertion","0:1"):
        return False
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf",surfloc+"/biv_i"):
        return False
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/RVinsertion.surf.vtx",surfloc+"/biv_i"):
        return False
    coordz=np.genfromtxt(coordzdir)
    
    ANTvtx=meshIO.read_vtx_File(surfloc+"/biv_i/Node_ANT.vtx")
    ANTsurf=meshIO.read_surf(surfloc+"/biv_i/Node_ANT")
    
    ANT_coordz=coordz[int(ANTvtx[0])]
    
    RVinsertionvtx=meshIO.read_vtx_File(surfloc+"/biv_i/RVinsertion.surf.vtx")
    RVinsertion_coordz=np.zeros(len(RVinsertionvtx))
    vtx_similarz=np.empty((0,1),int)
    
    for i in range(len(RVinsertionvtx)):
        if coordz[int(RVinsertionvtx[i])]<(ANT_coordz+0.005) and coordz[int(RVinsertionvtx[i])]>(ANT_coordz-0.005):
            vtx_similarz=np.append(vtx_similarz,[RVinsertionvtx[i]])
            
        
        
    RVinsertionsurf=meshIO.read_surf(surfloc+"/biv_i/RVinsertion")
    pts=meshIO.read_pts(meshname+"_i")
    coordz_ANT=np.array(pts.loc[ANTvtx[0]])
    coord_similarz= np.zeros((len(vtx_similarz),3))
    for i in range(len(vtx_similarz)):
        coord_similarz[i]=pts.loc[int(vtx_similarz[i])]
    
    distances = np.linalg.norm(coord_similarz-coordz_ANT, axis=1)
    if len(distances) == 0:
        return False
    NodeINF= int(vtx_similarz[np.argmax(distances)])
    
    ######################################
    #print this out for visualizing but not necesseary
    surfnew=RVinsertionsurf.loc[(RVinsertionsurf[1]==NodeINF)|(RVinsertionsurf[2]==NodeINF)|(RVinsertionsurf[3]==NodeINF),:]
    surf_vtx=np.unique(surfnew.loc[:,1:3])
    Filename=surfloc+"/biv_i/Node_INF"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    ######################################
    
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/biv_i/Node_INF.surf",surfloc,"s2m"):
        return False
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/biv_i/Node_INF.surf.vtx",surfloc,"s2m"):
        return False

    return True

 #########################################
 # Function to preprocessing data for rotational fuc for LV
 #########################################   
def extract_surf_forLV(folder,meshtoolloc,meshname,surfloc):
    #extract surface: apex,base(mitral_intersect), Node_ANT, Node_INF
    os.chdir(folder)
    
    LVmyosurf=surfloc+"/LVmyo"
    if not os.path.exists(LVmyosurf):
        os.mkdir(LVmyosurf)
        
    if not meshtool.extract_mesh(meshtoolloc,meshname+"_i","1","LVmyo_i"):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/apex_i.surf",LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/apex_i.vtx",LVmyosurf):
        return None
    
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/RVinsertion.surf",LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/RVinsertion.vtx",LVmyosurf):
        return None
    
    if not meshtool.extract_surf(meshtoolloc, "LVmyo_i",LVmyosurf+'/myoall_insertion','1-'+LVmyosurf+'/RVinsertion'):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",LVmyosurf+'/myoall_insertion.surf',surfloc+"/biv_i",mode='s2m'):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",LVmyosurf+'/myoall_insertion.vtx',surfloc+"/biv_i",mode='s2m'):
        return None
   
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/Node_ANT.surf",LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/Node_ANT.vtx",LVmyosurf):
        return None

    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/Node_INF.surf",LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo_i",surfloc+"/biv_i/Node_INF.vtx",LVmyosurf):
        return None
    
    #map the base surf from whole mesh as the mitral valve intersection with the LVmyo
    if not meshtool.extract_mesh(meshtoolloc,meshname,"1","LVmyo"):
        return None
    mitral_intersect = surfloc + '/mitral_intersect'
    if not meshtool.extract_surf(meshtoolloc,meshname,surfloc+"/mitral_intersect","1:4"):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo",surfloc+"/mitral_intersect.surf",LVmyosurf):
        return None
    if not meshtool.map_file(meshtoolloc,"LVmyo",surfloc+"/mitral_intersect.surf.vtx",LVmyosurf):
        return None
    
    Node_ANT=meshIO.read_vtx_File(LVmyosurf+"/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        return None
    NodeANTvtx= int(Node_ANT[0])
    
    Node_INF=meshIO.read_vtx_File(LVmyosurf+"/Node_INF.vtx")
    if len(Node_INF) == 0:
        return None
    NodeINFvtx= int(Node_INF[0])
    return [LVmyosurf+"/mitral_intersect",LVmyosurf+"/apex_i.vtx",NodeINFvtx,NodeANTvtx]


#########################################
# Function to preprocessing data for rotational fuc for RV
#########################################
def extract_surf_forRV(folder,meshtoolloc,meshname,surfloc,coordzdir):
    
    os.chdir(folder)
    RVmyosurf=surfloc+"/RVmyo"
    if not os.path.exists(RVmyosurf):
        os.mkdir(RVmyosurf)
    
    if not meshtool.extract_mesh(meshtoolloc,meshname+"_i","0","RVmyo_i"):
        return None
    
    RV_endo = surfloc + '/RV_all'
    if not meshtool.extract_surf(meshtoolloc,meshname+"_i",surfloc+"/RV_all","0"):
        return None
    
    coordz=np.genfromtxt(coordzdir)
    #Node_RVapex=np.argmin(coordz)
    
    RVvtx=meshIO.read_vtx_File(surfloc+"/RV_all.surf.vtx")
    RVsurf=meshIO.read_surf(surfloc+"/RV_all")
    
    coordz_RV= np.zeros((len(RVvtx),1))
    for i in range(len(RVvtx)):
        coordz_RV[i]=coordz[int(RVvtx[i])]
        
    Node_RVapex=int(RVvtx[np.argmin(coordz_RV)])
    ######################################
    #print this out for visualizing but not necesseary
    surfnew=RVsurf.loc[(RVsurf[1]==Node_RVapex)|(RVsurf[2]==Node_RVapex)|(RVsurf[3]==Node_RVapex),:]
    surf_vtx=np.unique(surfnew.loc[:,1:3])
    Filename=surfloc+"/Node_RV_apex"
    meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
    meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
    
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/Node_RV_apex.surf",surfloc+"/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/Node_RV_apex.vtx",surfloc+"/RVmyo"):
        return None

    tricuspid_intersect = surfloc + '/tricuspid_intersect'
    if not meshtool.extract_surf(meshtoolloc,meshname,surfloc+"/tricuspid_intersect","0:6"):
        return None
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/tricuspid_intersect.surf",surfloc+"/biv_i"):
        return None
    if not meshtool.map_file(meshtoolloc,meshname+"_i",surfloc+"/tricuspid_intersect.surf.vtx",surfloc+"/biv_i"):
        return None
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/biv_i/tricuspid_intersect.surf",surfloc+"/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/biv_i/tricuspid_intersect.surf.vtx",surfloc+"/RVmyo"):
        return None
    
    
    
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/biv_i/Node_INF.surf",surfloc+"/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/biv_i/Node_INF.vtx",surfloc+"/RVmyo"):
        return None
    
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/biv_i/Node_ANT.surf",surfloc+"/RVmyo"):
        return None
    if not meshtool.map_file(meshtoolloc,"RVmyo_i",surfloc+"/biv_i/Node_ANT.vtx",surfloc+"/RVmyo"):
        return None
    
    RVsurfdir=surfloc+"/RVmyo"
    Node_ANT=meshIO.read_vtx_File(RVsurfdir+"/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        ########################
        #find the closest point in RVmyo_i to ANT node in biv_i mesh 
        ptsbiv=meshIO.read_pts(meshname+'_i')
        bivvtx=meshIO.read_vtx_File(surfloc+"/biv_i/Node_ANT.vtx")
        ptscoordANT=ptsbiv.loc[int(bivvtx[0])]
        
        if not meshtool.extract_surf(meshtoolloc,"RVmyo_i",surfloc+"/RVmyo/myoall","0"):
            return None
        
        ptsRV=meshIO.read_pts('RVmyo_i')
        RVvtx=meshIO.read_vtx_File(surfloc+"/RVmyo/myoall.surf.vtx")
        RVsurf=meshIO.read_surf(surfloc+"/RVmyo/myoall")
        ptscoordRV=ptsRV.loc[RVvtx]
        
        distances = np.linalg.norm(ptscoordRV-ptscoordANT, axis=1)
        NodeANT= int(RVvtx[np.argmin(distances)])
        
        ######################################
        #print this out for visualizing but not necesseary
        surfnew=RVsurf.loc[(RVsurf[1]==NodeANT)|(RVsurf[2]==NodeANT)|(RVsurf[3]==NodeANT),:]
        surf_vtx=np.unique(surfnew.loc[:,1:3])
        Filename=surfloc+"/RVmyo/Node_ANT"
        meshIO.write_vtx_File(vtxFilename=Filename, vtx=pd.DataFrame(surf_vtx))
        meshIO.write_surf(surfFilename=Filename, surf=pd.DataFrame(surfnew), shapes=None)
        RVsurfdir=surfloc+"/RVmyo"
        Node_ANT=meshIO.read_vtx_File(RVsurfdir+"/Node_ANT.vtx")
    if len(Node_ANT) == 0:
        return None
    NodeANTvtx= int(Node_ANT[0])
    
    Node_INF=meshIO.read_vtx_File(RVsurfdir+"/Node_INF.vtx")
    if len(Node_INF) == 0:
        return None
    NodeINFvtx= int(Node_INF[0])
    
    return [surfloc+"/RVmyo/tricuspid_intersect", surfloc+"/RVmyo/Node_RV_apex.vtx", NodeINFvtx,NodeANTvtx]



#########################################
# Function to compute rotationall UVC coordinate by Martin
#########################################
def compute_uvc_rotational(folder,meshname,base_surfFilename,apex_pointFilename,RVinsertionNodeANT,RVinsertionNodeINF,phiname):
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apex_pointFilename)
    apex_vtx = int(apex_vtx[0])
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]
    
    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    centroids = meshIO.create_centroids(elems, pts)
    
    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    base_surf = meshIO.read_surf(base_surfFilename)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(),pts.loc[base_nodes][1].mean(),pts.loc[base_nodes][2].mean()]
    
    # Computes long axis between apex and CoM 
    longAxis = (apex_coord - CoM_coord)/np.linalg.norm(apex_coord-CoM_coord)
    
    # Translates mesh to have x,y point aligned with apex
    pts_t = pts - apex_coord
    
    # Rotates mesh to align long axis with z [001] direction
    z_axis = [0,0,1]
    R = geometrical.computeRotationMatrix(z_axis,longAxis)
    pts_r = np.transpose(np.matmul(R,np.transpose(pts_t)))
    
    
    #########################################
    # Computes phi based on relative angles
    #########################################
    # Defines 2 RV insertion sites and coords
    RVinsPtANT_coord = pts_r.iloc[RVinsertionNodeANT]
    RVinsPtINF_coord = pts_r.iloc[RVinsertionNodeINF]
    
    # Picks out the xy coord (projecting it into xy plane)
    zero_phase_RVANT = [RVinsPtANT_coord[0],RVinsPtANT_coord[1]]/np.linalg.norm([RVinsPtANT_coord[0],RVinsPtANT_coord[1]])
    zero_phase_RVINF = [RVinsPtINF_coord[0],RVinsPtINF_coord[1]]/np.linalg.norm([RVinsPtINF_coord[0],RVinsPtINF_coord[1]])
    
    # find singularity
    sing_phase = -0.5*(zero_phase_RVANT + zero_phase_RVINF)
    sing_phase = sing_phase/np.linalg.norm(sing_phase)
    
    # Projects all points in mesh into xy plane
    pts_proj = pts_r.loc[:,0:1]
    mag_pts = np.linalg.norm(pts_proj,axis=1)
    
    # computes the angles between every point in the mesh and...
    # ... anterior RV insertion site
    cosThetas_RVANT = np.dot(pts_proj,zero_phase_RVANT)/mag_pts
    phi_RVANT = np.arccos(cosThetas_RVANT)
    # ... Interior RV insertion site
    cosThetas_RVINF = np.dot(pts_proj,zero_phase_RVINF)/mag_pts
    phi_RVINF = np.arccos(cosThetas_RVINF)
    # ... centre of free wall
    cosThetas_SING = np.dot(pts_proj,sing_phase)/mag_pts
    phi_SING = np.arccos(cosThetas_SING)
    
    # Defines array to store phi coords
    phi_coord = np.zeros(len(pts))
    
    # defines anterior nodes and interior nodes
    ant_LV_nodes = np.where(phi_RVANT<phi_RVINF)
    inf_LV_nodes = np.where(phi_RVANT>=phi_RVINF)
    
    phi_coord[ant_LV_nodes] = -np.pi/2 - (np.pi/2)*phi_RVANT[ant_LV_nodes]/(phi_RVANT[ant_LV_nodes]+phi_SING[ant_LV_nodes])
    phi_coord[inf_LV_nodes] = np.pi/2 + (np.pi/2)*phi_RVINF[inf_LV_nodes]/(phi_RVINF[inf_LV_nodes]+phi_SING[inf_LV_nodes])
    
    #############################
    # Computes phi for septum
    #############################
    septal_angle = np.arccos(np.dot(zero_phase_RVANT,zero_phase_RVINF))
    septal_nodes = np.intersect1d(np.intersect1d(np.where(phi_RVANT<septal_angle),np.where(phi_RVINF<septal_angle)),np.where(phi_SING>np.pi/2))
    phi_coord[septal_nodes] = np.pi/2 - np.pi*(phi_RVINF[septal_nodes]/(phi_RVANT[septal_nodes]+phi_RVINF[septal_nodes]))
    meshIO.write_data(dataFilename=phiname, data=phi_coord)     
    
    return folder+"/"+phiname

########################################
# Function to compute rotationall UVC coordinate by Martin
#########################################
def compute_uvc_rotationalRV(folder,meshname,base_surfFilename,apex_pointFilename,RVinsertionNodeANT,RVinsertionNodeINF,phiname):
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apex_pointFilename)
    apex_vtx = int(apex_vtx[0])
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]
    
    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    centroids = meshIO.create_centroids(elems, pts)
    
    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    base_surf = meshIO.read_surf(base_surfFilename)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(),pts.loc[base_nodes][1].mean(),pts.loc[base_nodes][2].mean()]
    
    # Computes long axis between apex and CoM 
    longAxis = (apex_coord - CoM_coord)/np.linalg.norm(apex_coord-CoM_coord)
    
    # Translates mesh to have x,y point aligned with apex
    pts_t = pts - apex_coord
    
    # Rotates mesh to align long axis with z [001] direction
    z_axis = [0,0,1]
    R = geometrical.computeRotationMatrix(z_axis,longAxis)
    pts_r = np.transpose(np.matmul(R,np.transpose(pts_t)))
    
    
    #########################################
    # Computes phi based on relative angles
    #########################################
    # Defines 2 RV insertion sites and coords
    RVinsPtANT_coord = pts_r.iloc[RVinsertionNodeANT]
    RVinsPtINF_coord = pts_r.iloc[RVinsertionNodeINF]
    
    # Picks out the xy coord (projecting it into xy plane)
    zero_phase_RVANT = [RVinsPtANT_coord[0],RVinsPtANT_coord[1]]/np.linalg.norm([RVinsPtANT_coord[0],RVinsPtANT_coord[1]])
    zero_phase_RVINF = [RVinsPtINF_coord[0],RVinsPtINF_coord[1]]/np.linalg.norm([RVinsPtINF_coord[0],RVinsPtINF_coord[1]])
    
    # find singularity
    sing_phase = -0.5*(zero_phase_RVANT + zero_phase_RVINF)
    sing_phase = sing_phase/np.linalg.norm(sing_phase)
    
    # Projects all points in mesh into xy plane
    pts_proj = pts_r.loc[:,0:1]
    mag_pts = np.linalg.norm(pts_proj,axis=1)
    
    # computes the angles between every point in the mesh and...
    # ... anterior RV insertion site
    cosThetas_RVANT = np.dot(pts_proj,zero_phase_RVANT)/mag_pts
    phi_RVANT = np.arccos(cosThetas_RVANT)
    # ... Interior RV insertion site
    cosThetas_RVINF = np.dot(pts_proj,zero_phase_RVINF)/mag_pts
    phi_RVINF = np.arccos(cosThetas_RVINF)
    
    # Defines array to store phi coords
    phi_coord = np.zeros(len(pts))
    
    # defines anterior nodes and interior nodes
    ant_LV_nodes = np.where(phi_RVANT<phi_RVINF)
    inf_LV_nodes = np.where(phi_RVANT>=phi_RVINF)
    
    phi_coord = np.pi/2 - np.pi*(phi_RVINF/(phi_RVANT+phi_RVINF))
    meshIO.write_data(dataFilename=phiname, data=phi_coord)     
    
    return folder+"/"+phiname


def merge_PHI(folder,meshtoolloc,meshname,phiLV,phiRV,surfloc,UVCfolder):
    os.chdir(folder)
    if not meshtool.insert_data(meshtoolloc,'LVmyo_i',phiLV,meshname+'_i','Phi_insertLV.dat','0'):
        return None
    if not meshtool.insert_data(meshtoolloc,'RVmyo_i',phiRV,meshname+'_i','Phi_insertRV.dat','0'):
        return None
    Phi_insertLV=np.genfromtxt('Phi_insertLV.dat')
    Phi_insertRV=np.genfromtxt('Phi_insertRV.dat')
    if not meshtool.extract_surf(meshtoolloc,meshname+'_i',surfloc+"/biv_i/RVinsertion","0:1"):
        return None
    RVinsertionvtx=meshIO.read_vtx_File(surfloc+"/biv_i/RVinsertion.surf.vtx")
    RVinsertionvtx=RVinsertionvtx.astype(int)
    Phi=Phi_insertLV+Phi_insertRV
    Phi[RVinsertionvtx]=Phi_insertLV[RVinsertionvtx]
    meshIO.write_data(dataFilename=UVCfolder+'/PHI.dat', data=Phi)
    return UVCfolder+'/PHI.dat'
    


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])




# This function is a modified version of the original by Martin Bishop
def compute_fiber_new(folder,meshtoolloc,meshname,apexvtx,base_surf,PHOname,coordszname,alpha_epi,alpha_endo,beta_epi,beta_endo):
    '''
    return the mesh with longitudinal fiber according to alpha_epi and alpha_endo (usually:pi/3 and -pi/3)
    the 'meshname' is only LV or RV
    '''
    os.chdir(folder)
    # Reads in apex point vtx electrode
    apex_vtx = meshIO.read_vtx_File(apexvtx)
    apex_vtx = int(apex_vtx[0])
    #meshname="LVmyo"
    pts = meshIO.read_pts(meshname)
    apex_coord = pts.iloc[apex_vtx]
    
    # Reads in elems and computes centroids
    elems = meshIO.read_elems(meshname)
    #centroids = meshIO.create_centroids(elems, pts)
    
    ##########################################
    # Rotates mesh to align with z axis
    ##########################################
    # Computes centre of mass of base of LV
    #base_surf=predata[0]
    base_surf = meshIO.read_surf(base_surf)
    base_nodes = meshIO.read_surf_to_nodeList(base_surf)
    CoM_coord = [pts.loc[base_nodes][0].mean(),pts.loc[base_nodes][1].mean(),pts.loc[base_nodes][2].mean()]
    
    # Computes long axis between apex and CoM 
    longAxis = (apex_coord - CoM_coord)/np.linalg.norm(apex_coord-CoM_coord)
    #extract the PHO and COORDS_Z for the specific mesh (LV or RV) to a new .dat
    if not meshtool.extract_data(meshtoolloc,meshname,PHOname,'PHO_'+meshname+'.dat',"0"):
        return None
    if not meshtool.extract_data(meshtoolloc,meshname,coordszname,'COORDS_Z'+meshname+'.dat',"0"):
        return None
    #calculate the gradient of the PHO and COORDS_Z
    
    if not meshtool.extract_gradient(meshtoolloc, meshname, 'COORDS_Z'+meshname+'.dat', 'COORDS_Z_'+meshname+'_gradient', '1'):
        return None
    if not meshtool.extract_gradient(meshtoolloc, meshname, 'PHO_'+meshname+'.dat', 'PHO_'+meshname+'_gradient', '1'):
        return None    
    
    transmuralFile='PHO_'+meshname+'.dat'
    coordz_vec=np.genfromtxt('COORDS_Z_'+meshname+'_gradient.grad.vec')
    transmural_vec=np.genfromtxt('PHO_'+meshname+'_gradient.grad.vec')
    print('write in all three files!')
    #calculate the fiber along the myo surface (circumferential direction)
    circumferential=np.cross(transmural_vec,-coordz_vec)
        
    #calculate the PHO for each element by averaging the PHO for four pts
    transmural=np.genfromtxt(transmuralFile)
    trans_n0 = transmural[elems.iloc[:,0]]
    trans_n1 = transmural[elems.iloc[:,1]]
    trans_n2 = transmural[elems.iloc[:,2]]
    trans_n3 = transmural[elems.iloc[:,3]]
    mean_trans = (trans_n0 + trans_n1 + trans_n2 + trans_n3)*0.25
    
    alpha_trans = np.zeros(len(circumferential))
    beta_trans = np.zeros(len(circumferential))
    
    lon_longitudinal=circumferential #to initialize the sheet lon vector 
    lonsheet=circumferential #to initialize the sheet lon vector 
    
    lonall=np.zeros((len(circumferential),8))
    lonall[:]=np.nan
    for i in range(len(circumferential)):
       
        alpha_trans[i]=-((alpha_endo-alpha_epi))*mean_trans[i]+alpha_endo
        #rotate the fiber vectors by alpha_trans (in radian) along transmural_vec
        lon_longitudinal[i]=np.dot(rotation_matrix(transmural_vec[i], alpha_trans[i]), circumferential[i])
        lon_longitudinal[i]=lon_longitudinal[i]/np.linalg.norm(lon_longitudinal[i])
        lonall[i,0:3]=lon_longitudinal[i]
        
    for i in range(len(circumferential)):
        beta_trans[i]=-((beta_endo-beta_epi))*mean_trans[i]+beta_endo
        lonsheet[i]=np.dot(rotation_matrix(circumferential[i], beta_trans[i]), transmural_vec[i])
        lonsheet[i]=lonsheet[i]/np.linalg.norm(lonsheet[i])
        lonall[i,5:8]=lonsheet[i]
     
    meshIO.write_lon_includesheet(lonFilename=meshname+'_fiber', lon=pd.DataFrame(lonall))
    meshIO.write_lon(lonFilename=meshname+'_longitudinal_fiber', lon=pd.DataFrame(lon_longitudinal))
    meshIO.write_lon(lonFilename=meshname+'_sheet_fiber', lon=pd.DataFrame(lonsheet))
    #meshIO.write_lon(lonFilename=meshname+'_fiber', lon=pd.DataFrame(lon_longitudinal))
    
    shutil.copyfile(meshname+".elem", meshname+"_fiber.elem")
    shutil.copyfile(meshname+".pts", meshname+"_fiber.pts")
    if not meshtool.convert_mesh(meshtoolloc,meshname+"_fiber",meshname+"_fiber",ofmt="vtk"):
        return None
    return meshname+"_fiber"
    

def merge_fibertomyoonly(folder,meshtoolloc,meshname,fiberfolder):
    
    os.chdir(folder)
    meshall_fiber=meshIO.read_fibres(basename=meshname+'_i')
    lonall=np.zeros((len(meshall_fiber.index),8))
    meshIO.write_lon_includesheet(lonFilename=meshname+'_i', lon=pd.DataFrame(lonall))
    submesh='LVmyo_i'
    shutil.copyfile(submesh+'.eidx', submesh+'_fiber.eidx')
    shutil.copyfile(submesh+'.nod', submesh+'_fiber.nod')    
    if not meshtool.insert_submesh(meshtoolloc,submesh+'_fiber',meshname+'_i',meshname+'_insertLV'):
        return False
    if not meshtool.convert_mesh(meshtoolloc,meshname+'_insertLV',meshname+'_insertLV',ofmt="vtk"):
        return False
    submesh='RVmyo_i'
    shutil.copyfile(submesh+'.eidx', submesh+'_fiber.eidx')
    shutil.copyfile(submesh+'.nod', submesh+'_fiber.nod')    
    finalmesh_i=meshname+'_i_fiber'
    if not meshtool.insert_submesh(meshtoolloc,submesh+'_fiber',meshname+'_insertLV',finalmesh_i):
        return False
    if not meshtool.convert_mesh(meshtoolloc,finalmesh_i,finalmesh_i,ofmt="vtk"):
        return False
    
    submesh=meshname+'_i'
    shutil.copyfile(submesh+'.eidx', finalmesh_i+'.eidx')
    shutil.copyfile(submesh+'.nod', finalmesh_i+'.nod') 
    
    meshall=meshIO.read_fibres(basename=meshname)
    lonall1=np.zeros((len(meshall.index),8))
    meshIO.write_lon_includesheet(lonFilename=meshname, lon=pd.DataFrame(lonall1))
    
    finalmesh=fiberfolder+'/'+meshname+'_fiber'
    if not meshtool.insert_submesh(meshtoolloc,finalmesh_i,meshname,finalmesh):
        return False
    meshall_pts=meshIO.read_pts(basename=meshname)
    #change pts unit from 'mm' to 'um' to fit for carp simulations
    meshall_pts=meshIO.write_pts(ptsFilename=finalmesh, pts=meshall_pts*1000)
    if not meshtool.convert_mesh(meshtoolloc,finalmesh,finalmesh,ofmt="vtk"):
        return False

    return True
    
    
def bivlabel(folder,meshname,UVCfolder):
    #make a dat file to seperate LV and RV
    #LV: -1, RV:1
    os.chdir(folder)
    pts=meshIO.read_pts(basename=meshname+'_i')
    elem=meshIO.read_elems(basename=meshname+'_i')
    
    bivcoord=np.ones(len(pts))
    for i in range(len(elem)):
        if elem.loc[i,5]==1:
            bivcoord[elem.loc[i,1]]=-1
            bivcoord[elem.loc[i,2]]=-1
            bivcoord[elem.loc[i,3]]=-1
            bivcoord[elem.loc[i,4]]=-1
    meshIO.write_list(dataFilename=UVCfolder+'/biv.dat', data=bivcoord)
    return UVCfolder+'/biv.dat'
