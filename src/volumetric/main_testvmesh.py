#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 22:36:00 2022

@author: sq20
"""
import argparse
import configparser
import pandas as pd
import numpy as np
import pyvista as pv
import os
import itertools
import logging
import math
import random
import re
import shutil
import json

from . import carpfunc
from . import compute_UVC_fiber as UVC
from . import geometrical
from . import meshIO
from . import meshtool_func as meshtool
from . import py_atrial_fibres as paf
from . import v_mesh_generation as vmesh

logger = logging.getLogger('volumetric')

CARP_BIN = None
OPTS = None

def get_timeframe_n(filename):
    return int(re.search(r'timeframe(\d+)', filename).group(1))

def timeframes(subject_id, input_dir):
    vtk_files = [f for f in os.listdir(input_dir) if re.search(r'timeframe\d+\.vtk$', f)]

    # sort by timeframe number then alphabetically
    vtk_files.sort(key=lambda s: (get_timeframe_n(s), s))

    for n, vtk_files_n in itertools.groupby(vtk_files, get_timeframe_n):
        mesh_names = [filename[:-4] for filename in vtk_files_n]

        if len(mesh_names) != 8:
            logger.error('subject {}, {}: incorrect number of surface meshes'.format(subject_id, n))
            continue

        yield n, mesh_names

def run_timeframe(subject_id, timeframe_num, input_dir, mesh_names, par_file, output_dir):
    #inputs: 8 surface mesh in .vtk and LVendo_RVseptum_RVendo.par

    logger.debug('subject {}, timeframe {}: started mesh'.format(subject_id, timeframe_num))
    folder = output_dir
    meshtoolloc = os.path.join(CARP_BIN, 'meshtool')
    surfacemesh = mesh_names
    surfacemesh=sorted(mesh_names)
    
    #os.mkdir("temp")
    #surfacemesh list is:
    #    0: LV_endo
    #    1: RV_FW
    #    2: RV_septum
    #    3: aorta_valve
    #    4: epi
    #    5: mitral_valve
    #    6: pulmonary_valve
    #    7: tricuspid_valve
    
    
    '''vertices/points index of original surface mesh are defined in the following order:
       
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
    
    #print (surfacemesh)
    os.chdir(folder)

    valves = {'aortamesh': (3, 0, 5), 'mitralmesh' : (5, 0, 4), 'pulmonarymesh' : (6, 1, 7), 'tricuspidmesh': (7, 1, 6)}
    boundaries = [0.5, 0.2, 0.1, 0.01, 1, 1.5]

    for valve, (valve_name, endo_name, pt_index) in valves.items():
        path = os.path.join(output_dir, valve)
        success = False
        for boundary in map(str, boundaries):
            if os.path.isdir(path):
                shutil.rmtree(path)
            if vmesh.generate_valve(meshtoolloc, input_dir, folder, surfacemesh[valve_name], surfacemesh[endo_name], valve, boundary, pt_index):
                success = True
                break
        if not success:
            logger.error('subject {}, timeframe {}: failed to generate {}'.format(subject_id, timeframe_num, valve))
            return
    
    Valves=['aortamesh', 'mitralmesh', 'pulmonarymesh', 'tricuspidmesh']
    outmsh="mergeandresample"
    if not vmesh.merge_resample(meshtoolloc,input_dir,folder,surfacemesh[0],surfacemesh[1],surfacemesh[2],surfacemesh[4],Valves,'0.3','1.5',outmsh):
        logger.error('subject {}, timeframe {}: failed to merge and resample'.format(subject_id, timeframe_num))
        return
    pts5810=meshIO.read_pts(basename=folder+"/"+Valves[0]+"/"+surfacemesh[3])
    surfmeshloc="surfbeforesplit"
    RV_septum=surfacemesh[2]
    surf_endo=vmesh.extract_surfacenolabel(meshtoolloc,folder+"/"+outmsh+"/"+outmsh, pts5810,surfmeshloc,input_dir,RV_septum)
    
    if surf_endo == None:
        logger.error('subject {}, timeframe {}: failed to extract surface'.format(subject_id, timeframe_num))
        return
    os.chdir(folder)
    
    
    CARP = os.path.join(CARP_BIN, 'openCARP')
    SIMID=os.path.join(folder,"LV_RV_split_results")
    
    mesh_nolabel=folder+"/"+outmsh+"/"+outmsh
    thres=0.51
    IGBEXTRACT = os.path.join(CARP_BIN, 'igbextract')
    MPIEXEC = os.path.join(CARP_BIN, 'mpiexec')
    split_myo_only=outmsh+"_i_split"
    splitmeshdir=vmesh.split_RVLV(meshtoolloc,MPIEXEC,OPTS,CARP,par_file,mesh_nolabel+"_i",SIMID,surf_endo,IGBEXTRACT,thres,split_myo_only)
    if not splitmeshdir:
        logger.error('subject {}, timeframe {}: failed to split ventricles'.format(subject_id, timeframe_num))
        return

    #save the final mesh to "/mesh_all/mesh_all"
    finalmeshdir=folder+"/"+subject_id+"_mesh_all"
    if not os.path.exists(finalmeshdir):
        os.mkdir(finalmeshdir)
    
    meshtool.insert_submesh(meshtoolloc,splitmeshdir,mesh_nolabel,finalmeshdir+'/'+subject_id+"_mesh_all")
    if not meshtool.convert_mesh(meshtoolloc,finalmeshdir+'/'+subject_id+"_mesh_all",finalmeshdir+'/'+subject_id+"_mesh_all",ofmt="vtk"):
        logger.error('subject {}, timeframe {}: failed to generate final mesh'.format(subject_id, timeframe_num))
        return

    edge_info = os.path.join(output_dir, 'mergeandresample', 'edgeinfo.txt')
    shutil.copyfile(edge_info, os.path.join(output_dir, subject_id + '_mesh_all', 'edgeinfo.txt'))

    logger.debug('subject {}, timeframe {}: finished mesh'.format(subject_id, timeframe_num))
    
def run_timeframe_uvc(subject_id, timeframe_num, input_dir, mesh_names, par_file, output_dir, output_dir_uvc):
    logger.debug('subject {}, timeframe {}: started uvc'.format(subject_id, timeframe_num))
    
    
    meshtoolloc = os.path.join(CARP_BIN, 'meshtool')
    
    CARP = os.path.join(CARP_BIN, 'openCARP')
    IGBEXTRACT = os.path.join(CARP_BIN, 'igbextract')
    MPIEXEC = os.path.join(CARP_BIN, 'mpiexec')
    ###############
    #for testing old vol mesh
    #meshdir=os.path.join(data_dir, subject_id, 'Volumetic_mesh_Outputs')
    ############
    meshdir=os.path.join(output_dir, subject_id + '_mesh_all')
    ###############
    meshname=subject_id+"_mesh_all"
    surfloc="surfaces"

    if not os.path.exists(meshdir):
        logger.error('subject {}, timeframe {}: no volumetric mesh found'.format(subject_id, timeframe_num))
        return
    
    
    
    finalmeshdir=output_dir_uvc
    
    os.chdir(finalmeshdir)
    if not os.path.exists(meshdir+'/'+meshname+".elem"):
        logger.error('subject {}, timeframe {}: no volumetric mesh found'.format(subject_id, timeframe_num))
        return
    
    shutil.copyfile(meshdir+'/'+meshname+".elem", meshname+".elem")
    shutil.copyfile(meshdir+'/'+meshname+".pts", meshname+".pts")
    shutil.copyfile(meshdir+'/'+meshname+".lon", meshname+".lon")
    
    
    UVCfolder=finalmeshdir+'/UVC_i'
    if not os.path.exists(UVCfolder):
        os.mkdir(UVCfolder)
    #UVC.coord z generation
    try:
        coordszname=UVC.coord_z(finalmeshdir,meshtoolloc,meshname,surfloc,UVCfolder)
    except Exception:
        logger.error('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe_num))
        return
    if coordszname == None:
        logger.error('subject {}, timeframe {}: failed to generate coord z'.format(subject_id, timeframe_num))
        return

    #UVC.rotational coords generation:LV
    if not UVC.find_two_RV_insertion_pts(finalmeshdir,meshname, surfloc,meshtoolloc,coordszname):
        logger.error('subject {}, timeframe {}: failed to find RV insertion points'.format(subject_id, timeframe_num))
        return
    predata=UVC.extract_surf_forLV(finalmeshdir,meshtoolloc,meshname,surfloc)
    if predata == None:
        logger.error('subject {}, timeframe {}: failed to extract surface for LV'.format(subject_id, timeframe_num))
        return
    phiLV='Phi_LV.dat'
    PhiLV=UVC.compute_uvc_rotational(finalmeshdir,'LVmyo_i',predata[0],predata[1],predata[2],predata[3],phiLV)

    #UVC.rotational coords generation:RV
    predataRV=UVC.extract_surf_forRV(finalmeshdir,meshtoolloc,meshname,surfloc,coordszname)
    if predataRV == None:
        logger.error('subject {}, timeframe {}: failed to extract surface for RV'.format(subject_id, timeframe_num))
        return
    phiRV='Phi_RV.dat'
    PhiRV=UVC.compute_uvc_rotationalRV(finalmeshdir,'RVmyo_i',predataRV[0],predataRV[1],predataRV[2],predataRV[3],phiRV)
    #merge UVC.rotational coords (PHI) files of LV and RV
    if UVC.merge_PHI(finalmeshdir,meshtoolloc,meshname,phiLV,phiRV,surfloc,UVCfolder) == None:
        logger.error('subject {}, timeframe {}: failed to merge phi'.format(subject_id, timeframe_num))
        return
    
    #UVC.transmural coords generation
    SIMID="transmural_results"
    meshtool.convert_mesh(meshtoolloc, f'{input_dir}/Mesh_aorta_valve_timeframe{timeframe_num:03}', f'Mesh_aorta_valve_timeframe{timeframe_num:03}', 'vtk')
    pts5810=meshIO.read_pts(basename=f'Mesh_aorta_valve_timeframe{timeframe_num:03}')
    PHOname=UVC.transmural(finalmeshdir,meshtoolloc,meshname,pts5810,surfloc,MPIEXEC,OPTS,CARP,par_file,SIMID,IGBEXTRACT,UVCfolder)
    if PHOname == None:
        logger.error('subject {}, timeframe {}: failed to generate transmural coordinates'.format(subject_id, timeframe_num))
        return
    #UVC.biventricular coords (LV:-1 and RV:1)
    UVC.bivlabel(finalmeshdir,meshname,UVCfolder)

    # #########################################
    #fiber generation
    alpha_epi = -np.pi/3
    alpha_endo = np.pi/3

    beta_epi = 2*np.pi*25/360
    beta_endo = -2*np.pi*65/360

    #for LV
    apexvtx=predata[1]
    base_surf=predata[0]
    LV_fiber=UVC.compute_fiber_new(finalmeshdir,meshtoolloc,"LVmyo_i",apexvtx,base_surf,PHOname,coordszname,alpha_epi,alpha_endo,beta_epi,beta_endo)
    if LV_fiber == None:
        logger.error('subject {}, timeframe {}: failed to generate LV fiber'.format(subject_id, timeframe_num))
        return
    #for RV
    apexvtx=predataRV[1]
    base_surf=predataRV[0]
    RV_fiber=UVC.compute_fiber_new(finalmeshdir,meshtoolloc,"RVmyo_i",apexvtx,base_surf,PHOname,coordszname,alpha_epi,alpha_endo,beta_epi,beta_endo)
    if RV_fiber == None:
        logger.error('subject {}, timeframe {}: failed to generate RV fiber'.format(subject_id, timeframe_num))
        return

    #merge fiber files and save to folder 'fiberfolder'
    fiberfolder=finalmeshdir+'/fiberfolder'
    if not os.path.exists(fiberfolder):
        os.mkdir(fiberfolder)
    if not UVC.merge_fibertomyoonly(finalmeshdir,meshtoolloc,meshname,fiberfolder):
        logger.error('subject {}, timeframe {}: failed to merge fiber'.format(subject_id, timeframe_num))
        return
    
    logger.debug('subject {}, timeframe {}: finished uvc'.format(subject_id, timeframe_num))

def generate_rv_mesh(subject_id, timeframe_num, vmtf_dir):
    logger.debug('subject {}, timeframe {}: started rv mesh'.format(subject_id, timeframe_num))

    meshtool_path = os.path.join(CARP_BIN, 'meshtool')

    if not os.path.exists(os.path.join(vmtf_dir, subject_id + '_mesh_all', subject_id + '_mesh_all.vtk')):
        logger.error('subject {}, timeframe {}: no volumetric mesh found'.format(subject_id, timeframe_num))
        return False

    imsh = os.path.join(vmtf_dir, subject_id + '_mesh_all', subject_id + '_mesh_all')
    omsh = os.path.join(vmtf_dir, subject_id + '_mesh_all', subject_id + '_mesh_all_rv')
    if not meshtool.extract_mesh(meshtool_path, imsh, '0', omsh, ofmt='vtk'):
        logger.error('subject {}, timeframe {}: failed to extract rv mesh'.format(subject_id, timeframe_num))
        return False

    logger.debug('subject {}, timeframe {}: finished rv mesh'.format(subject_id, timeframe_num))

    return True

def generate_uvc_meshes(subject_id, timeframe_num, uvc_dir):
    logger.debug('subject {}, timeframe {}: started uvc meshes'.format(subject_id, timeframe_num))

    meshtool_path = os.path.join(CARP_BIN, 'meshtool')

    if not os.path.exists(os.path.join(uvc_dir, 'fiberfolder', subject_id + '_mesh_all_fiber.vtk')):
        logger.error('subject {}, timeframe {}: no fiber mesh found'.format(subject_id, timeframe_num))
        return False

    fiber_i_dir = os.path.join(uvc_dir, 'fiber_i')
    if not os.path.exists(fiber_i_dir):
        os.mkdir(fiber_i_dir)

    imsh = os.path.join(uvc_dir, 'fiberfolder', subject_id + '_mesh_all_fiber')
    omsh = os.path.join(fiber_i_dir, subject_id + '_mesh_all_fiber_i')
    if not meshtool.extract_mesh(meshtool_path, imsh, '0,1', omsh):
        logger.error('subject {}, timeframe {}: failed to extract myocardium'.format(subject_id, timeframe_num))
        return False

    imsh = os.path.join(fiber_i_dir, subject_id + '_mesh_all_fiber_i')
    ifmt = 'carp_txt'
    ofmt = 'vtu_bin'

    omsh = os.path.join(uvc_dir, 'UVC_i', subject_id + '_uvc_coords_z')
    nod = os.path.join(uvc_dir, 'UVC_i', 'COORDS_Z.dat')
    if not meshtool.collect_nodal(meshtool_path, imsh, ifmt, omsh, ofmt, nod):
        logger.error('subject {}, timeframe {}: failed to generate coords-z mesh'.format(subject_id, timeframe_num))
        return False

    omsh = os.path.join(uvc_dir, 'UVC_i', subject_id + '_uvc_phi')
    nod = os.path.join(uvc_dir, 'UVC_i', 'PHI.dat')
    if not meshtool.collect_nodal(meshtool_path, imsh, ifmt, omsh, ofmt, nod):
        logger.error('subject {}, timeframe {}: failed to generate phi mesh'.format(subject_id, timeframe_num))
        return False

    omsh = os.path.join(uvc_dir, 'UVC_i', subject_id + '_uvc_rho')
    nod = os.path.join(uvc_dir, 'UVC_i', 'PHO.dat')
    if not meshtool.collect_nodal(meshtool_path, imsh, ifmt, omsh, ofmt, nod):
        logger.error('subject {}, timeframe {}: failed to generate rho mesh'.format(subject_id, timeframe_num))
        return False

    if os.path.isdir(fiber_i_dir):
        shutil.rmtree(fiber_i_dir)

    logger.debug('subject {}, timeframe {}: finished uvc meshes'.format(subject_id, timeframe_num))

    return True

def generate_fiber_glyphs_mesh(subject_id, timeframe_num, uftf_dir):
    logger.debug('subject {}, timeframe {}: started fiber glyphs mesh'.format(subject_id, timeframe_num))

    fiber_dir = os.path.join(uftf_dir, 'fiberfolder')
    fiber_name = subject_id + '_mesh_all_fiber'

    if not os.path.exists(os.path.join(fiber_dir, fiber_name + '.vtk')):
        logger.error('subject {}, timeframe {}: no fiber mesh found'.format(subject_id, timeframe_num))
        return False

    glyphs = paf.carp_to_pyvista(os.path.join(fiber_dir, fiber_name), stride=1, skip=20)
    glyphs.save(os.path.join(fiber_dir, fiber_name + '_glyphs.vtp'), binary=True)

    logger.debug('subject {}, timeframe {}: finished fiber glyphs mesh'.format(subject_id, timeframe_num))

def take_screenshots(out_dir, out_name, mesh_filename, title, camera_positions, size=(640, 480), azimuth=0, elevation=0, clip=None, scalars=None):
    dataset = pv.read(mesh_filename)

    if clip != None:
        dataset = dataset.clip(clip)

    points = dataset[scalars] if scalars else None

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(dataset, scalars=points, cmap='coolwarm')
    plotter.remove_scalar_bar()
    plotter.show_axes()
    plotter.set_background('white')

    for position in camera_positions:
        plotter.camera_position = position
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = elevation
        plotter.add_title(title, color='black', font_size=12)
        filename = os.path.join(out_dir, out_name + '_' + position + '.png')
        plotter.screenshot(filename=filename, window_size=size)
        logger.debug('taken screenshot {}'.format(filename))

    plotter.close()

def generate_quality_control_images(subject_id, timeframe_num, output_dir, uvc_dir):
    logger.debug('subject {}, timeframe {}: started quality control'.format(subject_id, timeframe_num))

    mesh_dir = os.path.join(output_dir, subject_id + '_mesh_all')

    mesh_file = os.path.join(mesh_dir, subject_id + '_mesh_all.vtk')
    if not os.path.exists(mesh_file):
        logger.error('subject {}, {}: failed to take screenshots, no volumetric mesh found'.format(subject_id, timeframe_num))
        return False

    take_screenshots(mesh_dir, subject_id + '_mesh_all', mesh_file, '{}:{}'.format(subject_id, timeframe_num), ['xy'])

    #rv_file = os.path.join(mesh_dir, subject_id + '_mesh_all_rv.vtk')
    #if not os.path.exists(rv_file):
    #    logger.error('subject {}, {}: failed to take screenshots, no rv myo mesh found'.format(subject_id, timeframe_num))
    #    return False
    #take_screenshots(mesh_dir, subject_id + '_mesh_all_rv', rv_file, '{}:{}'.format(subject_id, timeframe_num), ['xy', 'xz', 'yz'], azimuth=45)

    uvc_dir = os.path.join(uvc_dir, 'UVC_i')

    mesh_file = os.path.join(uvc_dir, subject_id + '_uvc_coords_z.vtu')
    if not os.path.exists(mesh_file):
        logger.error('subject {}, {}: failed to take screenshots, no uvc coords z mesh found'.format(subject_id, timeframe_num))
        return False
    take_screenshots(uvc_dir, subject_id + '_uvc_coords_z', mesh_file, '{}:{}, z'.format(subject_id, timeframe_num), ['xy'], scalars='COORDS_Z_0')

    mesh_file = os.path.join(uvc_dir, subject_id + '_uvc_phi.vtu')
    if not os.path.exists(mesh_file):
        logger.error('subject {}, {}: failed to take screenshots, no uvc phi mesh found'.format(subject_id, timeframe_num))
        return False
    take_screenshots(uvc_dir, subject_id + '_uvc_phi', mesh_file, '{}:{}, phi'.format(subject_id, timeframe_num), ['xy'], scalars='PHI_0')

    mesh_file = os.path.join(uvc_dir, subject_id + '_uvc_rho.vtu')
    if not os.path.exists(mesh_file):
        logger.error('subject {}, {}: failed to take screenshots, no uvc rho mesh found'.format(subject_id, timeframe_num))
        return False
    take_screenshots(uvc_dir, subject_id + '_uvc_rho', mesh_file, '{}:{}, rho'.format(subject_id, timeframe_num), ['xy'], clip='z', scalars='PHO_0')

    fiber_dir = os.path.join(uvc_dir, 'fiberfolder')

    mesh_file = os.path.join(fiber_dir, subject_id + '_mesh_all_fiber_glyphs.vtp')
    if not os.path.exists(mesh_file):
        logger.error('subject {}, {}: failed to take screenshots, no fiber glyphs mesh found'.format(subject_id, timeframe_num))
        return False

    take_screenshots(fiber_dir, subject_id + '_mesh_all_fiber_glyphs', mesh_file, '{}:{}, fiber'.format(subject_id, timeframe_num), ['xy'], size=(1280, 960))

    logger.debug('subject {}, timeframe {}: finished quality control'.format(subject_id, timeframe_num))

    return True

def main(do_mesh=False, do_uvc=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--profile', '-p', action='store', default='default', help='config profile to be used')
    parser.add_argument('--job', '-j', action='store', default='default', help='job identifier')
    parser.add_argument('--force', '-f', action='store_true', help='force overwrite')

    parser.add_argument('--data-dir', '-d', action='store', help='path to data directory')
    parser.add_argument('--input-dir', '-I', action='store', default='Mesh_Outputs', help='name of input directories')
    parser.add_argument('--carp-bin-dir', '-c', action='store', help='path to carp bin directory')

    parser.add_argument('--instance', '-i', type=int, action='store', default=2, help='instance to be processed')

    parser.add_argument('--all', '-a', action='store_true', help='process all subjects')
    parser.add_argument('--subject', '-s', action='store', help='subject id to be processed')
    parser.add_argument('--start', '-S', action='store', type=int, help='index of first subject id to be processed')
    parser.add_argument('--number', '-n', action='store', type=int, help='number of subjects to be processed from first subject id')
    parser.add_argument('--allowlist', '-l', action='store', help='path to subject allowlist')

    parser.add_argument('--timeframe', '-t', type=int, action='store', help='timeframe to be processed')

    parser.add_argument('--mesh', '-m', action='store_true', help='generate mesh')
    parser.add_argument('--uvc', '-u', action='store_true', help='generate uvc and fiber')
    parser.add_argument('--uvc-meshes', '-z', action='store_true', help='generate uvc meshes')
    parser.add_argument('--quality-control', '-q', action='store_true', help='output quality control screenshots')

    args, _ = parser.parse_known_args()

    cfg = configparser.ConfigParser()
    cfg.read('config.ini')

    global CARP_BIN
    CARP_BIN = args.carp_bin_dir if args.carp_bin_dir else cfg[args.profile]['CarpBinDir']

    global OPTS
    OPTS = os.path.abspath('./amg_cg_opts')

    data_dir = args.data_dir if args.data_dir else cfg[args.profile]['DataDir']
    par_file = os.path.abspath('./LVendo_RVseptum_RVendo.par')
    uvc_par_file = os.path.abspath('./transmural.par')

    if args.allowlist and os.path.exists(args.allowlist):
        with open(args.allowlist) as f:
            sids = [n for n in f.read().splitlines() if os.path.isdir(os.path.join(data_dir, n))]
    else:
        sids = [n for n in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, n))]

    if args.all:
        subject_ids = sorted(sids)
    elif args.subject:
        sid = args.subject
        if sid in sids:
            subject_ids = [sid]
        else:
            subject_ids = []
    elif args.start is not None and args.start >= 0 and args.start < len(sids):
        if args.number is not None and args.number > 0:
            end = args.start + args.number - 1
            subject_ids = sorted(sids)[args.start:end+1]
        else:
            subject_ids = sorted(sids)[args.start:]
    else:
        subject_ids = []

    log_filename = os.path.join(data_dir, f'volumetric-{args.job}.log')
    formatter = logging.Formatter(fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    np.seterr(all='warn')

    for subject_id in subject_ids:
        if not os.path.isdir(os.path.join(data_dir, subject_id)):
            continue

        input_dir = os.path.join(data_dir, subject_id, f'Instance_{args.instance}', f'{args.input_dir}')
        output_dir = os.path.join(data_dir, subject_id, f'Instance_{args.instance}', 'volumetric_mesh_outputs')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir_uvc = os.path.join(data_dir, subject_id, f'Instance_{args.instance}', 'UVC_fiber_Outputs')
        if not os.path.exists(output_dir_uvc):
            os.mkdir(output_dir_uvc)

        for tf_n, mesh_names in timeframes(subject_id, input_dir):
            if (args.timeframe is not None and args.timeframe != tf_n):
                continue

            if do_mesh or args.mesh:
                vmtf_dir = os.path.join(output_dir, 'timeframe{}'.format(tf_n))
                if not os.path.exists(vmtf_dir):
                    os.mkdir(vmtf_dir)

                if os.path.exists(os.path.join(vmtf_dir, subject_id + '_mesh_all', subject_id + '_mesh_all.vtk')) and not args.force:
                    logger.debug('subject {}, timeframe {}: skipping, volumetric mesh exists'.format(subject_id, tf_n))
                    continue

                run_timeframe(subject_id, tf_n, input_dir, mesh_names, par_file, vmtf_dir)

                for filename in os.listdir(vmtf_dir):
                    filepath = os.path.join(vmtf_dir, filename)
                    if os.path.isdir(filepath):
                        if filename != f'{subject_id}_mesh_all':
                            shutil.rmtree(filepath)
                    else:
                        os.remove(filepath)

            if do_uvc or args.uvc:
                uftf_dir = os.path.join(output_dir_uvc, 'timeframe{}'.format(tf_n))
                if not os.path.exists(uftf_dir):
                    os.mkdir(uftf_dir)

                if os.path.exists(os.path.join(uftf_dir, 'fiberfolder', subject_id + '_mesh_all_fiber.vtk')) and not args.force:
                    logger.debug('subject {}, timeframe {}: skipping, fiber mesh exists'.format(subject_id, tf_n))
                    continue

                vmtf_dir = os.path.join(output_dir, 'timeframe{}'.format(tf_n))
                run_timeframe_uvc(subject_id, tf_n, input_dir, mesh_names, uvc_par_file, vmtf_dir, uftf_dir)

                for filename in os.listdir(uftf_dir):
                    filepath = os.path.join(uftf_dir, filename)
                    if os.path.isdir(filepath):
                        #the folder 'surfaces' is not a must, and can be removed for integrating in CemrgAPP
                        if filename not in ['fiberfolder', 'surfaces', 'UVC_i']:
                            shutil.rmtree(filepath)
                    else:
                        os.remove(filepath)

            if args.uvc_meshes:
                uftf_dir = os.path.join(output_dir_uvc, 'timeframe{}'.format(tf_n))
                generate_uvc_meshes(subject_id, tf_n, uftf_dir)
                generate_fiber_glyphs_mesh(subject_id, tf_n, uftf_dir)

            if args.quality_control:
                vmtf_dir = os.path.join(output_dir, 'timeframe{}'.format(tf_n))
                #generate_rv_mesh(subject_id, tf_n, vmtf_dir)
                uftf_dir = os.path.join(output_dir_uvc, 'timeframe{}'.format(tf_n))
                generate_quality_control_images(subject_id, tf_n, vmtf_dir, uftf_dir)

if __name__ == '__main__':
    main()
