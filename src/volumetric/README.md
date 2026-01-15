# volumetric_mesh ---Shuang Qian

#######################################

The main script is "main_testvmesh.py".
The dependencies include:
1. carpfunc.py
2. meshtool_func.py
3. v_mesh_generation.py
4. meshIO.py
5. LVendo_RVseptum_Rvendo.par (this is a file required for running CARP/OpenCARP)

########################################

Step 1: load surface meshes
 (1) Input the location where 8 surface meshes in .vtk are.
 (2) Sort them in names will get a list as:
    #   surfacemesh list is:
    #    0: LV_endo
    #    1: RV_FW
    #    2: RV_septum
    #    3: aorta_valve
    #    4: epi
    #    5: mitral_valve
    #    6: pulmonary_valve
    #    7: tricuspid_valve
*may need to check if the input names change.

########################################

Step 2: generate valves using the function called generate_valve()
Example: “aorta=generate_valve(meshtoolloc,folder,Valvename=surfacemesh[3],endoname=surfacemesh[0],outmsh="aortamesh",bdry_step="0.5",ptindex=5)”
“folder” is the folder where surface mesh is
“Valvename” is the filename of the valves (according to the order of sorted filename)
“endoname” is which endo surface that the target valve is located (also according to the order of sorted filename)
“outmsh” is the foldername and the filename of the volumetric valve mesh
“bdry_step” is how far is the middle point in the lower plane from the original valve surface (see next slide)
“ptindex” is the index of the valve points in original .pts(totally 5810) and also the element tag for the volumetric valve mesh 

########################################

Step 3: merge all volumetric meshes and resample using function merge_resample(): 
splitmeshdir=vmesh.split_RVLV(meshtoolloc,OPTS,CARP,parfile,mesh_nolabel+"_i",SIMID,surf_endo,IGBEXTRACT,thres,split_myo_only)
Note the edge length of resampled mesh can be checked in “edgeinfo.txt”
*Currently the resample setting is the optimal value to achieve mean edge length of  ~1mm
*It can be potentially modified to find the optimal resample setting by using the info in this “edgeinfo.txt”

########################################

Step 4: extract surfaces using the function extract_surfacenolabel()  
This step is to extract three surfaces: LV_endo, RV_endo, RV_septum for the next step
surf_endo=vmesh.extract_surfacenolabel(meshtoolloc,mesh_nolabel=folder+"/"+outmsh+"/"+outmsh, pts_5810=pts,surfmeshloc)
(1) extract allvalves_surf and myo_surf. 
(2) then “myo_surf – allvalves_surf” gives separated epi, RV_endo and LV_endo
(3) use points on LV_endo to select the LV_endo
(4) use points on RV_endo and “-edge=10” to select the RV_endo
(5) use points on RV_sep and “-edge=10” to select the RV_sep

########################################

Step 5: Split/retag the LV and RV using function split_RVLV():
splitmeshdir=vmesh.split_RVLV(meshtoolloc,OPTS,CARP,parfile,mesh_nolabel+"_i",SIMID,surf_endo,IGBEXTRACT,thres,split_myo_only)
This step require Laplace solver running by OpenCARP
The “thres” is the value to determine where LV and RV boundary is. This 0.7 is an optimal value to split it neatly. 

########################################

If it runs correctly, it will produce the final mesh to "/mesh_all/mesh_all" in both CARP format (.pts, .elem, .lom) and .vtk format

############################################
Next step is to generate UVCs and fibers.
The main script is "main_UVC_fiber.py".
The dependencies include:
1. carpfunc.py
2. meshtool_func.py
3. compute_UVC_fiber.py
4. meshIO.py
5. transmural.par (this is a file required for running CARP/OpenCARP)
############################################


At last, three folders are required:
(1) The final mesh in "/mesh_all/fiberfolder".
(2) The UVCs in '/mesh_all/UVC_i'.
(3) The surfaces in'/mesh_all/surface/biv_i'. 








