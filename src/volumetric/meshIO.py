# Some functions included in this file were written by Martin Bishop
import pandas as pd
import glob
import logging
import math
import random
import numpy as np

logger = logging.getLogger(__name__)

#########################################
# Function to read in mesh from basename
#########################################
def read_mesh(basename=None, file_pts=None, file_elem=None, file_lon=None):
    # Function to read in mesh from basename

    # Defines pts, elem and lon files from basename
    if file_pts is None:
        file_pts = glob.glob(basename + '*.pts')
        if len(file_pts) > 1:
            raise ValueError('Too many matching .pts files')
        elif len(file_pts) == 0:
            raise ValueError('No matching .pts files')
        file_pts = file_pts[0]
    if file_elem is None:
        file_elem = glob.glob(basename + '*.elem')
        if len(file_elem) > 1:
            raise ValueError('Too many matching .elem files')
        elif len(file_elem) == 0:
            raise ValueError('No matching .elem files')
        file_elem = file_elem[0]
    if file_lon is None:
        file_lon = glob.glob(basename + '*.lon')
        if len(file_lon) > 1:
            raise ValueError('Too many matching .lon files')
        elif len(file_lon) == 0:
            raise ValueError('No matching .lon files')
        file_lon = file_lon[0]

    # Read mesh files
    try:
        pts = pd.read_csv(file_pts, sep=' ', skiprows=1, header=None)
        logger.debug("Successfully read {}".format(file_pts))
    except ValueError:
        pts = None
    elem = pd.read_csv(file_elem, sep=' ', skiprows=1, usecols=(1, 2, 3, 4, 5), header=None)
    logger.debug("Successfully read {}".format(file_elem))
    lon = pd.read_csv(file_lon, sep=' ', skiprows=1, header=None)
    logger.debug("Successfully read {}".format(file_lon))

    return pts, elem, lon


#########################################
# Function to write mesh
#########################################
def write_mesh(basename, pts=None, elem=None, lon=None, shapes=None, precision_pts=None, precision_lon=None):
    # Write pts, elem and lon data to file

    # Ensure *something* is being written!
    assert ((pts is not None) and (elem is not None) and (lon is not None)), "No data given to write to file."

    # Adapt precision to default formats
    if precision_pts is None:
        precision_pts = '%.12g'
    if precision_lon is None:
        precision_lon = '%.5g'

    # Basic error checking on output file name
    if basename[-1] == '.':
        basename = basename[:-1]

    #######################
    # Writes-out pts file
    #######################
    if pts is not None:
        with open(basename + '.pts', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
        pts.to_csv(basename + '.pts', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        logger.debug("pts data written to file {}".format(basename + '.pts'))

    ######################
    # Writes-out elems file
    ######################
    # If we haven't defined a shape for our elements, set to be tets
    if shapes is None:
        shapes = 'Tt'

    if elem is not None:
        with open(basename + '.elem', 'w') as pFile:
            pFile.write('{}\n'.format(len(elem)))
        elem.insert(loc=0, value=shapes, column=0)
        elem.to_csv(basename + '.elem', sep=' ', header=False, index=False, mode='a')
        logger.debug("elem data written to file {}".format(basename + '.elem'))
        del elem[0]  # Remove added column to prevent cross-talk problems later

    ######################
    # Writes-out lon file
    ######################
    if lon is not None:
        with open(basename + '.lon', 'w') as pFile:
            pFile.write('1\n')
        lon.to_csv(basename + '.lon', sep=' ', header=False, index=False, mode='a', float_format=precision_lon)
        logger.debug("lon data written to file {}".format(basename + '.lon'))

    return None


#########################################
# Function to read UVC data and interpolate onto elements
#########################################

#########################################
# Function to read pts file
#########################################
def read_pts(basename=None, file_pts=None):
    # Function to read in mesh from basename

    if file_pts is None:
        file_pts = glob.glob(basename + '.pts')
        #if len(file_pts) > 1:
         #   raise ValueError('Too many matching .pts files')
        if len(file_pts) == 0:
            raise ValueError('No matching .pts files')
        file_pts = file_pts[0]

    # Read mesh files
    pts = pd.read_csv(file_pts, sep=' ', skiprows=1, header=None)
    logger.debug("Successfully read {}".format(file_pts))
    logger.debug('Mesh has {} nodes'.format(len(pts)))


    return pts

#########################################
# Function to read cpts file
#########################################
def read_cpts(basename=None, file_cpts=None):
    # Function to read in mesh from basename

    if file_cpts is None:
        file_cpts = glob.glob(basename + '*.cpts')
        #if len(file_pts) > 1:
         #   raise ValueError('Too many matching .pts files')
        if len(file_cpts) == 0:
            raise ValueError('No matching .cts files')
        file_cpts = file_cpts[0]

    # Read mesh files
    cpts = pd.read_csv(file_pts, sep=' ', skiprows=1, header=None)
    logger.debug("Successfully read {}".format(file_pts))

    return cpts



#########################################
# Function to read elems file
#########################################
def read_elems(basename=None, file_elem=None):
    # Function to read in mesh from basename

    if file_elem is None:
        file_elem = glob.glob(basename + '.elem')
        if len(file_elem) > 1:
            raise ValueError('Too many matching .elem files')
        elif len(file_elem) == 0:
            raise ValueError('No matching .elem files')
        file_elem = file_elem[0]

    # Read mesh files
    elem = pd.read_csv(file_elem, sep=' ', skiprows=1, usecols=(1, 2, 3, 4, 5), header=None)
    logger.debug("Successfully read {}".format(file_elem))
    logger.debug('Mesh has {} elements'.format(len(elem)))
    return elem


#########################################
# Function to read lon file
#########################################
def read_fibres(basename=None, file_lon=None):
   
    # Defines  lon files from basename
    if file_lon is None:
        file_lon = glob.glob(basename + '.lon')
        if len(file_lon) > 1:
            raise ValueError('Too many matching .lon files')
        elif len(file_lon) == 0:
            raise ValueError('No matching .lon files')
        file_lon = file_lon[0]

    # Read mesh files
    lon = pd.read_csv(file_lon, sep=' ', skiprows=1, header=None)
    logger.debug("Successfully read {}".format(file_lon))

    return lon


#########################################
# Function to write element file
#########################################
def write_elems(elemFilename=None, elem=None, shapes=None):
    # Write elem

    # Ensure *something* is being written!
    assert ((elem is not None)), "No data given to write to file."

    ######################
    # Writes-out elems file
    ######################
    # If we haven't defined a shape for our elements, set to be tets
    if shapes is None:
        shapes = 'Tt'

    if elem is not None:
        with open(elemFilename + '.elem', 'w') as pFile:
            pFile.write('{}\n'.format(len(elem)))
        elem.insert(loc=0, value=shapes, column=0)
        elem.to_csv(elemFilename + '.elem', sep=' ', header=False, index=False, mode='a')
        logger.debug("elem data written to file {}".format(basename + '.elem'))
        del elem[0]  # Remove added column to prevent cross-talk problems later

    return None


#########################################
# Function to write lon file
#########################################
def write_lon(lonFilename=None, lon=None):
    # Ensure *something* is being written!
    assert ((lon is not None)), "No data given to write to file."

    ######################
    # Writes-out lon file
    ######################
    if lon is not None:
        with open(lonFilename + '.lon', 'w') as pFile:
            pFile.write('1\n')
        lon.to_csv(lonFilename + '.lon', sep=' ', header=False, index=False, mode='a')
        logger.debug("lon data written to file {}".format(lonFilename + '.lon'))

    return None


#########################################
def write_lon_includesheet(lonFilename=None, lon=None):
    # Ensure *something* is being written!
    assert ((lon is not None)), "No data given to write to file."

    ######################
    # Writes-out lon file
    ######################
    if lon is not None:
        with open(lonFilename + '.lon', 'w') as pFile:
            pFile.write('2\n')
        lon.to_csv(lonFilename + '.lon', sep=' ', header=False, index=False, mode='a')
        logger.debug("lon data written to file {}".format(lonFilename + '.lon'))

    return None


#########################################
# Function to write pts file
#########################################
def write_pts(ptsFilename=None, pts=None):
    # Ensure *something* is being written!
    assert ((pts is not None)), "No data given to write to file."

    precision_pts = '%.12g'

    ######################
    # Writes-out pts file
    ######################
    if pts is not None:
        with open(ptsFilename + '.pts', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
        pts.to_csv(ptsFilename + '.pts', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        logger.debug("pts data written to file {}".format(ptsFilename + '.pts'))

    return None

#########################################
# Function to write auxgrid pts file
#########################################
def write_auxpts(auxptsFilename=None, pts=None):
    # Ensure *something* is being written!
    assert ((pts is not None)), "No data given to write to file."

    precision_pts = '%.12g'

    ######################
    # Writes-out pts file
    ######################
    if pts is not None:
        with open(auxptsFilename + '.pts_t', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
            pFile.write("1\n")
        pts.to_csv(auxptsFilename + '.pts_t', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        logger.debug("pts data written to file {}".format(auxptsFilename + '.pts_t'))

    return None

#########################################
# Function to write out points data
#########################################
def write_data(dataFilename=None, data=None):
    # Ensure *something* is being written!
    assert ((data is not None)), "No data given to write to file."
    
    if data is not None:
        dFile = open(dataFilename, '+w') 
        for i in data:
            dFile.write("%f\n" %i)
        dFile.close()  
        
    return None
        
#########################################
# Function to write out node/element list
#########################################
def write_list(dataFilename=None, data=None):
    # Ensure *something* is being written!
    assert ((data is not None)), "No data given to write to file."
    
    if data is not None:
        dFile = open(dataFilename, '+w') 
        for i in data:
            dFile.write("%i\n" %i)
        dFile.close()  
        
    return None
        

#########################################
# Function to create a centroids file
#########################################
def create_centroids(elems=None, pts=None):
    
    coords_n0 = np.array(pts.iloc[elems.iloc[:,0]])
    coords_n1 = np.array(pts.iloc[elems.iloc[:,1]])
    coords_n2 = np.array(pts.iloc[elems.iloc[:,2]])
    coords_n3 = np.array(pts.iloc[elems.iloc[:,3]])
    mean_coords = (coords_n0 + coords_n1 + coords_n2 + coords_n3)*0.25
    
    centroids = pd.DataFrame(mean_coords)
        
    return centroids
           
#########################################
# Function to read surf file for surface
#########################################
def read_surf(basename=None, file_surf=None):
    # Function to read in mesh from basename

    if file_surf is None:
        file_surf = glob.glob(basename + '.surf')
        #if len(file_pts) > 1:
         #   raise ValueError('Too many matching .pts files')
        if len(file_surf) == 0:
            raise ValueError('No matching .surf files')
        file_surf = file_surf[0]

    # Read mesh files
    surf = pd.read_csv(file_surf, sep=' ', skiprows=1, header=None)
    logger.debug("Successfully read {}".format(file_surf))
    logger.debug('Surface has {} tets'.format(len(surf)))


    return surf

#########################################
# Function to read elem file for surface
#########################################
def read_elem(basename=None, file_elem=None):
    # Function to read in mesh from basename

    if file_elem is None:
        file_elem = glob.glob(basename + '.elem')
        #if len(file_pts) > 1:
         #   raise ValueError('Too many matching .pts files')
        if len(file_elem) == 0:
            raise ValueError('No matching .elem files')
        file_elem = file_elem[0]

    # Read mesh files
    elem = pd.read_csv(file_elem, sep=' ', skiprows=1, header=None)
    logger.debug("Successfully read {}".format(file_elem))
    logger.debug('Mesh has {} elements'.format(len(elem)))


    return elem

#########################################
# Function to write element file for surface
#########################################
def write_surf(surfFilename=None, surf=None, shapes=None):
    # Write elem

    # Ensure *something* is being written!
    assert ((surf is not None)), "No data given to write to file."

    ######################
    # Writes-out elems file
    ######################
    # If we haven't defined a shape for our elements, set to be tets
    if shapes is None:
        shapes = 'Tr'

    if surf is not None:
        with open(surfFilename + '.surf', 'w') as pFile:
            pFile.write('{}\n'.format(len(surf)))
        if surf.iloc[0,0]!='Tr':
            surf.insert(loc=0, value=shapes, column=0)
            surf.to_csv(surfFilename + '.surf', sep=' ', header=False, index=False, mode='a')
            logger.debug("surf data written to file {}".format(surfFilename + '.surf'))
            del surf[0]  # Remove added column to prevent cross-talk problems later
        else:
            surf.to_csv(surfFilename + '.surf', sep=' ', header=False, index=False, mode='a')
            logger.debug("surf data written to file {}".format(surfFilename + '.surf'))
            del surf[0]  # Remove added column to prevent cross-talk problems later

    return None

#########################################
# Function to write element file for surfaces
#########################################
def write_elem(elemFilename=None, elem=None, shapes=None):
    # Write elem

    # Ensure *something* is being written!
    assert ((elem is not None)), "No data given to write to file."

    ######################
    # Writes-out elems file
    ######################
    # If we haven't defined a shape for our elements, set to be tets
    if shapes is None:
        shapes = 'Tr'

    if elem is not None:
        with open(elemFilename + '.elem', 'w') as pFile:
            pFile.write('{}\n'.format(len(elem)))
        if elem.iloc[0,0]!='Tr':
            elem.insert(loc=0, value=shapes, column=0)
            elem.to_csv(elemFilename + '.elem', sep=' ', header=False, index=False, mode='a')
            logger.debug("elem data written to file {}".format(elemFilename + '.elem'))
            del elem[0]  # Remove added column to prevent cross-talk problems later
        else:
            elem.to_csv(elemFilename + '.elem', sep=' ', header=False, index=False, mode='a')
            logger.debug("elem data written to file {}".format(elemFilename + '.elem'))
            del elem[0]  # Remove added column to prevent cross-talk problems later

    return None

#########################################
# Function to read in vtx file
#########################################
def read_vtx_File(vtxFilename=None):
    
    vtxs = np.loadtxt(vtxFilename,skiprows=2)
        
    return vtxs


#########################################
# Function to write out vtx file
#########################################
def write_vtx_File(vtxFilename=None, vtx=None):
    # Ensure *something* is being written!
    assert ((vtx is not None)), "No data given to write to file."
    
    if vtx is not None:
        #vtx.to_csv(vtxFilename + '.vtx', header=False, index=False, mode='a')
        dFile = open(vtxFilename+ '.vtx', '+w') 
        #dFile.write("1\n")
        dFile.write("%i\n" %len(vtx))
        dFile.write("extra\n")
        for i in range(len(vtx)):
            dFile.write("%i\n" %vtx.loc[i])
        #vtx.to_csv(vtxFilename + '.vtx', header=False, index=False, mode='a')
        dFile.close()   
        logger.debug("vtx data written to file {}".format(vtxFilename + '.vtx'))
    return None

#########################################
# Function to immediately convert surface to list of unique nodes
#########################################
def read_surf_to_nodeList(surf=None,Filename=False):
    # Ensure *something* is being written!
    if surf is not None:
        
        surf_nodes = []
        surf_nodes = np.append(surf[1],surf_nodes)
        surf_nodes = np.append(surf[2],surf_nodes)
        surf_nodes = np.append(surf[3],surf_nodes)
        surf_nodes = np.unique(surf_nodes)
        
        surf_nodes = surf_nodes.astype(int)
        
    return surf_nodes



