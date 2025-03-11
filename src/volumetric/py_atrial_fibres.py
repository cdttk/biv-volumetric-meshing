import random

import numpy as np
import pyvista as pv
import vtk

def read_pts(filename):
    return np.loadtxt(filename, dtype=float, skiprows=1)

def read_elem(filename):
    return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3,4))

def read_lon(filename):
    return np.loadtxt(filename, dtype=float, skiprows=1)

def carp_to_pyvista(meshname, stride=3, tube_radius=0.08, skip=0):
    pts = read_pts(meshname + '.pts')
    elem = read_elem(meshname + '.elem')
    print(elem.shape[0])
    elem = elem if skip == 0 else elem[::skip, :]
    print(elem.shape[0])

    tets = np.column_stack((np.ones((elem.shape[0],), dtype=int) * 4, elem)).flatten()
    cell_type = np.ones((elem.shape[0],), dtype=int) * vtk.VTK_TETRA

    plt_msh = pv.UnstructuredGrid(tets, cell_type, pts)

    lon = read_lon(meshname + ".lon")
    lon = lon if skip == 0 else lon[::skip, :]
    fibres = lon[:, :3]

    nelem = lon.shape[0]
    nelem_nofibres = int(nelem * (1 - 1. / stride))
    exclude = random.sample(range(0, nelem), nelem_nofibres)

    fibres[exclude, :] = np.zeros((nelem_nofibres, 3), dtype=float)

    plt_msh["fibres"] = fibres

    line = pv.Line()
    glyphs = plt_msh.glyph(orient='fibres', scale=True, factor=2000.0, geom=line.tube(radius=tube_radius))

    return glyphs
