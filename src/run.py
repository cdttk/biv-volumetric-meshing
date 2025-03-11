#!/bin/env python3

import argparse
import logging
import os
import warnings

from segmentation import predict_UKBB
from surface import contour2gp_QS
from surface import perform_fit
from surface import mesh_txt_to_vtk
from volumetric import main_testvmesh

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--all-components', action='store_true', help='run all components')
    parser.add_argument('--segmentation', action='store_true', help='run segmentation')
    parser.add_argument('--contour', action='store_true', help='run contour')
    parser.add_argument('--surface', action='store_true', help='run surface')
    parser.add_argument('--volumetric', action='store_true', help='run volumetric')
    parser.add_argument('--uvc-fiber', action='store_true', help='run UVC and fiber')

    args, _ = parser.parse_known_args()

    base_dir = os.path.dirname(__file__)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    if args.all_components:
        args.segmentation, args.contour, args.surface, args.volumetric, args.uvc_fiber = True, True, True, True, True

    if args.segmentation:
        os.chdir(os.path.join(base_dir, 'segmentation'))
        predict_UKBB.main()
    if args.contour:
        os.chdir(os.path.join(base_dir, 'surface'))
        contour2gp_QS.main()
    if args.surface:
        os.chdir(os.path.join(base_dir, 'surface'))
        perform_fit.main()
        mesh_txt_to_vtk.main()
    if args.volumetric:
        os.chdir(os.path.join(base_dir, 'volumetric'))
        main_testvmesh.main(True, False)
    if args.uvc_fiber:
        os.chdir(os.path.join(base_dir, 'volumetric'))
        main_testvmesh.main(False, True)
