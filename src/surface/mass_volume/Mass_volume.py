'''
15/09/2022 - Laura Dal Toso
Based on A.M's scripts.
Script for the measurement of LV and LV mass and volume from biventricular models.
'''

import argparse
import csv
import numpy as np
import os
import re

from pathlib import Path

from mesh import Mesh


def conv(filepath):
    outname = str(filepath.name).replace('model', 'Model').replace('txt', 'csv')
    outpath = filepath.parent / outname

    with open(filepath) as f:
        inlines = f.read().splitlines()

    outlines = []
    for line in inlines[1:]:
        outlines.append(','.join(line.split()[:3]) + '\n')

    with open(outpath, 'w+') as f:
        f.write('x,y,z\n')
        f.writelines(outlines)

    return outpath

def find_volume(case_name: str, model_file: os.PathLike, output_file: os.PathLike, biv_model_folder: os.PathLike, precision : int) -> None:
    '''
        # Authors: ldt, cm
        # Date: 09/22, revised 08/24 by cm

        This function measures the mass and volume of LV and RV.
        #--------------------------------------------------------------
        Inputs: case_name = model case name
                model_file = fitted model (.txt), containing only data relative to one frame
                output_file = path to the output csv file

        Output: dictionary and csv file containing masses and volumes
    '''

    # get the frame number
    frame_name = int(re.search(r'timeframe(\d+)\.csv', str(model_file))[1])

    # read GP file
    control_points = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)

    # assign values to dict
    results_dict = {k: '' for k in ['lv_endo', 'rv_endo', 'lv_epi', 'rv_epi', 'lv_mass', 'rv_mass']}

    subdivision_matrix_file = biv_model_folder / "subdivision_matrix.txt"
    assert subdivision_matrix_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {subdivision_matrix_file} file!"

    elements_file = biv_model_folder / 'ETIndicesSorted.txt'
    assert elements_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {elements_file} file!"

    material_file = biv_model_folder / 'ETIndicesMaterials.txt'
    assert material_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {material_file} file!"

    thru_wall_file = biv_model_folder / 'epi_to_septum_ETindices.txt'
    assert thru_wall_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {thru_wall_file} file!"

    if control_points.shape[0] > 0:
        subdivision_matrix = (np.loadtxt(subdivision_matrix_file)).astype(float)
        faces = np.loadtxt(elements_file).astype(int)-1
        mat = np.loadtxt(material_file, dtype='str')

        # A.M. :there is a gap between septum surface and the epicardial
        #   Which needs to be closed if the RV/LV epicardial volume is needed
        #   this gap can be closed by using the et_thru_wall facets
        et_thru_wall = np.loadtxt(thru_wall_file, delimiter='\t').astype(int)-1

        ## convert labels to integer corresponding to the sorted list
        # of unique labels types
        unique_material = np.unique(mat[:,1])

        materials = np.zeros(mat.shape)
        for index, m in enumerate(unique_material):
            face_index = mat[:, 1] == m
            materials[face_index, 0] = mat[face_index, 0].astype(int)
            materials[face_index, 1] = [index] * np.sum(face_index)

        # add material for the new facets
        new_elem_mat = [list(range(materials.shape[0], materials.shape[0] + et_thru_wall.shape[0])),
                        [len(unique_material)] * len(et_thru_wall)]

        vertices = np.dot(subdivision_matrix, control_points)
        faces = np.concatenate((faces.astype(int), et_thru_wall))
        materials = np.concatenate((materials.T, new_elem_mat), axis=1).T.astype(int)

        model = Mesh('mesh')
        model.set_nodes(vertices)
        model.set_elements(faces)
        model.set_materials(materials[:, 0], materials[:, 1])

        # components list, used to get the correct mesh components:
        # ['0 AORTA_VALVE' '1 AORTA_VALVE_CUT' '2 LV_ENDOCARDIAL' '3 LV_EPICARDIAL'
        # ' 4 MITRAL_VALVE' '5 MITRAL_VALVE_CUT' '6 PULMONARY_VALVE' '7 PULMONARY_VALVE_CUT'
        # '8 RV_EPICARDIAL' '9 RV_FREEWALL' '10 RV_SEPTUM' '11 TRICUSPID_VALVE'
        # '12 TRICUSPID_VALVE_CUT', '13' THRU WALL]

        lv_endo = model.get_mesh_component([0, 2, 4], reindex_nodes=False)

        # Select RV endocardial
        rv_endo = model.get_mesh_component([6, 9, 10, 11], reindex_nodes=False)

        # switching the normal direction for the septum
        rv_endo.elements[rv_endo.materials == 10, :] = \
            np.array([rv_endo.elements[rv_endo.materials == 10, 0],
                      rv_endo.elements[rv_endo.materials == 10, 2],
                      rv_endo.elements[rv_endo.materials == 10, 1]]).T

        lv_epi = model.get_mesh_component([0, 1, 3, 4, 5, 10, 13], reindex_nodes=False)
        # switching the normal direction for the thru wall
        lv_epi.elements[lv_epi.materials == 13, :] = \
            np.array([lv_epi.elements[lv_epi.materials == 13, 0],
                      lv_epi.elements[lv_epi.materials == 13, 2],
                      lv_epi.elements[lv_epi.materials == 13, 1]]).T

        # switching the normal direction for the septum
        rv_epi = model.get_mesh_component([6, 7, 8, 10, 11, 12, 13], reindex_nodes=False)
        rv_epi.elements[rv_epi.materials == 10, :] = \
            np.array([rv_epi.elements[rv_epi.materials == 10, 0],
                      rv_epi.elements[rv_epi.materials == 10, 2],
                      rv_epi.elements[rv_epi.materials == 10, 1]]).T

        lv_endo_vol = lv_endo.get_volume()
        rv_endo_vol = rv_endo.get_volume()
        lv_epi_vol = lv_epi.get_volume()
        rv_epi_vol = rv_epi.get_volume()

        rv_mass = (rv_epi_vol - rv_endo_vol) * 1.05  # mass in grams
        lv_mass = (lv_epi_vol - lv_endo_vol) * 1.05

        # assign values to dict
        results_dict['lv_vol'] = round(lv_endo_vol, precision)
        results_dict['rv_vol'] = round(rv_endo_vol, precision)
        results_dict['lv_epivol'] = round(lv_epi_vol, precision)
        results_dict['rv_epivol'] = round(rv_epi_vol, precision)
        results_dict['lv_mass'] = round(lv_mass, precision)
        results_dict['rv_mass'] = round(rv_mass, precision)

    # append to the output_file
    with open(output_file, 'a', newline='') as f:
        # print out measurements in spreadsheet
        writer = csv.writer(f)
        writer.writerow([case_name, frame_name, results_dict['lv_vol'], results_dict['lv_mass'],
                         results_dict['rv_vol'], results_dict['rv_mass'],
                         results_dict['lv_epivol'], results_dict['rv_epivol']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', '-j', action='store', default='default', help='job identifier')
    parser.add_argument('--data-dir', '-d', action='store', help='path to data directory')
    parser.add_argument('--input-dir', '-I', action='store', default='Mesh_Outputs', help='name of input directories')
    parser.add_argument('--model-dir', '-m', action='store', help='path to model directory')
    parser.add_argument('--instance', '-i', type=int, action='store', default=2, help='instance to be processed')
    parser.add_argument('--all', '-a', action='store_true', help='process all subjects')
    parser.add_argument('--subject', '-s', action='store', help='subject id to be processed')
    parser.add_argument('--start', '-b', action='store', type=int, help='index of first subject id to be processed')
    parser.add_argument('--number', '-n', action='store', type=int, help='number of subjects to processed from first subject id')
    parser.add_argument('--precision', '-p', action='store', type=int, default=2, help='output precision')
    args = parser.parse_args()

    output_file = f'mv-{args.job}.csv'
    fieldnames = ['subject', 'timeframe',
                  'lv_vol', 'lv_mass',
                  'rv_vol', 'rv_mass',
                  'lv_epivol', 'rv_epivol']

    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    data_dir = args.data_dir
    model_dir = Path(args.model_dir)
    instance = args.instance

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

    for subject_id in subject_ids:
        mesh_dir = Path(data_dir) / subject_id / f'Instance_{instance}' / args.input_dir

        model_filepaths = Path(mesh_dir).glob('*model*.txt')

        for model_filepath in model_filepaths:
            csv_filepath = conv(model_filepath)
            find_volume(subject_id, csv_filepath, output_file, model_dir, args.precision)

if __name__ == '__main__':
    main()
