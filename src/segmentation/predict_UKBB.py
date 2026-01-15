"""
Utility script for nnUNet to predict UKBB NIFTIs (all time frames). Since creating a separate
image for each timeframe in the whole UKBB dataset would create millions of files, this script
is designed to work in batches of subjects rather than predicting the whole dataset at once.

For the given batch and view (SAX, 2Ch etc.), the script will first create a temporary nnUNet
dataset folder under nnUNet_raw. The temp dataset folder name will be either Dataset12345_temp
if a dataset with id 12345 doesn't exist or the first id in the range 12345-12445 that is free to
use. This allows running the script separately on different GPUs at once without manually
determining the temp dataset name. It is recommended you have no prior datasets in the 12345-12445
id range to make sure the script has free ids it can use. The temporary dataset will consist of
single timeframe images created from the full NIFTIs of subjects in the batch for the given view.

After the nnUNet dataset is created, the script just runs nnUNet prediction as usual.

After nnUNet prediction is complete, the script will repack the segmentations into NIFTI files
that contain all timeframes in a single file and then delete the temporary dataset.

Currently, the batch is given as a range of integers such as 1-1000 which would make the
script process the first 1000 subjects (wrt how sorted function sorts the UKBB folder). We
will probably add an alternative way to specify batches later on like giving a file with a
list of specific subjects.

The possible views are currently "SAX", "LAX_4Ch", "LAX_3Ch" and "LAX_2Ch. These are not
supplied as a separate argument. Instead, just name the model dataset so that it ends with the
name of the view. e.g. the trained SAX model can be in an nnUNet dataset called
Dataset100_UKBB_ManualSeg_SAX, the trained 2Ch model in a dataset called
Dataset101_UKBB_ManualSeg_LAX_2Ch etc."
"""

import argparse
import configparser
import logging
import os.path
import re
import shutil

import nibabel as nib
import numpy as np
import subprocess

from itertools import groupby


logger = logging.getLogger(__name__)

def split_nifti_t(subject_id: str, src: str, dest: str):
    """
    Takes a NIFTI image that includes a time dimension and splits it into multiples images, each
        image being one time frame.

    Args:
        src: The source NIFTI file that is to be split.
        dest: The destination directory where the output files will be created
    """
    input_nim = nib.load(src)

    src_split = src.split(os.sep)
    if src_split[-1][:3] == 'SAX':
        view = 'sax'
    elif src_split[-1][:7] == 'LAX_2Ch':
        view = 'lax_2ch'
    elif src_split[-1][:7] == 'LAX_3Ch':
        view = 'lax_3ch'
    elif src_split[-1][:7] == 'LAX_4Ch':
        view = 'lax_4ch'
    else:
        logger.error(f'{subject_id}: invalid view type')
        raise ValueError('Unexpected view type (Must be SAX, LAX_2Ch, LAX_3Ch or LAX_4Ch)')

    for t in range(input_nim.header['dim'][4]):
        input_data = input_nim.get_fdata()
        output_nim = nib.Nifti1Image(input_data[:, :, :, t], input_nim.affine, dtype=np.int16)
        # Change the NIFTI header as necessary
        output_nim.header.set_intent(1011)  # dimless
        output_nim.header['pixdim'][:4] = input_nim.header['pixdim'][:4]
        output_nim.header.set_xyzt_units(xyz=2, t=16)
        output_nim.header['qform_code'] = 1
        output_nim.header['sform_code'] = 1

        # write to output file
        nib.save(output_nim, os.path.join(dest, f'{subject_id}_{view}_fr_{t:02d}_0000.nii.gz'))


def repack_nifti_t(src_list: list[str], dest: str, input_nim: nib.Nifti1Image):
    """
    Takes multiple NIFTI files, each file corresponding to a different timeframe of the same
    volume and repacks them into one NIFTI file with a time dimension. Also requires the
    header of the original input NIFTI file because the time spacing can anly be recovered from
    there.

    Args:
        src_list: Source NIFTI files that are to be repacked. Each file should be a different time
            frame of the same volume.
        dest: The output NIFTI file to be created.
        input_nim: The input NIFTI file, i.e. the NIFTI image before segmentation that
            includes all timeframes in the single file.
    """
    # sort the source file names.
    src_sorted = sorted(src_list)

    # After sorting, the time frames should be in order but let's double-check to make sure.
    if int(src_sorted[0][-9:-7]) != 0:
        raise ValueError("Time frames not in correct order. Check file names and sorting code.")
    for i in range(len(src_sorted) - 1):
        if int(src_sorted[i + 1][-9:-7]) - int(src_sorted[i][-9:-7]) != 1:
            raise ValueError(
                "Time frames not in correct order. Check file names and sorting code.")

    # Determine output array size and initialize to zeros.
    pixel_array = np.zeros(input_nim.header['dim'][1:5])

    # fill the pixel_array
    for t in range(len(src_sorted)):
        nim = nib.load(src_sorted[t])
        pixel_array[:, :, :, t] = nim.get_fdata()

    # create the output nifti image
    output_nim = nib.Nifti1Image(pixel_array, input_nim.affine, dtype=np.int16)

    # Change the NIFTI header as necessary.
    output_nim.header.set_intent(2001)  # time series
    output_nim.header['xyzt_units'] = input_nim.header['xyzt_units']
    output_nim.header['qform_code'] = input_nim.header['qform_code']
    output_nim.header['sform_code'] = input_nim.header['sform_code']

    # Write the output file
    nib.save(output_nim, dest)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--profile', '-p', action='store', default='default', help='config profile to be used')
    parser.add_argument('--job', '-j', action='store', default='default', help='job name')

    parser.add_argument('--data-dir', '-d', action='store', help='path to data directory')
    parser.add_argument('--workspace-dir', '-w', action='store', help='path to workspace directory')
    parser.add_argument('--input-dir', '-I', action='store', help='name of input directories')
    parser.add_argument('--output-dir', '-o', action='store', default='nnUNet_segs', help='name of output directories')

    parser.add_argument('--instance', '-i', type=int, action='store', default=2, help='instance to be processed')

    parser.add_argument('--all', '-a', action='store_true', help='process all subjects')
    parser.add_argument('--subject', '-s', action='store', help='subject id to be processed')
    parser.add_argument('--start', '-S', action='store', type=int, help='index of first subject id to be processed')
    parser.add_argument('--number', '-n', action='store', type=int, help='number of subjects to be processed from first subject id')

    parser.add_argument('--model-datasets', nargs='+', default=['100_UKBB_Petersen_SAX', '101_UKBB_LAX_2Ch', '102_UKBB_LAX_3Ch', '103_UKBB_LAX_4Ch'], help='names of the training model datasets to be used for prediction')
    parser.add_argument('--gpu', '-g', action='store_true', help='run on gpu')

    args, _ = parser.parse_known_args()

    cfg = configparser.ConfigParser()
    cfg.read('config.ini')

    WORKSPACE_DIR = args.workspace_dir if args.workspace_dir else cfg[args.profile]['WorkspaceDir']

    os.environ['nnUNet_raw'] = os.path.join(WORKSPACE_DIR, 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = os.path.join(WORKSPACE_DIR, 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = os.path.join(WORKSPACE_DIR, 'nnUNet_results')

    nnunet_raw_dir = os.path.join(WORKSPACE_DIR, 'nnUNet_raw')
    ukbb_nifti_dir = args.data_dir if args.data_dir else cfg[args.profile]['DataDir']

    log_filename = os.path.join(ukbb_nifti_dir, f'segmentation-{args.job}.log')
    formatter = logging.Formatter(fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    sids = [name for name in os.listdir(ukbb_nifti_dir) if os.path.isdir(os.path.join(ukbb_nifti_dir, name))]

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

    logger.debug(f'starting job: {args.job}')

    for model_dataset in args.model_datasets:
        run(subject_ids, model_dataset, nnunet_raw_dir, ukbb_nifti_dir, args)

    logger.debug(f'finished job: {args.job}')

def run(subject_ids, model_dataset, nnunet_raw_dir, ukbb_nifti_dir, args):
    # determine the view type from the model dataset name
    model_dataset_name = [ds for ds in os.listdir(nnunet_raw_dir)
                          if
                          ds.startswith("Dataset" + model_dataset) and not ds.endswith(".zip")]
    if len(model_dataset_name) != 1:
        raise ValueError("Matching dataset dirs for the given model dataset ID is not 1. Check "
                         "model id and nnUNet dataset names. Something is wrong.")
    model_dataset_name = model_dataset_name[0]

    if model_dataset_name[-3:] == 'SAX':
        view = 'SAX'
    elif model_dataset_name[-7:] == 'LAX_2Ch':
        view = 'LAX_2Ch'
    elif model_dataset_name[-7:] == 'LAX_3Ch':
        view = 'LAX_3Ch'
    elif model_dataset_name[-7:] == 'LAX_4Ch':
        view = 'LAX_4Ch'
    else:
        logger.error('invalid dataset type')
        raise ValueError('Unexpected view type (The dataset name must end with SAX, LAX_2Ch, '
                         'LAX_3Ch or LAX_4Ch)')

    # Create temporary nnUNet dataset from the given batch of subjects
    # existing datasets in nnUNet_raw
    existing_datasets = os.listdir(nnunet_raw_dir)
    existing_dataset_ids = [dataset.partition("_")[0][7:] for dataset in existing_datasets]

    # determine an id for the temp dataset to be created and create the folder
    min_id = 12345
    max_id = 12445
    temp_dataset_id = "-1"
    for i in range(min_id, max_id):
        if str(i) in existing_dataset_ids:
            i = i + 1
            if i == max_id:
                raise ValueError("The 12345 - 12445 dataset id range is already used. Cannot "
                                 "create temp dataset")
        else:
            os.makedirs(os.path.join(nnunet_raw_dir, 'Dataset' + str(i) + '_temp', 'imagesTs'),
                        exist_ok=True)
            os.makedirs(os.path.join(nnunet_raw_dir, 'Dataset' + str(i) + '_temp', 'labelsTs'),
                        exist_ok=True)
            temp_dataset_id = str(i)
            break

    # split the nifti images and copy to temp dataset folder
    for subject_id in subject_ids:
        subject_dir = os.path.join(ukbb_nifti_dir, subject_id, f'Instance_{args.instance}')
        input_dir = os.path.join(subject_dir, args.input_dir) if args.input_dir else subject_dir

        if not os.path.exists(input_dir):
            logger.error(f'{subject_id}: missing input directory')
            continue

        files = sorted(os.listdir(input_dir))
        # process the view that corresponds to the model dataset
        for f in files:
            if f[:3] == view or f[:7] == view:
                logger.debug(f'{subject_id}: unpacking {os.path.join(input_dir, f)}')
                try:
                    split_nifti_t(subject_id, os.path.join(input_dir, f),
                                  os.path.join(nnunet_raw_dir, 'Dataset' + temp_dataset_id + '_temp',
                                               'imagesTs'))
                except ValueError:
                    logger.error(f'{subject_id}: error unpacking {f}')

    logger.debug('finished unpacking')

    # # run nnUNet prediction
    temp_input_folder = os.path.join(nnunet_raw_dir, 'Dataset' + temp_dataset_id + '_temp',
                                     'imagesTs')
    temp_output_folder = os.path.join(nnunet_raw_dir, 'Dataset' + temp_dataset_id + '_temp',
                                      'labelsTs')
    try:
        subprocess.run(["nnUNetv2_predict",
                        "-i", temp_input_folder,
                        "-o", temp_output_folder,
                        "-d", 'Dataset' + model_dataset,
                        '-device', 'cuda' if args.gpu else 'cpu',
                        "-c", "2d"],
                       check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f'nnUNet prediction failed: {e.output}')

    # repack the time frames of output segmentations into single file

    # group the segmentation files by subject id
    seg_files = sorted([f for f in os.listdir(temp_output_folder) if f.endswith(".nii.gz")])

    def get_subject_id(filename):
        return re.search(r'^(.+)_(sax|lax_[234]ch)_fr_\d+\.nii\.gz$', filename).group(1)

    seg_files_grouped = [(s_id, list(i)) for s_id, i in groupby(seg_files, get_subject_id)]

    # Each group consists of different timeframes belonging to the same image. Repack them
    # into a single file.
    for subject_id, group in seg_files_grouped:
        # read the input NIFTI. We need this to correctly write the header of the segmentation.
        subject_dir = os.path.join(ukbb_nifti_dir, subject_id, f'Instance_{args.instance}')
        input_dir = os.path.join(subject_dir, args.input_dir) if args.input_dir else subject_dir

        if not os.path.exists(input_dir):
            logger.error(f'{subject_id}: missing input directory')
            continue

        files = sorted(os.listdir(input_dir))

        input_filename = None
        for f in files:
            if f[:3] == view or f[:7] == view:
                input_filename = os.path.join(input_dir, f)

        if not input_filename:
            logger.error(f'{subject_id}: missing input file')
            raise ValueError("The view could not be matched to an input view. This should not "
                             "happen here so either there is a bug or the input directory changed"
                             "during execution.")

        input_nim = nib.load(input_filename)

        # determine output filename
        out_dir = os.path.join(subject_dir, f'{args.output_dir}')
        os.makedirs(out_dir, exist_ok=True)
        out_f = input_filename.split(os.sep)[-1][:-7] + '_nnUNetSeg.nii.gz'
        out_filename = os.path.join(out_dir, out_f)

        logger.debug(f'{subject_id}: packing {out_filename}')
        src_files = [os.path.join(temp_output_folder, f) for f in group]
        repack_nifti_t(src_files, out_filename, input_nim)
        try:
            repack_nifti_t(src_files, out_filename, input_nim)
        except ValueError:
            logger.error(f'{subject_id}: error packing {out_f}')

    logger.debug('finished packing')

    # Remove temporary directories
    shutil.rmtree(os.path.join(nnunet_raw_dir, 'Dataset' + temp_dataset_id + '_temp'))

if __name__ == "__main__":
    main()
