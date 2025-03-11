import argparse
import logging
import os
import typing
from collections import defaultdict
from typing import Optional
from typing import Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.affines import apply_affine
from scipy.spatial import distance
from skimage import measure

#mpl.use('TkAgg')
np.set_printoptions(suppress=True, floatmode="fixed", precision=2)

logger = logging.getLogger(__name__)

def auto_pick_es_frame(segs: list[nib.nifti1.Nifti1Image], lv_label: int = 1):
    """
    Takes the nnUNet segmentation directory of a subject and returns the time frame ID of the ES
    frame. The ES frame is chosen based on the LV volume.

    The function makes the following assumptions that may not be true for all datasets:

    1 - The LV label is the same in all views.

    2 - We have 3 LAX views.

    3 - The SAX view has at least 5 slices.

    4 - All views have the same number of timeframes.

    5 - The time desync between different views are small enough to ignore for the purposes of
    picking the ES. For all views, we pick a shared time frame id for the ES as the frame
    with the minimum sum of LV volumes across all the views. Hence, the frames must,
    at least roughly, be in sync w.r.t time along the heart cycle.

    Args:
        segs: The segmantation images for each view in the following order: SAX, LAX 2Ch, LAX 3Ch,
            LAX 4Ch.
        lv_label: The label of the LV blood pool in segmentations.
    Returns:
        es_frame_id: The time frame ID of the ES frame. -1 if an error is detected.
    """
    sax_fdata = segs[0].get_fdata()
    lax_fdata = [nim.get_fdata() for nim in segs[1:]]

    num_timeframes = sax_fdata.shape[3]

    if len(lax_fdata) != 3:
        logger.error("In auto_pick_es_frame: The number of LAX views is different than 3."
                      " Skipping instance.")
        return -1
    if sax_fdata.shape[2] < 5:
        logger.error("In auto_pick_es_frame: SAX has less than 5 slices. Skipping instance.")
        return -1

    # We'll count lv voxels in the LAX views + the 5 mid-slices of SAX
    lv_voxel_counts = np.zeros([8, num_timeframes])

    # lv voxel counts for LAX
    for i, fdata in enumerate(lax_fdata):
        for t in range(num_timeframes):
            lv_voxel_counts[i][t] = np.count_nonzero(np.rint(fdata[:, :, :, t]) == lv_label)

    # lv voxel counts for the 5 mid-slices of SAX
    mid_slice = int(sax_fdata.shape[2] / 2.0)
    for i in range(5):
        for t in range(num_timeframes):
            lv_voxel_counts[i + 3][t] = np.count_nonzero(
                np.rint(sax_fdata[:, :, mid_slice + i - 2, t]) == lv_label)

    # lv voxel counts all views summed
    lv_voxel_counts_all_summed = np.sum(lv_voxel_counts, axis=0)

    # return the time frame index where lv_voxel_counts_all_summed reaches its minimum
    return np.argmin(lv_voxel_counts_all_summed)


def get_contours_and_landmarks_2ch(seg: npt.NDArray[np.uint8], label_defs: dict,
                                   affine: npt.NDArray[float]):
    """
    Get contours and landmarks from a 2-chamber long-axis segmentation image.

    Args:
        seg: The segmentation image. NIFTI voxel coordinates. (2D ndarray (uint8))
        label_defs (dict): dictionary with definition of labels (e.g. 0: background, 1: LV etc.)
        affine: The affine transformation matrix mapping NIFTI voxel coordinates to NIFTI
            anatomical coordinates (RAS+).

    Returns:
        None if the 2ch slice fails QC, or a tuple of contours and landmarks.

        The tuple contains the following::

            {
                'lv_endo_contour': 2D ndarray (int) of points defining the LV endocardial contour.
                    NIFTI voxel coordinates,
                'lv_epi_contour': 2D ndarray (int) of points defining the LV epicardial contour.
                    NIFTI voxel coordinates,
                'mv_1': Coordinates of the first of the mitral valve end points in NIFTI voxel
                    coordinates (1D ndarray),
                'mv_2': Coordinates of the second of the mitral valve end points in NIFTI voxel
                    coordinates (1D ndarray),
                'apex': Coordinates of the apex point in NIFTI voxel coordinates (1D ndarray). Apex
                    is selected as the furthest point on lv_epi_contour from the mid-point of the
                    mitral valve plane,
            }
    """
    # Convert seg image to uint8
    seg = np.rint(seg).astype(np.uint8)

    # Set the LV endo segmentation image as the largest connected component from the 'LV' labelled
    # pixels
    lv_endo_seg = (seg == label_defs['LV']).astype(np.uint8)
    lv_endo_seg = get_largest_cc(lv_endo_seg).astype(np.uint8)

    # Set the LV epi segmentation image.
    # The myocardium may be split to two parts due to the very thin apex, so we do not apply
    # get_largest_cc() to it.
    # However, we remove small pieces, which may cause problems in determining the contours.
    lv_myo_seg = (seg == label_defs['Myo']).astype(np.uint8)
    lv_myo_seg = remove_small_cc(lv_myo_seg).astype(np.uint8)

    # Union the LV endo and myo segmentations and get the largest connected component. Set that as
    # the LV epi segmentation.
    lv_epi_seg = (lv_endo_seg | lv_myo_seg).astype(np.uint8)
    lv_epi_seg = get_largest_cc(lv_epi_seg).astype(np.uint8)

    # Extract LV endocardial contour
    contours, _ = cv2.findContours(lv_endo_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV endo contour QC
    if not basic_contours_qc(contours, contour_name="LV endo", view_name="2Ch", min_points=50):
        return None

    lv_endo_contour = contours[0][:, 0, :]
    lv_endo_contour[:, [0, 1]] = lv_endo_contour[:, [1, 0]]   # cv2-NIFTI coordinate swap

    # Extract LV epicardial contour
    contours, _ = cv2.findContours(lv_epi_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV epi contour QC
    if not basic_contours_qc(contours, contour_name="LV epi", view_name="2Ch", min_points=50):
        return None

    lv_epi_contour = contours[0][:, 0, :]
    lv_epi_contour[:, [0, 1]] = lv_epi_contour[:, [1, 0]]   # cv2-NIFTI coordinate swap

    # Set the LA segmentation image as the largest connected component from the 'LA' labelled
    # pixels
    la_seg = (seg == label_defs['LA']).astype(np.uint8)
    la_seg = get_largest_cc(la_seg).astype(np.uint8)

    # Extract LA contour
    contours, _ = cv2.findContours(la_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LA contour QC
    if not basic_contours_qc(contours, contour_name="LA", view_name="2Ch", min_points=20):
        return None

    la_contour = contours[0][:, 0, :]
    la_contour[:, [0, 1]] = la_contour[:, [1, 0]]   # cv2-NIFTI coordinate swap

    # The points on the lv_endo_contour that neighbor the left atrium are set as mv_contour
    mv_contour, _ = split_closed_contour_wrt_to_seg(lv_endo_contour, la_seg, la_contour)

    if mv_contour is None:
        logger.warning(f" Mitral valve contour extraction from 2Ch slice has failed."
                        f" Slice discarded.")
        return None

    # mv_contour QC
    min_size = 5
    if mv_contour.shape[0] < min_size:
        logger.warning(f" Mitral valve contour extracted from 2Ch slice has less than {min_size}"
                        f" points. Slice discarded.")
        return None

    # find the end points of mv contour
    distances = distance.cdist(mv_contour, mv_contour)
    mv_contour_ends = mv_contour[np.unravel_index(distances.argmax(), distances.shape), :]

    # Set the coordinates of the mv_mid as the mean of the mitral valve end points
    mv_mid = np.mean([mv_contour_ends[0], mv_contour_ends[1]], axis=0)

    # Apex is defined as the point on the LV epi contour with the highest distance from mv_mid
    distances = distance.cdist(lv_epi_contour, [mv_mid])
    apex = lv_epi_contour[distances.argmax(), :]

    # remove LV endo-epi overlapping points
    lv_endo_contour_temp = setdiff_2d_array(lv_endo_contour, lv_epi_contour)
    lv_epi_contour_temp = setdiff_2d_array(lv_epi_contour, lv_endo_contour)

    lv_endo_contour = lv_endo_contour_temp
    lv_epi_contour = lv_epi_contour_temp

    logger.info(" 2Ch slice passed QC.")

    return lv_endo_contour, lv_epi_contour, mv_contour_ends[0], mv_contour_ends[1], apex


def get_contours_and_landmarks_3ch(seg: npt.NDArray[np.uint8], label_defs: dict,
                                   affine: npt.NDArray[float]):
    """
    Extract contours and landmarks from a 3-chamber long-axis segmentation image.

    Args:
        seg: The segmentation image. NIFTI voxel coordinates. (2D ndarray (uint8))
        label_defs (dict): dictionary with definition of labels (e.g. 0: background, 1: LV etc.)
        affine: The affine transformation matrix mapping NIFTI voxel coordinates to NIFTI
            anatomical coordinates (RAS+).

    Returns:
        None if the 3ch slice fails QC, or a tuple of contours and landmarks.

        The tuple contains the following::

            {
                'lv_endo_contour': 2D ndarray (int) of points defining the LV endocardial contour.
                    NIFTI voxel coordinates,
                'lv_epi_contour': 2D ndarray (int) of points defining the LV epicardial contour.
                    NIFTI voxel coordinates,
                'aorta_1': Coordinates of the first of the aorta-lv intersection end points in
                    NIFTI voxel coordinates (1D ndarray),
                'aorta_2': Coordinates of the second of the aorta-lv intersection end points in
                    NIFTI voxel coordinates (1D ndarray),
                'mv_1': Coordinates of the first of the mitral valve end points in NIFTI voxel
                    coordinates (1D ndarray),
                'mv_2': Coordinates of the second of the mitral valve end points in NIFTI voxel
                    coordinates (1D ndarray),
                'rv_septum_contour': 2D ndarray of points defining the RV septum contour. NIFTI
                    voxel coordinates. This can also return None if the RV contours are to be
                    discarded due to failing QC,
                'rv_free_wall_contour': 2D ndarray of points defining the RV free wall contour.
                    NIFTI voxel coordinates. This can also return None if the RV contours are to be
                    discarded due to failing QC
            }
    """
    # Convert seg image to uint8
    seg = np.rint(seg).astype(np.uint8)

    # Set the LV endo segmentation image as the largest connected component from the 'LV' labelled
    # pixels
    lv_endo_seg = (seg == label_defs['LV']).astype(np.uint8)
    lv_endo_seg = get_largest_cc(lv_endo_seg).astype(np.uint8)

    # Set the LV epi segmentation image.
    # The myocardium may be split to two parts due to the very thin apex, so we do not apply
    # get_largest_cc() to it.
    # However, we remove small pieces, which may cause problems in determining the contours.
    lv_myo_seg = (seg == label_defs['Myo']).astype(np.uint8)
    lv_myo_seg = remove_small_cc(lv_myo_seg).astype(np.uint8)

    # Union the LV endo and myo segmentations and get the largest connected component. Set that as
    # the LV epi segmentation.
    lv_epi_seg = (lv_endo_seg | lv_myo_seg).astype(np.uint8)
    lv_epi_seg = get_largest_cc(lv_epi_seg).astype(np.uint8)

    # Extract LV endocardial contour
    contours, _ = cv2.findContours(lv_endo_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV endo contour QC
    if not basic_contours_qc(contours, contour_name="LV endo", view_name="3Ch", min_points=50):
        return None

    lv_endo_contour = contours[0][:, 0, :]
    lv_endo_contour[:, [0, 1]] = lv_endo_contour[:, [1, 0]]   # cv2-NIFTI coordinate swap

    # Extract LV epicardial contour
    contours, _ = cv2.findContours(lv_epi_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV epi contour QC
    if not basic_contours_qc(contours, contour_name="LV epi", view_name="3Ch", min_points=50):
        return None

    lv_epi_contour = contours[0][:, 0, :]
    lv_epi_contour[:, [0, 1]] = lv_epi_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Set the aorta segmentation image as the largest connected component from the 'Ao' labelled
    # pixels
    aorta_seg = (seg == label_defs['Ao']).astype(np.uint8)
    aorta_seg = get_largest_cc(aorta_seg).astype(np.uint8)

    # Extract aorta contour
    contours, _ = cv2.findContours(aorta_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # aorta contour QC
    if not basic_contours_qc(contours, contour_name="aorta", view_name="3Ch", min_points=20):
        return None

    aorta_contour = contours[0][:, 0, :]
    aorta_contour[:, [0, 1]] = aorta_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Set the LA segmentation image as the largest connected component from the 'LA' labelled
    # pixels
    la_seg = (seg == label_defs['LA']).astype(np.uint8)
    la_seg = get_largest_cc(la_seg).astype(np.uint8)

    # Extract LA contour
    contours, _ = cv2.findContours(la_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LA contour QC
    if not basic_contours_qc(contours, contour_name="LA", view_name="3Ch", min_points=50):
        return None

    la_contour = contours[0][:, 0, :]
    la_contour[:, [0, 1]] = la_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # The points on the lv_endo_contour that neighbor the aorta are set as aorta_lv_contour
    aorta_lv_contour, _ = split_closed_contour_wrt_to_seg(lv_endo_contour, aorta_seg, aorta_contour)

    # The points on the lv_endo_contour that neighbor the left atrium are set as mv_contour
    mv_contour, _ = split_closed_contour_wrt_to_seg(lv_endo_contour, la_seg, la_contour)

    if aorta_lv_contour is None:
        logger.warning(f" aorta_lv_contour extraction from 3Ch slice has failed. "
                        f" Slice discarded.")
        return None

    # aorta_lv_contour QC
    min_size = 5
    if aorta_lv_contour.shape[0] < min_size:
        logger.warning(f" aorta_lv_contour extracted from 3Ch slice has less than {min_size} "
                        f" points. Slice discarded.")
        return None

    if mv_contour is None:
        logger.warning(f" Mitral valve contour extraction from 3Ch slice has failed."
                        f" Slice discarded.")
        return None

    # mv_contour QC
    min_size = 5
    if mv_contour.shape[0] < min_size:
        logger.warning(f" Mitral valve contour extracted from 3Ch slice has less than {min_size}"
                        f" points. Slice discarded.")
        return None

    # find the end points of mv contour
    distances = distance.cdist(mv_contour, mv_contour)
    mv_contour_ends = mv_contour[np.unravel_index(distances.argmax(), distances.shape), :]
    mv_1 = mv_contour_ends[0]
    mv_2 = mv_contour_ends[1]

    # find the end points of aorta-lv contour
    distances = distance.cdist(aorta_lv_contour, aorta_lv_contour)
    aorta_lv_contour_ends = aorta_lv_contour[np.unravel_index(distances.argmax(), distances.shape),
                            :]
    aorta_1 = aorta_lv_contour_ends[0]
    aorta_2 = aorta_lv_contour_ends[1]

    # Extract the RV contour. In 3 chamber view, the rv segmentation isn't very
    # reliable, so if RV QC fails, we will only discard the RV contour instead of the whole slice.
    discard_rv_flag = False

    # Set the RV segmentation image as the largest connected component from the 'RV' labelled
    # pixels
    rv_seg = (seg == label_defs['RV']).astype(np.uint8)
    rv_seg = get_largest_cc(rv_seg).astype(np.uint8)

    # Extract the RV contour
    contours, _ = cv2.findContours(rv_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # RV contour QC
    if not basic_contours_qc(contours, contour_name="RV", view_name="3Ch", min_points=30):
        return lv_endo_contour, lv_epi_contour, aorta_1, aorta_2, mv_1, mv_2, None, None

    rv_contour = contours[0][:, 0, :]
    rv_contour[:, [0, 1]] = rv_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Set the points on rv_contour that neighbor lv epi as septum, set the rest as free wall.
    rv_septum_contour, rv_free_wall_contour = (
        split_closed_contour_wrt_to_seg(rv_contour, lv_epi_seg, lv_epi_contour))

    if rv_septum_contour is None:
        logger.warning(f"RV Septum - FW separation failed on 3Ch slice.")
        logger.info(" Excluding the rv, 3Ch slice passed QC.")
        return lv_endo_contour, lv_epi_contour, aorta_1, aorta_2, mv_1, mv_2, None, None

    # RV septum and free wall QC
    min_size = 15
    if rv_septum_contour.shape[0] < min_size:
        logger.warning(f" RV septum contour extracted from 3Ch slice has less than {min_size}"
                        f" points. RV contours discarded.")
        logger.info(" Excluding the rv, 3Ch slice passed QC.")
        return lv_endo_contour, lv_epi_contour, aorta_1, aorta_2, mv_1, mv_2, None, None
    if rv_free_wall_contour.shape[0] < min_size:
        logger.warning(f" RV free wall contour extracted from 3Ch slice has less than {min_size}"
                        f" points. RV contours discarded.")
        logger.info(" Excluding the rv, 3Ch slice passed QC.")
        return lv_endo_contour, lv_epi_contour, aorta_1, aorta_2, mv_1, mv_2, None, None

    # Find points on the lv_epi_contour that neighbor the RV.
    lv_epi_rv_neighbors, _ = split_closed_contour_wrt_to_seg(lv_epi_contour, rv_seg, rv_contour)

    # Finally, we do the removals
    # Remove LV endo-epi overlapping points
    lv_endo_contour_temp = setdiff_2d_array(lv_endo_contour, lv_epi_contour)
    lv_epi_contour_temp = setdiff_2d_array(lv_epi_contour, lv_endo_contour)

    lv_endo_contour = lv_endo_contour_temp
    lv_epi_contour = lv_epi_contour_temp

    # Remove lv_epi_rv_neighbors from lv_epi
    if lv_epi_rv_neighbors is not None:
        lv_epi_contour = setdiff_2d_array(lv_epi_contour, lv_epi_rv_neighbors)

    logger.info(" 3Ch slice passed QC.")

    return lv_endo_contour, lv_epi_contour, aorta_1, aorta_2, mv_1, mv_2, rv_septum_contour, rv_free_wall_contour


def get_contours_and_landmarks_4ch(seg: npt.NDArray[np.uint8], label_defs: dict,
                                   affine: npt.NDArray[float]):
    """
    Extract contours and landmarks from a 4-chamber long-axis segmentation image.

    Args:
        seg: The segmentation image. NIFTI voxel coordinates. (2D ndarray (uint8))
        label_defs (dict): dictionary with definition of labels (e.g. 0: background, 1: LV etc.)
        affine: The affine transformation matrix mapping NIFTI voxel coordinates to NIFTI
            anatomical coordinates (RAS+).

    Returns:
        None if the 4ch slice fails QC, or a tuple of contours and landmarks.

        The tuple contains the following::

            {
                'lv_endo_contour': 2D ndarray (int) of points defining the LV endocardial contour.
                    NIFTI voxel coordinates,
                'lv_epi_contour': 2D ndarray (int) of points defining the LV epicardial contour.
                    NIFTI voxel coordinates,
                'rv_septum_contour': 2D ndarray of points defining the RV septum contour. NIFTI
                    voxel coordinates. This can also return None if the RV contours are to be
                    discarded due to failing QC,
                'rv_free_wall_contour': 2D ndarray of points defining the RV free wall contour.
                    NIFTI voxel coordinates. This can also return None if the RV contours are to be
                    discarded due to failing QC,
                'mv_left': Coordinates of the left-most mitral valve point in NIFTI voxel
                    coordinates (1D ndarray),
                'mv_right': Coordinates of the right-most mitral valve point in NIFTI voxel
                    coordinates (1D ndarray),
                'tv_left': Coordinates of the left-most tricuspid valve point in NIFTI voxel
                    coordinates (1D ndarray),
                'tv_right': Coordinates of the right-most tricuspid valve point in NIFTI voxel
                    coordinates (1D ndarray),
            }
    """
    # Convert seg image to uint8
    seg = np.rint(seg).astype(np.uint8)

    # Set the LV endo segmentation image as the largest connected component from the 'LV' labelled
    # pixels
    lv_endo_seg = (seg == label_defs['LV']).astype(np.uint8)
    lv_endo_seg = get_largest_cc(lv_endo_seg).astype(np.uint8)

    # Set the LV epi segmentation image.
    # The myocardium may be split to two parts due to the very thin apex, so we do not apply
    # get_largest_cc() to it.
    # However, we remove small pieces, which may cause problems in determining the contours.
    lv_myo_seg = (seg == label_defs['Myo']).astype(np.uint8)
    lv_myo_seg = remove_small_cc(lv_myo_seg).astype(np.uint8)

    # Union the LV endo and myo segmentations and get the largest connected component. Set that as
    # the LV epi segmentation.
    lv_epi_seg = (lv_endo_seg | lv_myo_seg).astype(np.uint8)
    lv_epi_seg = get_largest_cc(lv_epi_seg).astype(np.uint8)

    # Extract LV endocardial contour
    contours, _ = cv2.findContours(lv_endo_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV endo contour QC
    if not basic_contours_qc(contours, contour_name="LV endo", view_name="4Ch", min_points=50):
        return None

    lv_endo_contour = contours[0][:, 0, :]
    lv_endo_contour[:, [0, 1]] = lv_endo_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Extract LV epicardial contour
    contours, _ = cv2.findContours(lv_epi_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV epi contour QC
    if not basic_contours_qc(contours, contour_name="LV epi", view_name="4Ch", min_points=50):
        return None

    lv_epi_contour = contours[0][:, 0, :]
    lv_epi_contour[:, [0, 1]] = lv_epi_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Set the LA segmentation image as the largest connected component from the 'LA' labelled
    # pixels
    la_seg = (seg == label_defs['LA']).astype(np.uint8)
    la_seg = get_largest_cc(la_seg).astype(np.uint8)

    # Extract LA contour
    contours, _ = cv2.findContours(la_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LA contour QC
    if not basic_contours_qc(contours, contour_name="LA", view_name="4Ch", min_points=20):
        return None

    la_contour = contours[0][:, 0, :]
    la_contour[:, [0, 1]] = la_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Set the RA segmentation image as the largest connected component from the 'RA' labelled
    # pixels
    ra_seg = (seg == label_defs['RA']).astype(np.uint8)
    ra_seg = get_largest_cc(ra_seg).astype(np.uint8)

    # Extract RA contour
    contours, _ = cv2.findContours(ra_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # RA contour QC
    if not basic_contours_qc(contours, contour_name="RA", view_name="4Ch", min_points=20):
        return None

    ra_contour = contours[0][:, 0, :]
    ra_contour[:, [0, 1]] = ra_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # The points on the lv_endo_contour that neighbor the left atrium are set as mv_contour
    mv_contour, _ = split_closed_contour_wrt_to_seg(lv_endo_contour, la_seg, la_contour)

    if mv_contour is None:
        logger.warning(f" Mitral valve contour extraction from 4Ch slice has failed."
                        f" Slice discarded.")
        return None

    # mv_contour QC
    min_size = 5
    if mv_contour.shape[0] < min_size:
        logger.warning(f" Mitral valve contour extracted from 4Ch slice has less than {min_size}"
                        f" points. Slice discarded.")
        return None

    # Now that we have the mv_contour, we want to determine the 2 anatomical end-points
    # on the contour. For 4-chamber view, selecting the left-most and right-most points
    # should work fine in most cases.
    mv_contour_anatomical = convert_contour_to_anatomical_coor(mv_contour, affine)

    mv_left = mv_contour[np.argmin(mv_contour_anatomical, axis=0)[0], :]
    mv_right = mv_contour[np.argmax(mv_contour_anatomical, axis=0)[0], :]

    # Get the RV contour (only endo will be extracted from the RV since epi is not segmented)
    rv_seg = (seg == label_defs['RV']).astype(np.uint8)
    rv_seg = get_largest_cc(rv_seg).astype(np.uint8)

    contours, _ = cv2.findContours(rv_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # RV contour QC
    if not basic_contours_qc(contours, contour_name="RV", view_name="4Ch", min_points=50):
        return None

    rv_contour = contours[0][:, 0, :]
    rv_contour[:, [0, 1]] = rv_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Set the points on rv_contour that neighbor lv epi as septum, set the rest as free wall.
    rv_septum_contour, rv_free_wall_contour = (
        split_closed_contour_wrt_to_seg(rv_contour, lv_epi_seg, lv_epi_contour))

    if rv_septum_contour is None:
        logger.warning(f"RV Septum - FW separation failed on 4Ch slice. Slice discarded.")
        return None

    # RV septum and free wall QC
    min_size = 15
    if rv_septum_contour.shape[0] < min_size:
        logger.warning(f" RV septum contour extracted from 4Ch slice has less than {min_size}"
                        f" points. Slice discarded.")
        return None
    if rv_free_wall_contour.shape[0] < min_size:
        logger.warning(f" RV free wall contour extracted from 4Ch slice has less than {min_size}"
                        f" points. Slice discarded.")
        return None

    # The points on the rv_contour that neighbor the RA are set as tv_contour
    tv_contour, _ = split_closed_contour_wrt_to_seg(rv_contour, ra_seg, ra_contour)

    if tv_contour is None:
        logger.warning(
            f" Tricuspid valve contour extraction from 4Ch slice has failed."
            f" Slice discarded.")
        return None

    # tv_contour QC
    min_size = 5
    if tv_contour.shape[0] < min_size:
        logger.warning(
            f" Tricuspid valve contour extracted from 4Ch slice has less than {min_size}"
            f" points. Slice discarded.")
        return None

    # set the left and right end-points of the tricuspid valve as the points on the tv contour
    # that are closest to and furthest from the right end-point of the mitral valve
    tv_left = closest_node(mv_right, tv_contour)
    tv_right = furthest_node(mv_right, tv_contour)

    # Find points on the rv contour that neighbor the tricuspid valve plane.
    rv_tv_neighbors, _ = split_closed_contour_wrt_to_seg(rv_contour, ra_seg, ra_contour)

    # Find points on the lv_epi_contour that neighbor the RV.
    lv_epi_rv_neighbors, _ = split_closed_contour_wrt_to_seg(lv_epi_contour, rv_seg, rv_contour)

    # Finally, we do the removals
    # Remove LV endo-epi overlapping points
    lv_endo_contour_temp = setdiff_2d_array(lv_endo_contour, lv_epi_contour)
    lv_epi_contour_temp = setdiff_2d_array(lv_epi_contour, lv_endo_contour)

    lv_endo_contour = lv_endo_contour_temp
    lv_epi_contour = lv_epi_contour_temp

    # Remove lv_epi_rv_neighbors from lv_epi
    if lv_epi_rv_neighbors is not None:
        lv_epi_contour = setdiff_2d_array(lv_epi_contour, lv_epi_rv_neighbors)

    # Remove rv_tv_neighbors from rv contours
    if rv_tv_neighbors is not None:
        rv_septum_contour = setdiff_2d_array(rv_septum_contour, rv_tv_neighbors)
        rv_free_wall_contour = setdiff_2d_array(rv_free_wall_contour, rv_tv_neighbors)

    logger.info(" 4Ch slice passed QC.")

    return lv_endo_contour, lv_epi_contour, rv_septum_contour, rv_free_wall_contour, \
        mv_left, mv_right, tv_left, tv_right


def get_contours_sax_slice(seg: npt.NDArray[np.uint8], label_defs: dict, slice_id: int):
    """
    Extract LV endocardium, LV epicardium, RV septum and RV free wall contours from a 2D slice
    of a short axis segmentation image

    Args:
        seg: The segmentation image. NIFTI voxel coordinates. (2D ndarray (uint8))
        label_defs (dict): dictionary with definition of labels (e.g. 0: background, 1: LV etc.)
        slice_id: ID of the slice in the SAX segmentation currently being processed

    Returns:
        None if the slice fails QC, or a tuple of contours and landmarks.

        The tuple contains the following::

            {
                'lv_endo_contour': 2D ndarray (int) of points defining the LV endocardial contour.
                    NIFTI voxel coordinates,
                'lv_epi_contour': 2D ndarray (int) of points defining the LV epicardial contour.
                    NIFTI voxel coordinates,
                'rv_septum_contour': 2D ndarray of points defining the RV septum contour. NIFTI
                    voxel coordinates. This can also return None if the RV contours are to be
                    discarded due to failing QC,
                'rv_free_wall_contour': 2D ndarray of points defining the RV free wall contour.
                    NIFTI voxel coordinates. This can also return None if the RV contours are to be
                    discarded due to failing QC,
            }
    """
    # Convert seg image to uint8
    seg = np.rint(seg).astype(np.uint8)

    # Set the LV endo segmentation image as the largest connected component from the 'LV' labelled
    # pixels
    lv_endo_seg = (seg == label_defs['LV']).astype(np.uint8)
    lv_endo_seg = get_largest_cc(lv_endo_seg).astype(np.uint8)

    # Set the LV epi segmentation image.
    # The myocardium may be split to two parts due to the very thin apex, so we do not apply
    # get_largest_cc() to it.
    # However, we remove small pieces, which may cause problems in determining the contours.
    lv_myo_seg = (seg == label_defs['Myo']).astype(np.uint8)
    lv_myo_seg = remove_small_cc(lv_myo_seg).astype(np.uint8)

    # Union the LV endo and myo segmentations and get the largest connected component. Set that as
    # the LV epi segmentation.
    lv_epi_seg = (lv_endo_seg | lv_myo_seg).astype(np.uint8)
    lv_epi_seg = get_largest_cc(lv_epi_seg).astype(np.uint8)

    # Extract LV endocardial contour
    contours, _ = cv2.findContours(lv_endo_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV endo QC
    if not basic_contours_qc(contours, contour_name="LV endo", view_name="SAX", min_points=10,
                             slice_id=slice_id):
        return None

    lv_endo_contour = contours[0][:, 0, :]
    lv_endo_contour[:, [0, 1]] = lv_endo_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Extract LV epicardial contour
    contours, _ = cv2.findContours(lv_epi_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # LV epi contour QC
    if not basic_contours_qc(contours, contour_name="LV epi", view_name="SAX", min_points=10,
                             slice_id=slice_id):
        return None

    lv_epi_contour = contours[0][:, 0, :]
    lv_epi_contour[:, [0, 1]] = lv_epi_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Get the RV contour (only endo will be extracted from the RV since epi is not segmented)
    rv_seg = (seg == label_defs['RV']).astype(np.uint8)
    rv_seg = get_largest_cc(rv_seg).astype(np.uint8)

    contours, _ = cv2.findContours(rv_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # RV contour QC
    if not basic_contours_qc(contours, contour_name="RV", view_name="SAX", min_points=10,
                             slice_id=slice_id):
        return None

    rv_contour = contours[0][:, 0, :]
    rv_contour[:, [0, 1]] = rv_contour[:, [1, 0]]  # cv2-NIFTI coordinate swap

    # Set the points on rv_contour that neighbor lv epi as septum, set the rest as free wall.
    rv_septum_contour, rv_free_wall_contour = (
        split_closed_contour_wrt_to_seg(rv_contour, lv_epi_seg, lv_epi_contour))

    if rv_septum_contour is None:
        logger.warning(f"RV Septum - FW separation failed on SAX slice {slice_id}")
        return None

    # Find points on the lv_epi_contour that neighbor the RV.
    lv_epi_rv_neighbors, _ = split_closed_contour_wrt_to_seg(lv_epi_contour, rv_seg, rv_contour)

    # Finally, we do the removals
    # Remove LV endo-epi overlapping points
    lv_endo_contour_temp = setdiff_2d_array(lv_endo_contour, lv_epi_contour)
    lv_epi_contour_temp = setdiff_2d_array(lv_epi_contour, lv_endo_contour)

    lv_endo_contour = lv_endo_contour_temp
    lv_epi_contour = lv_epi_contour_temp

    # Remove lv_epi_rv_neighbors from lv_epi
    if lv_epi_rv_neighbors is not None:
        lv_epi_contour = setdiff_2d_array(lv_epi_contour, lv_epi_rv_neighbors)

    logger.info(f" SAX slice {slice_id} passed QC.")

    return lv_endo_contour, lv_epi_contour, rv_septum_contour, rv_free_wall_contour


def basic_contours_qc(contours, contour_name: str, view_name: str, min_points: int,
                      slice_id: int = 0):
    """

    Args:
        contours: The contours returned by cv2.findContours()
        contour_name: The name of the contour. Used for logging purposes (e.g. "RV endo", "LV epi")
        view_name: The name of the view. Used for logging purposes. (e.g. "SAX", "3Ch")
        min_points: The minimum number of points on contours[0] for it to pass QC.
        slice_id: ID of the slice. Used for logging purposes. Only relevant in SAX since the other
            views are 2D.

    Returns:
        True if contour passes QC, False otherwise

    """
    if view_name == "SAX":
        if len(contours) == 0:
            logger.warning(f" No {contour_name} contour detected on slice {slice_id} of"
                            f" {view_name}. Slice discarded.")
            return False
        elif len(contours) > 1:
            logger.warning(f" More than 1 {contour_name} contour detected on slice {slice_id} of"
                            f" {view_name}. Slice discarded.")
            return False
        elif contours[0].size < min_points:
            logger.warning(f" The detected {contour_name} contour on slice {slice_id} of "
                            f" {view_name} has less than {min_points} points. Slice discarded.")
            return False
    else:
        if len(contours) == 0:
            logger.warning(f" No {contour_name} contour detected on"
                            f" {view_name}. Slice discarded.")
            return False
        elif len(contours) > 1:
            logger.warning(f" More than 1 {contour_name} contour detected on"
                            f" {view_name}. Slice discarded.")
            return False
        elif contours[0].size < min_points:
            logger.warning(f" The detected {contour_name} contour on"
                            f" {view_name} has less than {min_points} points. Slice discarded.")
            return False

    return True


def setdiff_2d_array(arr1: npt.NDArray[int], arr2: npt.NDArray[int]) -> npt.NDArray[int]:
    """
    Takes two 2D numpy arrays and returns a 2D numpy array that is the set difference
    between them. This is used to remove overlapping points in 2 contours.
    """
    delta = set(map(tuple, arr2))
    return np.array([x for x in arr1 if tuple(x) not in delta])


def convert_contour_to_anatomical_coor(contour: npt.NDArray[int], affine: npt.NDArray[float]) -> \
        npt.NDArray[float]:
    """
    Converts a contour in NIFTI voxel coordinates to a contour in NIFTI anatomical coordinates

    Args:
        contour: ndarray of points in NIFTI voxel coordinates
        affine: The affine transformation matrix mapping NIFTI voxel coordinates to NIFTI
            anatomical coordinates (RAS+).

    Returns:
        contour_anatomical: ndarray of points in NIFTI anatomical coordinates. This is the input
            contour converted to anatomical coordinates.
    """
    contour_anatomical = np.zeros([contour.shape[0], 3])

    for i, point in enumerate(contour):
        point_anatomical = apply_affine(affine, np.append(point, 0))
        contour_anatomical[i] = point_anatomical

    return contour_anatomical


def detect_neighbors_to_seg(contour: npt.NDArray[int], seg: npt.NDArray[np.uint8]) -> Tuple[
    npt.NDArray[int], npt.NDArray[int]]:
    """
    Detect the points on a contour that are neighbors of a segmented region.

    Args:
        contour: 2D ndarray of points defining the contour.
        seg: Segmentation image (2D ndarray).

    Returns:
        Two arrays of points, the first one containing the points that are neighbors and the second
            one containing points that are not.
    """

    # define convolution kernel to help detect neighbors
    kernel = np.uint8([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    filtered = cv2.filter2D(seg, ddepth=-1, kernel=kernel)

    neighbor_pts = []
    non_neighbor_pts = []

    for pt in contour:
        if filtered[tuple(pt)] > 0:
            neighbor_pts.append(pt)
        else:
            non_neighbor_pts.append(pt)

    return np.asarray(neighbor_pts), np.asarray(non_neighbor_pts)


def calc_idx_distance_on_closed_contour(idx1: int, idx2: int, len_contour: int):
    """
    Calculate the index distance on a closed contour. E.g. index 3 and 5 will have distance 2.
    Index 0 and Index 49 will have distance 1 if len_contour = 50.
    """
    if idx1 > idx2:
        big_index = idx1
        small_index = idx2
    else:
        big_index = idx2
        small_index = idx1

    return min(big_index - small_index, small_index + len_contour - big_index)


def find_gaps(idx1: int, idx2: int, len_contour: int):
    """
    Find and return gaps between two contiguous parts of a closed contour. idx1 and idx2 are
    the indices that are closest between the two contiguous parts.
    """
    gaps = []

    if idx1 > idx2:
        big_index = idx1
        small_index = idx2
    else:
        big_index = idx2
        small_index = idx1

    if big_index - small_index < small_index + len_contour - big_index:
        for idx in range(small_index + 1, big_index):
            gaps.append(idx)
    else:
        for idx in range(big_index + 1, len_contour):
            gaps.append(idx)
        for idx in range(small_index):
            gaps.append(idx)

    return gaps


def split_closed_contour_wrt_to_seg(contour: np.ndarray, seg: np.ndarray,
                                    seg_contour: np.ndarray, dist_threshold: int = 2):
    """
    Split a contour into two contours, the first one consisting of points that neighbor the seg,
    the second one consisting of points that don't. Both contours must be contiguous, so if
    the neighboring points are not contiguous, either a False is returned or the neighboring
    points are connected depending on the maximum allowable distance threshold.

    Args:
        contour: 2D ndarray of points defining the contour.
        seg: Segmentation image (2D ndarray).
        seg_contour: contour enclosing the segmentation region
        dist_threshold: The maximum allowable distance from seg that can be considered a neighbor.
            This is only used to fix non-contiguous contours, it does not mean all points below
            this distance are taken as neighbors.

    Returns:
        On success, return two arrays of points, the first one containing the points that are
        neighbors and the second one containing points that are not. On fail, return None.
    """

    # define convolution kernel to help detect neighbors
    kernel = np.uint8([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    filtered = cv2.filter2D(seg, ddepth=-1, kernel=kernel)

    neighbor_idxs = []
    neighbor_idxs_contiguous_part = []

    # detect points neighboring the seg mask
    for i, pt in enumerate(contour):
        if filtered[tuple(pt)] > 0:
            if len(neighbor_idxs_contiguous_part) == 0:
                neighbor_idxs_contiguous_part.append(i)
            else:
                if neighbor_idxs_contiguous_part[-1] + 1 == i:
                    neighbor_idxs_contiguous_part.append(i)
                else:
                    neighbor_idxs.append(neighbor_idxs_contiguous_part)
                    neighbor_idxs_contiguous_part = [i]

    if len(neighbor_idxs_contiguous_part) > 0:
        neighbor_idxs.append(neighbor_idxs_contiguous_part)

    # print(f"Neighbor indices = {neighbor_idxs}")

    # Merge the contiguous parts if possible
    neighbor_idxs_contiguous = []
    if len(neighbor_idxs) == 1:
        neighbor_idxs_contiguous = neighbor_idxs[0].copy()

    if len(neighbor_idxs) > 1:
        neighbor_idxs_contiguous = neighbor_idxs[0].copy()
        for i in range(1, len(neighbor_idxs)):
            # Attach the next part, filling gaps if necessary
            min_idx_1 = np.min(neighbor_idxs_contiguous)
            max_idx_1 = np.max(neighbor_idxs_contiguous)

            min_idx_2 = np.min(neighbor_idxs[i])
            max_idx_2 = np.max(neighbor_idxs[i])

            distances = np.zeros([4])
            distances[0] = calc_idx_distance_on_closed_contour(min_idx_1, min_idx_2, len(contour))
            distances[1] = calc_idx_distance_on_closed_contour(min_idx_1, max_idx_2, len(contour))
            distances[2] = calc_idx_distance_on_closed_contour(max_idx_1, min_idx_2, len(contour))
            distances[3] = calc_idx_distance_on_closed_contour(max_idx_1, max_idx_2, len(contour))

            closest_dist_idx = np.argmin(distances)

            if closest_dist_idx == 0:
                gap_idxs = find_gaps(min_idx_1, min_idx_2, len(contour))
            elif closest_dist_idx == 1:
                gap_idxs = find_gaps(min_idx_1, max_idx_2, len(contour))
            elif closest_dist_idx == 2:
                gap_idxs = find_gaps(max_idx_1, min_idx_2, len(contour))
            elif closest_dist_idx == 3:
                gap_idxs = find_gaps(max_idx_1, max_idx_2, len(contour))
            else:
                raise ValueError("Invalid closest_dist_idx")

            if len(gap_idxs) > 0:
                gap_pts = np.zeros([len(gap_idxs), 2])
                for j in range(len(gap_idxs)):
                    gap_pts[j] = contour[gap_idxs[j]]

                gap_pts_to_seg_distances = distance.cdist(gap_pts, seg_contour)
                gap_pts_to_seg_min_distances = np.min(gap_pts_to_seg_distances, axis=1)

                # if a gap point has a higher distance to the seg than dist_threshold, return None
                if np.min(gap_pts_to_seg_min_distances) > dist_threshold:
                    return None, None
                else:
                    # Add gaps to neighbor indices
                    neighbor_idxs_contiguous.extend(gap_idxs)
                    neighbor_idxs_contiguous.extend(neighbor_idxs[i])
            # if no gaps, this means the two parts are at the end and start of the contour, and
            # we can simply attach them.
            else:
                neighbor_idxs_contiguous.extend(neighbor_idxs[i])

    neighbor_pts = []
    non_neighbor_pts = []

    # print(f"Neighbor indices = {neighbor_idxs}")
    # print(f"Neighbor indices contiguous = {neighbor_idxs_contiguous}")
    # input()

    for i in range(len(contour)):
        if i in neighbor_idxs_contiguous:
            neighbor_pts.append(contour[i])
        else:
            non_neighbor_pts.append(contour[i])

    # print(f"neighbor_pts = {neighbor_pts}")

    return np.array(neighbor_pts), np.array(non_neighbor_pts)


def write_contour_to_gp_points_file(file: typing.TextIO, contour: np.ndarray,
                                    affine: npt.NDArray[float], ctype: str, frame_id: int,
                                    time_frame_id: int, weight: Optional[float] = 1.0,
                                    slice_id: Optional[int] = 0):
    """
    Transform contour coordinates from NIFTI voxel to DICOM anatomical and write contour points
    to a gp points file. Each line in the gp points file records one point and will have the
    following information separated by a tab: x, y, z, contour_type, frame_id, weight and
    time_frame_id where x, y, z are the coordinates in the DICOM anatomical coordinate system.

    Args:
        file: The output gp points file.
        contour: The contour to be written to file. 2D ndarray of points in NIFTI (nibabel) voxel
            coordinates.
        affine: The affine transformation matrix from NIFTI (nibabel) voxel coordinates to
            NIFTI (nibabel) anatomical coordinates.
        ctype: The type (label) of the contour (e.g. 'rv_apex_point', 'mitral_valve' etc.)
        frame_id: the frame ID that was assigned for the gp output, to the 2D slice from
            which the contour was extracted.
        time_frame_id: ID of the time frame from which the contour was extracted.
        weight: The weight assigned to the contour for subsequent mesh fitting.
        slice_id: If the image the slice is from is a 3D short axis image, slice_id has to be
            supplied so that the voxel to anatomical coordinates transformation can be correctly
            calculated. It is assumed that the slice_id is the z-coordinate of the SAX image and
            this corresponds to the low-resolution axis.
    """
    # if contour is None or empty, don't do anything
    if contour is None or contour.size == 0:
        return

    # if contour is 1D (has only one point), expand to 2D.
    if contour.ndim == 1:
        contour = np.expand_dims(contour, axis=0)

    # apply affine transform after adding the slice_id to the point as the 3rd dimension.
    # (because slice_id is the z-coordinate in voxel coordinates)
    points_world = apply_affine(affine,
                                np.hstack((contour, np.full((contour.shape[0], 1), slice_id))))

    # change from NIFTI world coordinates to DICOM world coordinates
    points_world = points_world * np.array([-1, -1, 1])

    for pt in points_world:
        # # You can uncomment this when debugging. Output is prettier but surface mesh code can't
        # # read it properly.
        # file.write(f"{round(pt[0], 4):.2f}\t{round(pt[1], 4):.2f}\t{round(pt[2], 4):.2f}"
        #            f"\t{ctype:^20}\t{frame_id:^10}\t{weight:^9}\t{time_frame_id:^14}\n")

        file.write(f"{round(pt[0], 4):.2f}\t{round(pt[1], 4):.2f}\t{round(pt[2], 4):.2f}"
                   f"\t{ctype}\t{frame_id}\t{weight}\t{time_frame_id}\n")


def write_to_gp_frame_info_file(gp_frame_info_file: typing.TextIO, frame_id: int,
                                time_frame_id: int, header: nib.nifti1.Nifti1Header,
                                slice_id: Optional[int] = 0, slice_info: Optional[str] = 'N/A'):
    """
    Writes the info of the frame (a 2D slice) with frame_id to frame_info_file. The info is
    written as a line to the file and includes the following information: slice_info, frameID,
    timeFrameID (id of the time frame), ImagePositionPatient (x, y, z offset that determines where
    the (0,0) pixel in the slice is located with respect to DICOM BIPED anatomical (a.k.a. world or
    physical) coordinates (LPS+). Corresponds to (0020,0032) attribute in DICOM metadata),
    ImageOrientationPatient (6 values that define the direction cosines giving the orientation
    of the x and y axes of the image data with respect to the DICOM anatomical coordinate
    system. Corresponds to DICOM attribute (0020,0037)), PixelSpacing (the physical spacing
    between pixel centers. Corresponds to DICOM attribute (0028, 0030)).

    WARNING: This function uses the qform of the NIFTI header to figure out the correct
    transformation from NIFTI voxel to DICOM anatomical coordinates. If the qform is not defined or
    does not correspond to the NIFTI voxel to NIFTI anatomical coordinate (RAS+) transformation,
    this function will not work correctly. If such images are in the dataset, either this function
    or the NIFTI header should be modified accordingly, whichever is more appropriate for the
    application.

    Args:
        gp_frame_info_file: The gp frame info file that is used as input to the subsequent mesh
            fitting step along with the gp points file.
        frame_id: The frame ID that was assigned to the 2D slice for the gp file.
        time_frame_id: The time frame ID that was assigned to the 2D slice for the gp file.
        header: NIFTI header associated with the slice that has frame_id.
        slice_id: If the image the slice is from is a 3D short axis image, slice_id has to be
            supplied so that ImagePositionPatient can be correctly calculated. It is assumed
            that the slice_id is the z-coordinate of the SAX image and this corresponds to the
            low-resolution axis.
        slice_info: Optional informative string about the slice such as ('SA', '2Ch', '4Ch' etc.
            Can also use something like SOPInstanceUID if you know which DICOM image the slice
            corresponds to.)
    """
    # if slice id is not 0, the size of the 3rd dimension on the NIFTI header must be greater
    # than slice_id.
    if slice_id != 0:
        assert header['dim'][3] > slice_id

    # Extract from the header, the rotation matrix that rotates the voxel coordinate axes to NIFTI
    # anatomical coordinate axes.
    vox = header['pixdim'][1:4].copy()
    qfac = header['pixdim'][0]
    vox[-1] *= qfac
    quat = header.get_qform_quaternion()
    rotation_matrix = nib.nifti1.quat2mat(quat)

    # Extract the position of pixel (0,0) with respect to NIFTI anatomical coordinates
    im_pos_patient = np.dot(header.get_qform(), np.array([0, 0, slice_id, 1]))

    # Invert the x and y sign to switch from NIFTI to DICOM anatomical coordinates
    im_pos_patient[0:2] = -im_pos_patient[0:2]

    # The rotation matrix we extracted is from NIFTI voxel to NIFTI anatomical. To get DICOM
    # direction cosines, we take the first 2 columns and invert the sign of the first 2 rows.
    gp_frame_info_file.write(f'{slice_info}\tframeID\t{frame_id}\ttimeFrameID\t{time_frame_id}'
                             f'\tImagePositionPatient'
                             f'\t{im_pos_patient[0]:.2f}'
                             f'\t{im_pos_patient[1]:.2f}'
                             f'\t{im_pos_patient[2]:.2f}'
                             f'\tImageOrientationPatient'
                             f'\t{-rotation_matrix[0, 0]:.2f}'
                             f'\t{-rotation_matrix[1, 0]:.2f}'
                             f'\t{rotation_matrix[2, 0]:.2f}'
                             f'\t{-rotation_matrix[0, 1]:.2f}'
                             f'\t{-rotation_matrix[1, 1]:.2f}'
                             f'\t{rotation_matrix[2, 1]:.2f}'
                             f'\tPixelSpacing'
                             f"\t{header['pixdim'][1]:.2f}"
                             f"\t{header['pixdim'][2]:.2f}\n")


def read_gp_points_and_visualize(gp_points_filename: str, time_frame_id: Optional[int] = 1):
    """

    Read a gp points file and visualize the points at a specific time frame. This is for DEBUG
    purposes. We have better visualization tools in the mesh fitting module.

    Args:
        gp_points_filename: The name of the gp points file (with full path).
        time_frame_id: The time frame id for which the points will be visualized. (e.g. ED has
            time frame id: 1)

    """

    with open(gp_points_filename, 'r') as file:
        # skip first line, read the rest into lines
        lines = file.readlines()[1:]

        # extract the data for the given time frame.
        data = np.asarray(
            [line.split() for line in lines if line.split()[-1] == str(time_frame_id)])

        ctypes = np.unique(data[:, 3])

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        for ctype in ctypes:
            ctype_pts = data[data[:, 3] == ctype][:, :3].astype(float)

            if ctype == "APEX_POINT":
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="red", marker='*', s=100)
            elif ctype == "MITRAL_VALVE":
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="red", marker='+', s=100)
            elif ctype == "TRICUSPID_VALVE":
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="blue", marker='+', s=100)
            elif ctype == "AORTA_VALVE":
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="green", marker='+', s=100)
            elif ctype in ["LAX_RV_FREEWALL", "SAX_RV_FREEWALL"]:
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="yellow", marker='o', s=50)
            elif ctype in ["LAX_RV_SEPTUM", "SAX_RV_SEPTUM"]:
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="green", marker='o', s=50)
            elif ctype in ["LAX_LV_EPICARDIAL", "SAX_LV_EPICARDIAL"]:
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="red", marker='o', s=50)
            elif ctype in ["LAX_LV_ENDOCARDIAL", "SAX_LV_ENDOCARDIAL"]:
                ax.scatter(ctype_pts[:, 1], ctype_pts[:, 0], ctype_pts[:, 2],
                           color="blue", marker='o', s=50)
            else:
                raise ValueError(f"Unexpected contour type {ctype} passed to "
                                 "read_gp_points_and_visualize().")

        plt.show()


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc, n_cc = measure.label(binary, return_num=True)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return largest_cc


def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc, n_cc = measure.label(binary, return_num=True)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def furthest_node(node, nodes):
    furthest_index = distance.cdist([node], nodes).argmax()
    return nodes[furthest_index]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--job', '-j', action='store', default='default', help='job identifier')

    parser.add_argument('--data-dir', '-d', action='store', help='path to data directory')
    parser.add_argument('--input-dir', '-I', action='store', default='nnUNet_segs', help='name of input directories')
    parser.add_argument('--output-dir', '-o', action='store', default='Contour_Outputs', help='name of output directories')

    parser.add_argument('--instance', '-i', type=int, action='store', default=2, help='instance to be processed')

    parser.add_argument('--all', '-a', action='store_true', help='process all subjects')
    parser.add_argument('--subject', '-s', action='store', help='subject id to be processed')
    parser.add_argument('--start', '-S', action='store', type=int, help='index of first subject id to be processed')
    parser.add_argument('--number', '-n', action='store', type=int, help='number of subjects to be processed from first subject id')
    parser.add_argument('--allowlist', '-l', action='store', help='path to subject allowlist')

    parser.add_argument('--timeframe', '-t', action='store', type=int, help='timeframe to process')
    parser.add_argument('--all-timeframes', '-T', action='store_true', help='process all timeframes')

    args, _ = parser.parse_known_args()

    data_dir = args.data_dir
    log_dir = data_dir

    # Setup logging to file.
    log_filename = os.path.join(log_dir, f'contour2gp_QS-{args.job}.log')
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter(fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You should have the subject ID as a folder name for each subject on the data_dir and no
    # other files or folders on data_dir
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

    # Label definitions in the segmentations
    label_defs_sax = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}
    label_defs_2ch = {'BG': 0, 'LV': 1, 'Myo': 2, 'LA': 3}
    label_defs_3ch = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3, 'LA': 4, 'Ao': 5}
    label_defs_4ch = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3, 'LA': 4, 'RA': 5}

    # instances you want to process
    instance_ids = [f"Instance_{args.instance}"]

    # # if you want to test specific subjects instead of all subjects in the data dir, uncomment this
    # subject_ids = ['UKBB_88878_1150922']

    for subject_id in subject_ids:
        logger.info(f" Processing subject {subject_id}...")

        for instance_id in instance_ids:
            logger.info(f" Processing instance {instance_id}")

            # Read the input segmentations
            seg_dir = os.path.join(data_dir, subject_id, instance_id, args.input_dir)
            sax_filename = os.path.join(seg_dir, 'SAX_nnUNetSeg.nii.gz')
            lax_2ch_filename = os.path.join(seg_dir, 'LAX_2Ch_nnUNetSeg.nii.gz')
            lax_3ch_filename = os.path.join(seg_dir, 'LAX_3Ch_nnUNetSeg.nii.gz')
            lax_4ch_filename = os.path.join(seg_dir, 'LAX_4Ch_nnUNetSeg.nii.gz')

            if os.path.isfile(sax_filename):
                sax_seg_nifti = nib.load(sax_filename)
            else:
                logger.error("SAX file not found. Skipping instance.")
                continue

            if os.path.isfile(lax_2ch_filename):
                lax_2ch_seg_nifti = nib.load(lax_2ch_filename)
            else:
                logger.error("LAX 2 chamber file not found. Skipping instance.")
                continue

            if os.path.isfile(lax_3ch_filename):
                lax_3ch_seg_nifti = nib.load(lax_3ch_filename)
            else:
                logger.error("LAX 3 chamber file not found. Skipping instance.")
                continue

            if os.path.isfile(lax_4ch_filename):
                lax_4ch_seg_nifti = nib.load(lax_4ch_filename)
            else:
                logger.error("LAX 4 chamber file not found. Skipping instance.")
                continue

            # ED frame is assumed to have time frame id = 0. ES time frame is picked automatically
            # based on LV volume.
            es_frame = auto_pick_es_frame([sax_seg_nifti, lax_2ch_seg_nifti, lax_3ch_seg_nifti,
                                           lax_4ch_seg_nifti])
            # if an error was detected in auto es picking, skip the instance
            if es_frame == -1:
                logger.error(f"{subject_id}: ES picking failed. Skipping instance.")
                continue

            if args.timeframe:
                time_frames_to_process = [args.timeframe - 1]
            elif not args.all_timeframes:
                time_frames_to_process = [0, es_frame]
            else:
                time_frames_to_process = list(range(50))

            # We will give a unique frame id to each 2d slice for the gp output. Note that this
            # is different from time_frame_id
            frame_id = 0

            # Change this line to where you need to save the GPFile & SliceInfoFile
            gp_file_dir = os.path.join(data_dir, subject_id, instance_id, args.output_dir)

            # read_gp_points_and_visualize(os.path.join(gp_file_dir, 'gp_points_file.txt'))
            # exit()

            # # Uncomment the following to delete everything in the output folder
            # for filename in os.listdir(gp_file_dir):
            #     file_path = os.path.join(gp_file_dir, filename)
            #     try:
            #         if os.path.isfile(file_path) or os.path.islink(file_path):
            #             os.unlink(file_path)
            #         elif os.path.isdir(file_path):
            #             shutil.rmtree(file_path)
            #     except Exception as e:
            #         print('Failed to delete %s. Reason: %s' % (file_path, e))

            # create the output directory if it doesn't exist
            if not os.path.exists(gp_file_dir):
                os.makedirs(gp_file_dir)

            # open output files
            gp_frame_info_file = open(os.path.join(gp_file_dir, 'gp_frame_info_file.txt'), 'w+')
            gp_points_file = open(os.path.join(gp_file_dir, 'gp_points_file.txt'), 'w+')

            # # You can uncomment this when debugging. Output is prettier but surface mesh code
            # # can't process it properly.
            # gp_points_file.write(
            #     f"{'x':^5}\t{'y':^5}\t{'z':^5}\t{'contourType':^20}\t{'frameID':^10}\t{'weight':^9}\t{'timeFrameID':^14}\n")

            # write the first row of gp points file
            gp_points_file.write(
                f"{'x'}\t{'y'}\t{'z'}\t{'contourType'}\t{'frameID'}\t{'weight'}\t{'timeFrameID'}\n")

            for tf_id in time_frames_to_process:
                # ------------
                # Contours 2CH
                # ------------
                lax_2ch_seg = np.squeeze(lax_2ch_seg_nifti.get_fdata()[:, :, :, tf_id])
                affine_2ch = lax_2ch_seg_nifti.affine

                # Extract contours and landmarks from the 2Ch segmentation image
                lax_2ch_contours_and_landmarks = (
                    get_contours_and_landmarks_2ch(lax_2ch_seg, label_defs_2ch, affine_2ch))

                if not lax_2ch_contours_and_landmarks:
                    logger.error(f"{subject_id}: LAX 2 Ch failed QC.")
                else:
                    lv_endo_contour, lv_epi_contour, mv_1, mv_2, apex = (
                        lax_2ch_contours_and_landmarks)

                    # # Visualization for DEBUG: 2Ch LV contours
                    # plt.imshow(lax_2ch_seg)
                    # plt.scatter(lv_endo_contour[:, 1], lv_endo_contour[:, 0], marker="x", color="red",
                    #             s=100)
                    # plt.scatter(lv_epi_contour[:, 1], lv_epi_contour[:, 0], marker="o", color="blue",
                    #             s=50)
                    # plt.show()
                    #
                    # # Visualization for DEBUG: 2Ch LV landmarks
                    # plt.imshow(lax_2ch_seg)
                    # plt.scatter(mv_anterior[1], mv_anterior[0], marker="+", color="red", s=100)
                    # plt.scatter(mv_posterior[1], mv_posterior[0], marker="o", color="red", s=50)
                    # plt.scatter(apex[1], apex[0], marker="+", color="blue", s=100)
                    # plt.show()

                    # write contours to gp points file
                    write_contour_to_gp_points_file(gp_points_file, lv_endo_contour, affine_2ch,
                                                    'LAX_LV_ENDOCARDIAL', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, lv_epi_contour, affine_2ch,
                                                    'LAX_LV_EPICARDIAL', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    # write landmarks to gp points file
                    write_contour_to_gp_points_file(gp_points_file, mv_1, affine_2ch,
                                                    'MITRAL_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, mv_2, affine_2ch,
                                                    'MITRAL_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, apex, affine_2ch,
                                                    'APEX_POINT', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)

                # write the frame's info to gp info file
                write_to_gp_frame_info_file(gp_frame_info_file, frame_id, time_frame_id=tf_id + 1,
                                            header=lax_2ch_seg_nifti.header, slice_info='2Ch')

                # increase frame_id by 1 after processing each slice
                frame_id += 1

                logger.info("2Ch processing done!")

                # ------------
                # Contours 3CH
                # ------------
                lax_3ch_seg = np.squeeze(lax_3ch_seg_nifti.get_fdata()[:, :, :, tf_id])
                affine_3ch = lax_3ch_seg_nifti.affine

                # Extract contours and landmarks from the 3Ch segmentation image
                lax_3ch_contours_and_landmarks = (
                    get_contours_and_landmarks_3ch(lax_3ch_seg, label_defs_3ch, affine_3ch))

                if not lax_3ch_contours_and_landmarks:
                    logger.error(f"{subject_id}: LAX 3 Ch failed QC.")
                else:
                    (lv_endo_contour, lv_epi_contour, aorta_1, aorta_2, mv_1, mv_2,
                     rv_septum_contour, rv_free_wall_contour) = lax_3ch_contours_and_landmarks

                    # # Visualization for DEBUG: 3Ch contours
                    # plt.imshow(lax_3ch_seg)
                    # plt.scatter(lv_endo_contour[:, 1], lv_endo_contour[:, 0], marker="x",
                    #             color="red", s=100)
                    # plt.scatter(lv_epi_contour[:, 1], lv_epi_contour[:, 0], marker="o", color="blue",
                    #             s=50)
                    # if rv_septum_contour is not None and rv_free_wall_contour is not None:
                    #     plt.scatter(rv_septum_contour[:, 1], rv_septum_contour[:, 0], marker="+",
                    #                 color="green", s=100)
                    #     plt.scatter(rv_free_wall_contour[:, 1], rv_free_wall_contour[:, 0], marker="*",
                    #                 color="yellow", s=50)
                    # plt.show()

                    # # Visualization for DEBUG: 3Ch landmarks
                    # plt.imshow(lax_3ch_seg)
                    # plt.scatter(aorta_posterior[1], aorta_posterior[0], marker="x", color="red", s=100)
                    # plt.scatter(aorta_anterior[1], aorta_anterior[0], marker="o", color="red", s=50)
                    # plt.scatter(mv_point[1], mv_point[0], marker="*", color="blue", s=100)
                    # plt.show()

                    # write contours to gp points file
                    write_contour_to_gp_points_file(gp_points_file, lv_endo_contour, affine_3ch,
                                                    'LAX_LV_ENDOCARDIAL', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, lv_epi_contour, affine_3ch,
                                                    'LAX_LV_EPICARDIAL', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)

                    if rv_septum_contour is not None:
                        write_contour_to_gp_points_file(gp_points_file, rv_septum_contour,
                                                        affine_3ch,
                                                        'LAX_RV_SEPTUM', slice_id=0,
                                                        frame_id=frame_id,
                                                        weight=1, time_frame_id=tf_id + 1)
                    if rv_free_wall_contour is not None:
                        write_contour_to_gp_points_file(gp_points_file, rv_free_wall_contour,
                                                        affine_3ch,
                                                        'LAX_RV_FREEWALL', slice_id=0,
                                                        frame_id=frame_id,
                                                        weight=1, time_frame_id=tf_id + 1)

                    # write landmarks to gp points file
                    write_contour_to_gp_points_file(gp_points_file, mv_1, affine_3ch,
                                                    'MITRAL_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, mv_2, affine_3ch,
                                                    'MITRAL_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, aorta_1, affine_3ch,
                                                    'AORTA_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, aorta_2, affine_3ch,
                                                    'AORTA_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)

                # write the frame's info to gp info file
                write_to_gp_frame_info_file(gp_frame_info_file, frame_id, time_frame_id=tf_id + 1,
                                            header=lax_3ch_seg_nifti.header, slice_info='3Ch')

                # increase frame_id by 1 after processing each slice
                frame_id += 1

                logger.info("3Ch processing done!")

                # ------------
                # Contours 4CH
                # ------------
                lax_4ch_seg = np.squeeze(lax_4ch_seg_nifti.get_fdata()[:, :, :, tf_id])
                affine_4ch = lax_4ch_seg_nifti.affine

                # Extract contours and landmarks from the 4Ch segmentation image
                lax_4ch_contours_and_landmarks = (
                    get_contours_and_landmarks_4ch(lax_4ch_seg, label_defs_4ch, affine_4ch))

                if not lax_4ch_contours_and_landmarks:
                    logger.error(f"{subject_id}: LAX 4 Ch failed QC.")
                else:
                    (lv_endo_contour, lv_epi_contour, rv_septum_contour, rv_free_wall_contour,
                     mv_left, mv_right, tv_left, tv_right) = lax_4ch_contours_and_landmarks

                    # # Visualization for DEBUG: 4Ch LV, RV contours
                    # plt.imshow(lax_4ch_seg)
                    # plt.scatter(lv_endo_contour[:, 1], lv_endo_contour[:, 0], marker="x",
                    #             color="red", s=100)
                    # plt.scatter(lv_epi_contour[:, 1], lv_epi_contour[:, 0], marker="o", color="blue",
                    #             s=50)
                    # plt.scatter(rv_free_wall_contour[:, 1], rv_free_wall_contour[:, 0], marker="+",
                    #             color="green", s=100)
                    # plt.scatter(rv_septum_contour[:, 1], rv_septum_contour[:, 0], marker="*",
                    #             color="yellow", s=100)
                    # plt.show()
                    #
                    # # Visualization for DEBUG: 4Ch LV, RV landmarks
                    # plt.imshow(lax_4ch_seg)
                    # plt.scatter(mv_left[1], mv_left[0], marker="x", color="red", s=100)
                    # plt.scatter(mv_right[1], mv_right[0], marker="o", color="blue", s=50)
                    # plt.scatter(tv_left[1], tv_left[0], marker="+", color="green", s=100)
                    # plt.scatter(tv_right[1], tv_right[0], marker="*", color="yellow", s=50)
                    # plt.show()

                    # write contours to gp points file
                    write_contour_to_gp_points_file(gp_points_file, lv_endo_contour, affine_4ch,
                                                    'LAX_LV_ENDOCARDIAL', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, lv_epi_contour, affine_4ch,
                                                    'LAX_LV_EPICARDIAL', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, rv_septum_contour, affine_4ch,
                                                    'LAX_RV_SEPTUM', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, rv_free_wall_contour,
                                                    affine_4ch,
                                                    'LAX_RV_FREEWALL', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)

                    # write landmarks to gp points file
                    write_contour_to_gp_points_file(gp_points_file, mv_left, affine_4ch,
                                                    'MITRAL_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, mv_right, affine_4ch,
                                                    'MITRAL_VALVE', slice_id=0, frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, tv_left, affine_4ch,
                                                    'TRICUSPID_VALVE', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)
                    write_contour_to_gp_points_file(gp_points_file, tv_right, affine_4ch,
                                                    'TRICUSPID_VALVE', slice_id=0,
                                                    frame_id=frame_id,
                                                    weight=1, time_frame_id=tf_id + 1)

                # write the frame's info to gp info file
                write_to_gp_frame_info_file(gp_frame_info_file, frame_id, time_frame_id=tf_id + 1,
                                            header=lax_4ch_seg_nifti.header, slice_info='4Ch')

                # increase frame_id by 1 after processing each slice
                frame_id += 1

                logger.info("4Ch processing done!")

                # ------------
                # Contours SAX
                # ------------
                sax_seg = sax_seg_nifti.get_fdata()[:, :, :, tf_id]
                affine_sax = sax_seg_nifti.affine

                sax_slices_qc_fail_count = 0

                for slice_id in range(sax_seg.shape[2]):
                    sax_seg_slice = sax_seg[:, :, slice_id]

                    # Extract LV endocardium, LV epicardium, RV septum and RV free wall contours
                    # from the sax_seg_slice
                    sax_contours = get_contours_sax_slice(sax_seg_slice, label_defs_sax, slice_id)

                    if not sax_contours:
                        sax_slices_qc_fail_count += 1
                    else:
                        lv_endo_contour, lv_epi_contour, rv_septum_contour, rv_free_wall_contour = \
                            sax_contours

                        # # Visualization for DEBUG: SAX contours
                        # plt.imshow(sax_seg_slice)
                        # plt.scatter(lv_endo_contour[:, 1], lv_endo_contour[:, 0], marker="x",
                        #             color="red", s=100)
                        # plt.scatter(lv_epi_contour[:, 1], lv_epi_contour[:, 0], marker="o",
                        #             color="blue", s=50)
                        # plt.scatter(rv_free_wall_contour[:, 1], rv_free_wall_contour[:, 0],
                        #             marker="+", color="green", s=100)
                        # plt.scatter(rv_septum_contour[:, 1], rv_septum_contour[:, 0], marker="*",
                        #             color="yellow", s=50)
                        # plt.show()

                        # write contours to gp file
                        write_contour_to_gp_points_file(gp_points_file, lv_endo_contour,
                                                        affine_sax,
                                                        'SAX_LV_ENDOCARDIAL', slice_id=slice_id,
                                                        frame_id=frame_id, weight=1,
                                                        time_frame_id=tf_id + 1)
                        write_contour_to_gp_points_file(gp_points_file, lv_epi_contour, affine_sax,
                                                        'SAX_LV_EPICARDIAL', slice_id=slice_id,
                                                        frame_id=frame_id, weight=1,
                                                        time_frame_id=tf_id + 1)
                        write_contour_to_gp_points_file(gp_points_file, rv_septum_contour,
                                                        affine_sax,
                                                        'SAX_RV_SEPTUM', slice_id=slice_id,
                                                        frame_id=frame_id, weight=1,
                                                        time_frame_id=tf_id + 1)
                        write_contour_to_gp_points_file(gp_points_file, rv_free_wall_contour,
                                                        affine_sax,
                                                        'SAX_RV_FREEWALL', slice_id=slice_id,
                                                        frame_id=frame_id, weight=1,
                                                        time_frame_id=tf_id + 1)

                        # write the frame's info to gp info file
                        write_to_gp_frame_info_file(gp_frame_info_file, frame_id,
                                                    time_frame_id=tf_id + 1,
                                                    header=sax_seg_nifti.header, slice_id=slice_id,
                                                    slice_info='SAX')

                        # increase frame_id by 1 after processing each slice
                        frame_id += 1

                logger.info(f"{subject_id}: SAX slice fail rate = {sax_slices_qc_fail_count} / "
                             f"{sax_seg.shape[2]}")

                logger.info("SAX processing done!")

            gp_frame_info_file.close()
            gp_points_file.close()

            logger.info(f" Completed processing instance {instance_id}")

            # # # visualization for DEBUG
            # read_gp_points_and_visualize(os.path.join(gp_file_dir, 'gp_points_file.txt'))

        logger.info(f" Completed processing subject {subject_id}. \n")


if __name__ == '__main__':
    main()
