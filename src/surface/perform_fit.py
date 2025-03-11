# Input: 3D contours
# Output: Fitted model

import argparse
import configparser
import logging
import os
import numpy as np
import time
import pandas as pd
from pathlib import Path

from . import config_params as params
from .BiVFitting import BiventricularModel
from .BiVFitting import GPDataSet
from .BiVFitting import ContourType
from .BiVFitting import MultiThreadSmoothingED, SolveProblemCVXOPT
from .BiVFitting import plot_timeseries

if params.enable_visualizations:
    from plotly.offline import plot
    import plotly.graph_objs as go

logger = logging.getLogger('surface')

# This list of contours_to _plot was taken from Liandong Lee
contours_to_plot = [ContourType.LAX_RA, ContourType.LAX_RV_ENDOCARDIAL,
                    ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                    ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                    ContourType.SAX_LV_ENDOCARDIAL,
                    ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                    ContourType.APEX_POINT, ContourType.MITRAL_VALVE,
                    ContourType.TRICUSPID_VALVE, ContourType.AORTA_VALVE,
                    ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                    ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                    ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                    ContourType.AORTA_PHANTOM, ContourType.TRICUSPID_PHANTOM,
                    ContourType.MITRAL_PHANTOM
                    ]


def perform_fitting(input_dir, output_dir, gp_points_file='gp_points_file.txt',
                    gp_frame_info_file='gp_frame_info_file.txt', model_path='./model', **kwargs):
    try:
        #  performs all the BiVentricular fitting operations
        ''''''
        if 'iter_num' in kwargs:
            iter_num = kwargs.get('iter_num', None)
            pid = os.getpid()
            #logger.debug('child PID', pid)
            # assign a new process ID and a new CPU to the child process
            # iter_num corresponds to the id number of the CPU where the process will be run
            os.system("taskset -cp %d %d" %(iter_num, pid))

        if 'id_Frame' in kwargs:
            # acquire .csv file containing patient_id, ES frame number, ED frame number if present
            case_frame_dict = kwargs.get('id_Frame', None)

        # define the path to gp points file and gp frame info file
        filename = os.path.join(input_dir, gp_points_file)
        filename_info = os.path.join(input_dir, gp_frame_info_file)

        # extract the subject id from the input directory
        subject_id = input_dir.split(os.sep)[-3]

        if not os.path.exists(filename):
            logger.error(f'subject {subject_id}: gp points file does not exist')
            return
        if not os.path.exists(filename_info):
            logger.error(f'subject {subject_id}: gp points info file does not exist')
            return

        # create a log file to store fitting errors
        error_file = Path(os.path.join(output_dir, 'ErrorFile.txt'))
        error_file.touch(exist_ok=True)
        shift_file = Path(os.path.join(output_dir, 'Shiftfile.txt'))
        shift_file.touch(exist_ok=True)
        pos_file = Path(os.path.join(output_dir ,'Posfile.txt'))
        pos_file.touch(exist_ok=True)

        with open(error_file, 'w') as f:
            f.write(f'Log for subject: {subject_id} \n')

        #  read all the frames from the GPFile
        all_frames = pd.read_csv(filename, sep='\t')

        time_frames = np.unique(all_frames.values[:, 6]).astype(np.uint16)

        # if measure_shift_ed_only, we calculate and write the slice shifts based on the ED frame
        if params.measure_shift_ed_only:
            logger.debug('Shift measured only at ED frame')

            # time_frames[0] should be the ED frame.
            ed_dataset = GPDataSet(filename, filename_info, subject_id, sampling=params.sampling,
                                   time_frame_number=int(time_frames[0]))

            result_ed = ed_dataset.sinclaire_slice_shifting(frame_num=int(time_frames[0]))
            shift_ed = result_ed[0]
            pos_ed = result_ed[1]

            with open(shift_file, "w") as file:
                file.write(f'Shift measured only at ED (time frame {str(time_frames[0])}): \n')
                file.write(str(shift_ed))
                file.close()

            with open(pos_file, "w") as file:
                file.write(f'Pos measured only at ED (time frame {str(time_frames[0])}): \n')
                file.write(str(pos_ed))
                file.close()

        # Initialise time series lists
        time_series_step1 = []
        time_series_step2 = []

        logger.debug(f'Fitting of {subject_id} ----> started \n')

        timeframe_num = kwargs.get('timeframe_num', None)
        force_overwrite = kwargs.get('force_overwrite', False)

        # for all time frames
        for idx, time_frame_id in enumerate(time_frames):
            time_frame_id = int(time_frame_id)

            if timeframe_num is not None and timeframe_num != time_frame_id:
                logger.debug('subject {}, timeframe {}: skipping, timeframe not requested'.format(subject_id, time_frame_id))
                continue

            model_file = os.path.join(output_dir, f'{subject_id}_model_timeframe{time_frame_id:03}.txt')
            if os.path.exists(model_file) and not force_overwrite:
                logger.debug('subject {}, timeframe {}: skipping, model file exists'.format(subject_id, time_frame_id))
                continue

            logger.debug(f'Time frame id: {time_frame_id}')

            with open(error_file, 'a') as f:
                f.write(f"\nTime Frame # {time_frame_id}\n")

            data_set = GPDataSet(filename, filename_info, subject_id, sampling=params.sampling,
                                 time_frame_number=time_frame_id)
            biventricular_model = BiventricularModel(model_path, subject_id)

            if params.measure_shift_ed_only:
                # apply shift measured previously using ED frame
                data_set.apply_slice_shift(shift_ed, pos_ed)
            else:
                # measure and apply shift to current frame
                shifted_slice = data_set.sinclaire_slice_shifting(error_file, time_frame_id)
                shift_measure = shifted_slice[0]
                pos_measure = shifted_slice[1]

                if idx == 0:
                    with open(shift_file, "w") as file:
                        file.write(f'Time frame id: {time_frame_id}\n')
                        file.write(str(shift_measure))
                        file.close()
                    with open(pos_file, "w") as file:
                        file.write(f'Time frame id: {time_frame_id}\n')
                        file.write(str(pos_measure))
                        file.close()
                else:
                    with open(shift_file, "a") as file:
                        file.write(f'Time frame id: {time_frame_id}\n')
                        file.write(str(shift_measure))
                        file.close()
                    with open(pos_file, "w") as file:
                        file.write(f'Time frame id: {time_frame_id}\n')
                        file.write(str(pos_measure))
                        file.close()

            if not hasattr(data_set, 'tricuspid_centroid'):
                logger.error('subject {}, timeframe {}: missing attribute, tricuspid_centroid'.format(subject_id, time_frame_id))
                continue

            if not hasattr(data_set, 'apex'):
                logger.error('subject {}, timeframe {}: missing attribute, apex'.format(subject_id, time_frame_id))
                continue

            try:
                biventricular_model.update_pose_and_scale(data_set)
            except FloatingPointError:
                logger.error('subject {}, timeframe {}: failed to update pose and scale'.format(subject_id, time_frame_id))
                continue

            if params.enable_visualizations:
                contour_plots = data_set.PlotDataSet(contours_to_plot)

                data = contour_plots

                plot(go.Figure(data), filename=os.path.join(output_dir,
                                                            'pose_fitted_model_timeframe' + str(
                                                                time_frame_id) + '.html'),
                     auto_open=False)

            # Generates RV epicardial points if they have not been contoured
            # (can be commented if available) used in LL
            try:
                rv_epi_points, rv_epi_contour, rv_epi_slice = data_set.create_rv_epicardium(rv_thickness=3)
            except Exception:
                logger.error('subject {}, timeframe {}: failed to create RV epicardium'.format(subject_id, time_frame_id))
                continue

            # Generate phantom points for the mitral valve, the tricuspid valve, the pulmonary
            # artery and the aorta
            try:
                mitral_points = data_set.create_valve_phantom_points(30, ContourType.MITRAL_VALVE)
                tri_points = data_set.create_valve_phantom_points(30, ContourType.TRICUSPID_VALVE)
                pulmonary_points = data_set.create_valve_phantom_points(20, ContourType.PULMONARY_VALVE)
                aorta_points = data_set.create_valve_phantom_points(20, ContourType.AORTA_VALVE)
            except Exception:
                logger.error('subject {}, timeframe {}: failed to create phantom points'.format(subject_id, time_frame_id))
                continue

            # Example on how to set different weights for different points group (R.B.)
            data_set.weights[data_set.contour_type == ContourType.MITRAL_PHANTOM] = 2
            data_set.weights[data_set.contour_type == ContourType.AORTA_PHANTOM] = 2
            data_set.weights[data_set.contour_type == ContourType.PULMONARY_PHANTOM] = 2
            data_set.weights[data_set.contour_type == ContourType.TRICUSPID_PHANTOM] = 2

            data_set.weights[data_set.contour_type == ContourType.APEX_POINT] = 1
            data_set.weights[data_set.contour_type == ContourType.RV_INSERT] = 5

            data_set.weights[data_set.contour_type == ContourType.MITRAL_VALVE] = 2
            data_set.weights[data_set.contour_type == ContourType.AORTA_VALVE] = 2
            data_set.weights[data_set.contour_type == ContourType.PULMONARY_VALVE] = 2

            # Perform linear fit
            MultiThreadSmoothingED(biventricular_model, params.weight_gp, data_set, error_file)

            # Results after linear fit
            if params.enable_visualizations:
                model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)",
                                                         "rgb(127,0,0)", "all")
                data = model + contour_plots

                time_series_step1.append([data, time_frame_id])

                plot(go.Figure(data), filename=os.path.join(output_dir,
                                                            'linear_fitted_model_timeframe' + str(
                                                                time_frame_id) + '.html'),
                     auto_open=False)

            # Perform diffeomorphic fit (this step can take a while)
            SolveProblemCVXOPT(biventricular_model, data_set, params.weight_gp, params.low_smoothing_weight,
                               params.transmural_weight, error_file)

            # Results after diffeomorphic fit
            if params.enable_visualizations:
                model = biventricular_model.plot_surface("rgb(0,127,0)", "rgb(0,0,127)",
                                                         "rgb(127,0,0)", "all")

                data = model + contour_plots

                time_series_step2.append([data, time_frame_id])

                plot(go.Figure(data), filename=os.path.join(output_dir,
                                                            'diffeo_fitted_model_time_frame' + str(
                                                                time_frame_id) + '.html'),
                     auto_open=False)

            model_data = {'x': biventricular_model.control_mesh[:, 0],
                          'y': biventricular_model.control_mesh[:, 1],
                          'z': biventricular_model.control_mesh[:, 2],
                          'time_frame': [time_frame_id] *
                                        biventricular_model.control_mesh.shape[0]}
            model_dataframe = pd.DataFrame(data=model_data)

            with open(model_file, "w") as file:
                file.write(model_dataframe.to_string(header=True, index=False))

        if params.enable_visualizations:
            # Comment the following lines if you don't want html time series plots
            # if you want to plot time series in html files uncomment the next lines
            plot_timeseries(time_series_step1, output_dir, 'TimeSeries_linear_fit.html')

            # Comment if you did not run diffeomorphic fit
            plot_timeseries(time_series_step2, output_dir, 'TimeSeries_diffeo_fit.html')

    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--profile', '-p', action='store', default='default', help='config profile to be used')
    parser.add_argument('--job', '-j', action='store', default='default', help='job identifier')
    parser.add_argument('--force', '-f', action='store_true', default=False, help='force overwrite')

    parser.add_argument('--data-dir', '-d', action='store', help='path to data directory')
    parser.add_argument('--input-dir', '-I', action='store', default='Contour_Outputs', help='name of input directories')
    parser.add_argument('--output-dir', '-o', action='store', default='Mesh_Outputs', help='name of output directories')

    parser.add_argument('--instance', '-i', type=int, action='store', default=2, help='instance to be processed')

    parser.add_argument('--all', '-a', action='store_true', help='process all subjects')
    parser.add_argument('--subject', '-s', action='store', help='subject id to be processed')
    parser.add_argument('--start', '-S', action='store', type=int, help='index of first subject id to be processed')
    parser.add_argument('--number', '-n', action='store', type=int, help='number of subjects to be processed from first subject id')
    parser.add_argument('--allowlist', '-l', action='store', help='path to subject allowlist')

    parser.add_argument('--timeframe', '-t', type=int, action='store', help='timeframe to be processed')

    args, _ = parser.parse_known_args()

    cfg = configparser.ConfigParser()
    cfg.read('config.ini')

    data_dir = args.data_dir if args.data_dir else cfg[args.profile]['DataDir']

    start_time = time.time()

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

    log_filename = os.path.join(data_dir, f'surface-{args.job}.log')
    formatter = logging.Formatter(fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    np.seterr(all='raise')

    for subject_id in subject_ids:
        logger.debug(f"Processing subject {subject_id}...")

        i_dir = os.path.join(data_dir, subject_id, f'Instance_{args.instance}')

        if not os.path.exists(i_dir):
            logger.debug(f'Instance_{args.instance} directory does not exist for {subject_id}')
            continue

        input_dir = os.path.join(i_dir, args.input_dir)
        output_dir = os.path.join(i_dir, args.output_dir)

        # create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        perform_fitting(input_dir, output_dir, timeframe_num=args.timeframe, force_overwrite=args.force)

        logger.debug(f'Total run time: {time.time() - start_time}')


if __name__ == '__main__':
    main()
