# BiV Volumetric Meshing

## Overview

This software pipeline generates finite
element biventricular heart models from cardiovascular magnetic resonance (CMR) images in the
[UK Biobank](https://www.ukbiobank.ac.uk).
The pipeline has four components.

1. segmentation: NIfTI images > Segmentations
2. contour: Segmentations > Contours
3. surface: Contours > Surface meshes
4. volumetric: Surface meshes > Volumetric meshes (including universal ventricular coordinates (UVC) and myocardial fiber structure)

The pipeline may not work as expected on data from other sources.

## Prerequisites

+ Git (all)
+ Python >=3.9 (all)
+ openCARP >=10.0 (volumetric mesh, and UVC and fiber)

#### Notes
1. nnUNet (segmentation) may not work with Python >=3.12

### Installing openCARP

[openCARP](https://opencarp.org/download/installation) can be installed as an
unprivileged user in the following way. The resulting binary directory will be
`/opt/openCARP/usr/bin` or equivalent.

```
$ wget https://git.opencarp.org/api/v4/projects/16/packages/generic/opencarp-appimage/v11.0/openCARP-v11.0-x86_64_AppImage.tar.gz
$ tar xf openCARP-v11.0-x86_64_AppImage.tar.gz
$ ./openCARP-v11.0-x86_64_AppImage/openCARP-v11.0-x86_64.AppImage --appimage-extract
$ mv squashfs-root /opt/openCARP
```

## Setup

### Source code

Download the source code from GitHub.

```
$ git clone https://www.github.com/cdttk/biv-volumetric-meshing
```

### Installing Python dependencies

Install the necessary Python dependencies in a virtual environment.

```
$ cd cme-dt-pipeline
$ python3 -m venv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
```

### Data directory

All components of the pipeline work upon a data directory structured in a
particular way.

1. Each subject is given a subdirectory in the data directory named after its
   identifier.
2. Data instance (visit) number *n* for a subject is given a subdirectory in the
   subject directory, named `Instance_n`, containing the relevant initial NIfTI
   files, and any subsequent output subdirectories.

Each subject requires four input files in NIfTI format; one short axis (SAX)
and three long axis (LAX) &mdash; 2, 3 and 4 chamber (Ch) views.  These files
must be named as follows.

```
SAX.nii.gz  LAX_2Ch.nii.gz  LAX_3Ch.nii.gz  LAX_4Ch.nii.gz
```

#### Example of initial data directory

```
$ ls -R data

data/:
subject001  subject002

data/subject001:
Instance_2

data/subject001/Instance_2:
LAX_2Ch.nii.gz  LAX_3Ch.nii.gz  LAX_4Ch.nii.gz  SAX.nii.gz

data/subject002:
Instance_2  Instance_3

data/subject002/Instance_2:
LAX_2Ch.nii.gz  LAX_3Ch.nii.gz  LAX_4Ch.nii.gz  SAX.nii.gz

data/subject002/Instance_3:
LAX_2Ch.nii.gz  LAX_3Ch.nii.gz  LAX_4Ch.nii.gz  SAX.nii.gz
```

#### Notes
1. Log files are also written to the data directory.
2. No subdirectories other than those for subjects should be present in the data
   directory.
3. Sample data, which would need to be named and structured as above, may be
   obtained from [here](https://github.com/baiwenjia/ukbb_cardiac/blob/master/demo_pipeline.py).


### nnUNet workspace (segmentation)

The segmentation component requires that an nnUNet workspace be set up.

Download the four required model files from [GitHub](https://www.github.com/cdttk/biv-volumetric-meshing/releases/tag/v1-public).

```
Dataset100_UKBB_Petersen_SAX.20240108.tar
Dataset101_UKBB_LAX_2Ch.20240108.tar
Dataset102_UKBB_LAX_3Ch.20240108.tar
Dataset103_UKBB_LAX_4Ch.20240108.tar
```

Create a directory for the workspace wherever convenient and extract the model files to it.

```
$ mkdir nnunet_workspace
$ for i in Dataset*.tar; do tar xf $i -C nnunet_workspace; done
```

## Usage

```
$ cd cme-dt-pipeline
$ source ./venv/bin/activate
$ cd src
$ python run.py
```

Select the component or components to be run.

```
--segmentation     run segmentation
--contour          run contour
--surface          run surface
--volumetric       run volumetric
--uvc-fiber        run UVC and fiber

--all-components   run complete pipeline
```

### Useful common arguments

The following arguments are common across components. Selecting the data
directory is required, other arguments are optional. Instance 2 is processed by
default. Input and output directories have default values unless otherwise
specified.  All timeframes available are processed at each stage unless
otherwise specified, except during contour extraction which automatically selects
ED and ES timeframes.

```
--data-dir DATA_DIR, -d DATA_DIR         path to data directory
--instance INSTANCE, -i INSTANCE         instance to be processed
--timeframe TIMEFRAME, -t TIMEFRAME      timeframe to be processed
--input-dir INPUT_DIR, -I INPUT_DIR      name of input directories
--output-dir OUTPUT_DIR, -o OUTPUT_DIR   name of output directories
--job JOB, -j JOB                        job identifier
```

Subjects are ordered alphabetically and indexed from zero.

```
--all, -a                       process all subjects
--subject SUBJECT, -s SUBJECT   subject id to be processed
--start START, -S START         index of first subject id to be processed
--number NUMBER, -n NUMBER      number of subjects to processed from first subject id
```

#### Notes

The following limitations currently apply.

1. Timeframe selection is not possible for segmentation.
2. Output directories cannot be specified for volumetric.

### Component specific arguments

#### Segmentation

The path to the workspace directory must be specified. The `--gpu` flag may be
used to run inference on a CUDA capable GPU.

```
--workspace-dir WORKSPACE_DIR   path to the nnUNet workspace directory
--gpu                           run on gpu
```

#### Volumetric and UVC and fiber

The path to the openCARP binaries directory must be specified.

```
--carp-bin-dir CARP_BIN_DIR   the path to the openCARP binaries directory
```

### Examples

1. Run segmentation for 5 subjects starting with the 1st subject

    ```
    $ python run.py --data-dir /path/to/data --workspace-dir /path/to/workspace --segmentation --start 0 --number 5
    ```

2. Run contour extraction for ED and ES timeframes for all subjects and write
   output to subdirectories named `contours.test1`

    ```
    $ python run.py --data-dir /path/to/data --contour --all --output-dir contours.test1
    ```

3. Run surface meshing for subject `subject001`, reading input from subdirectories
   named `contours.test1`

    ```
    $ python run.py --data-dir /path/to/data --surface --subject subject001 --input-dir contours.test1
    ```

4. Run contour extraction and surface meshing for the first one hundred subjects and label the log `surface100`

    ```
    $ python run.py --data-dir /path/to/data --contour --surface --start 0 --number 100 --job surface100
    ```

5. Run volumetric meshing for timeframe 1 only for 5 subjects starting with the 10th subject

    ```
    $ python run.py --data-dir /path/to/data --carp-bin-dir /opt/openCARP/usr/bin --volumetric --start 9 --number 5 --timeframe 1
    ```

6. Run UVC and fiber generation for all timeframes for all subjects

    ```
    $ python run.py --data-dir /path/to/data --carp-bin-dir /opt/openCARP/usr/bin --uvc-fiber --all
    ```

7. Run the complete pipeline for subject `subject002`

    ```
    $ python run.py --data-dir /path/to/data --workspace-dir /path/to/workspace --carp-bin-dir /opt/openCARP/usr/bin --all-components --subject subject002
    ```

## Docker

The code may alternatively be built and run using [Docker](https://www.docker.com).


```
$ cd cme-dt-pipeline
$ docker build . -t cmedt/cdttk
$ docker run --rm -v /path/to/data/directory:/data -v /path/to/nnunet/workspace:/nnunet cmedt/cdttk --all-components --all
```

To run segmentation using a GPU the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
must be installed.

```
$ docker run --rm -v /path/to/data/directory:/data -v /path/to/nnunet/workspace:/nnunet --gpus=all cmedt/cdttk --all-components --all --gpu
```

## Credits

If you find this software useful, please consider giving appropriate credit by
citing one of the below papers.

1. Devran Ugurlu, Shuang Qian, Elliot Fairweather et al; Cardiac Digital Twins
   at Scale from MRI: Open Tools and Representative Modles from ~55000 UK
   Biobank Participants; to appear

2. Shuang Qian, Devran Ugurlu, Elliot Fairweather et al; Developing Cardiac
   Digital Twins at Scale: Insights from Personalised Myocardial Conduction
   Velocity; medRxiv 2023; [DOI](https://doi.org/10.1101/2023.12.05.23299435)
