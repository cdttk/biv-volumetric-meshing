
Biventricular model fitting framework
-----------------------------------------------
Author: Laura Dal Toso 

Date: 4 Feb 2022

-----------------------------------------------

This code performs patient-specific biventricular mesh customization. 

The process takes place in 2 steps:
1. correction of breath-hold misregistration between short-axis slices, and 
2. deformation of the subdivided template mesh to fit the manual contours while preserving 
the topology of the mesh.

Documentation: https://github.kcl.ac.uk/pages/YoungLab/BiV_Modelling/


Contents: 
-----------------------------------------------
- BiVFitting: contains the code that performs patient-specific biventricular mesh customization. 
- model: contains .txt files required by the fitting modules
- results: output folder
- test_data: contains one subfolder for each patient. Each subfolder contains the GPFile and SliceInfoFile relative to one patient.
- config_params: configuration of parameters for the fitting
- perform_fit: script that contains the routine to perform the biventricular fitting
- run_parallel: allows fitting using parallel CPUs. Each patient is assigned to one CPU.

Usage:
-----------------------------------------------

**Step 1**

Download the repository, and install the packages listed in requirements.txt.

**Step 2**

At first, the user needs to set the parameters in config_parameters.py. The variable 'measure_shift_EDonly' can be set to True if the user wants to measure the shift at ED frame only, and apply it to the other frames in the time series to correct for breath-hold misalignment. If 'measure_shift_EDonly' is set to False, the shift will be measured and applied at each frame. The variable 'sampling' can be used to sample the input guide points (sampling = 2 means every other point will be used). The variable workers defines the number of CPUs to be used for the fitting.

**Step 3**

At this point, you can modify the script perform_fit.py, which performs the model fitting. 

- Check that the relative paths defined after __name__ == '__main__' are correct.
- Check that filename and filenameInfo variables point to the corret guide points and slice information files that you want to process

You may need to change other variables in case you want to:  
- Fit the model only to a subset of all the available frames. To do so, change the 'frames_to_fit' parameter by assigning the frame numbers that should be fitted (i.e frames_to_fit = [0,1,2])
- Output a plot of the fitted models in html. Single frame plots can be generated by uncommenting the lines starting with "plot(go.Figure..". Time series plots that gather all time frames are controlled by the variables TimeSeries_step1 and TimeSeries_step2. To generate time series plots, uncomment all the lines containing these variables.


**Step 4**

After changing the script perform_fit.py according to your needs, there are two options to perform the model fitting. The first option is to fit the list of inout patients sequentially, by running perform_fit.py. 

To speed up the fitting, you may want to process different cases in parallel CPUs. To perform the fitting in parallel, you can use run_parallel.py. The relative paths at the bottom of this script need to be changed first, then the script can be lauched to generate the fitted models.

Credits
------------------------------------
Based on work by: Anna Mira, Liandong Lee, Richard Burns
