import pandas as pd
from dask import dataframe as dd
import shutil
import os
import numpy as np
import nibabel as nib
from collections import Counter


def main():
    qc_file = os.path.join(os.sep, "netapp", "cme_digital_twins", "UKBB_88878",
                           "contour2gp_QS_all.log")

    passed_subject_ids = []
    num_total_subjects = 0
    num_subjects_w_all_files = 0
    fails = {"No 2Ch file": [],
             "No 3Ch file": [],
             "No 4Ch file": [],
             "No SAX file": [],
             "SAX has less than 5 slices": [],
             "SAX QC fail": [],
             "LAX 2Ch QC fail": [],
             "LAX 3Ch QC fail": [],
             "LAX 4Ch QC fail": []}

    with open(qc_file) as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith("INFO: Processing subject"):
                qc_pass = True
                all_files_exist = True
                current_subject_id = line[25:43]

            if line.startswith("INFO: Completed processing subject"):
                num_total_subjects += 1

                if qc_pass:
                    passed_subject_ids.append(current_subject_id)

                if all_files_exist:
                    num_subjects_w_all_files += 1

            if "LAX 2 chamber file not found" in line:
                fails["No 2Ch file"].append(current_subject_id)
                qc_pass = False
                all_files_exist = False

            if "LAX 3 chamber file not found" in line:
                fails["No 3Ch file"].append(current_subject_id)
                qc_pass = False
                all_files_exist = False

            if "LAX 4 chamber file not found" in line:
                fails["No 4Ch file"].append(current_subject_id)
                qc_pass = False
                all_files_exist = False

            if "SAX file not found" in line:
                fails["No SAX file"].append(current_subject_id)
                qc_pass = False
                all_files_exist = False

            if "SAX has less than 5 slices" in line:
                fails["SAX has less than 5 slices"].append(current_subject_id)
                qc_pass = False

            if "SAX slice fail rate" in line:
                total = int(line[-3:])
                fail = int(line[46:49])
                success = total - fail

                if success < 5:
                    fails["SAX QC fail"].append(current_subject_id)
                    qc_pass = False

            if "LAX 2 Ch failed QC" in line:
                fails["LAX 2Ch QC fail"].append(current_subject_id)
                qc_pass = False

            # if "LAX 3 Ch failed QC" in line:
            #     fails["LAX 3Ch QC fail"].append(current_subject_id)
            #     qc_pass = False

            if "LAX 4 Ch failed QC" in line:
                fails["LAX 4Ch QC fail"].append(current_subject_id)
                qc_pass = False

            # if line.startswith("ERROR:"):
            #     qc_pass = False

    for cause, fail_subjects in fails.items():
        print(f"{cause} rate = {len(fail_subjects) / num_total_subjects * 100:.2f}%")

    print(f"All views exist for {num_subjects_w_all_files} out of {num_total_subjects} subjects.")

    print(
        f"Contour QC fail rate = {(num_subjects_w_all_files - len(passed_subject_ids)) / num_subjects_w_all_files * 100:.2f}%")

    print(
        f"Contour QC Passed = {len(passed_subject_ids)} out of {num_subjects_w_all_files} subjects")

    output_folder = os.path.join(os.sep, "netapp", "cme_digital_twins", "UKBB_88878", "qc_overview")

    with open(os.path.join(output_folder, "subjects_to_mesh.txt"), 'w') as fp:
        fp.write('\n'.join(passed_subject_ids))


if __name__ == "__main__":
    main()
