# utility script for nnUNet V2 to train multiple databases. 2D configuration only.

import argparse
import subprocess
import os
import time
import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", help="IDs of the datasets to train.", nargs='+')
    parser.add_argument("--folds", help="folds to train. If this argument is not passed, "
                                        "5 folds will be trained by default. ", nargs='*')
    parser.add_argument("--checkcudadevices", help="If this argument is passed, the script will "
                                                   "just print the available cuda devices and "
                                                   "exit. You can use this before training to "
                                                   "decide which device you want to use",
                        action="store_true")

    args = parser.parse_args()

    nnunet_preproc_dir = os.path.join(os.sep, "workspace", "nnUNet_workspace",
                                      "nnUNet_preprocessed")
    nnunet_results_dir = os.path.join(os.sep, "workspace", "nnUNet_workspace", "nnUNet_results")

    # if checkcudadevices is passed, just print the available cuda devices and exit
    if args.checkcudadevices:
        for i in range(torch.cuda.device_count()):
            print(f"Cuda device {i}: {torch.cuda.get_device_name(i)}")
        exit()

    if args.datasets is None:
        print("No Dataset given for training. Use --datasets to tell the script which datasets "
              "to train.")
        exit()

    # folds to train
    folds = ["0", "1", "2", "3", "4"]
    if args.folds is not None:
        folds = args.folds

    for dataset in args.datasets:

        # Check if plan and process was previously run on this dataset by checking the existence
        # of the plan file with default name. If the file doesn't exist, run plan and preprocess.
        dataset_folder_name = None
        for f in os.listdir(nnunet_preproc_dir):
            if f.startswith("Dataset" + dataset):
                dataset_folder_name = f

        if not dataset_folder_name:
            subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", dataset, "-c", "2d",
                            "--verify_dataset_integrity"])
        else:
            default_plan_file = os.path.join(nnunet_preproc_dir, dataset_folder_name,
                                             "nnUNetPlans.json")

            if os.path.exists(default_plan_file):
                print(f"Default plan file already exists for datasset {dataset}. Skipping "
                      f"plan and process.")
            else:
                subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", dataset, "-c", "2d",
                                "--clean", "--verify_dataset_integrity"])

        # Train
        # create timer to time how long training takes
        timer_file = open(os.path.join(nnunet_results_dir, "Dataset_" + dataset +
                                       "_timerFile.txt"), "a")
        for fold in folds:
            timer_file.write(f"\nStarted training fold {fold}. \n")
            tic = time.perf_counter()
            subprocess.run(["nnUNetv2_train", dataset, "2d", fold])
            toc = time.perf_counter()

            timer_file.write(f"Fold {fold} total train time: {toc - tic: 0.4f} seconds. ")

        timer_file.close()


if __name__ == "__main__":
    main()
