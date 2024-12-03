import os
import random
import argparse
from glob import glob
from pathlib import Path

import SimpleITK as sitk


def get_args_parser():
    """
    Get the argument parser for the script.

    Returns:
    - argparse.ArgumentParser: Argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI dataset")
    
    # Experiment and data configuration
    parser.add_argument("--data-dir", required=True, type=str, help="Path to the data directory")
    parser.add_argument("--output-dir", required=True, type=str, help="Path to the output directory")
    
    return parser


def main(args):

    # Get a list of all subjects in the LIDC-IDRI dataset
    subjects = sorted([subject for subject in os.listdir(
        args.data_dir) if subject.startswith("LIDC-IDRI-")])
    random.seed(42)
    random.shuffle(subjects)
    
    # Divide the subjects into training and validation sets
    num_subjects = len(subjects)
    num_train = int(0.9 * num_subjects)
    train_subjects = subjects[:num_train]

    # Get the list of directories containing the image series
    series_dirs = glob(os.path.join(args.data_dir, "LIDC-IDRI-*", "*", "*"))

    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop over the series directories
    for series_dir in series_dirs:
        
        # Check if the series dir contains an image series
        if not os.path.exists(os.path.join(series_dir, "scan.nii.gz")):
            continue

        # Load the segmentation files
        mask_files = glob(os.path.join(series_dir, "nodule=*-rad=*.nii.gz"))

        # Combine the masks of different radiologists into a single mask per nodule using majority voting
        # Only include nodules with at least 2 radiologist annotations
        # Then combine the masks of the nodules into a single mask for the series
        nodule_masks = {}
        for mask_file in mask_files:
            nodule_id = mask_file.split("nodule=")[1].split("-rad=")[0]
            if nodule_id not in nodule_masks:
                nodule_masks[nodule_id] = []
            nodule_masks[nodule_id].append(mask_file)
        
        # Combine the masks for all nodules
        series_mask = None
        for nodule_masks_list in nodule_masks.values():
            if len(nodule_masks_list) < 2:
                continue
            nodule_masks_list = [sitk.ReadImage(mask_file) for mask_file in nodule_masks_list]
            # Combine the masks of the same nodule using majority voting
            nodule_mask = sitk.BinaryThreshold(sitk.LabelVoting(nodule_masks_list), lowerThreshold=1)
            if series_mask is None:
                series_mask = nodule_mask
            else:
                series_mask = sitk.Or(series_mask, nodule_mask)
        
        if series_mask is None:
            continue

        # Get the subject ID from the series directory
        subject_id = "LIDC-IDRI-" + series_dir.split("LIDC-IDRI-")[1].split("/")[0]
        split = "train" if subject_id in train_subjects else "valid"

        # Create the output directory for the series
        series_output_dir = output_dir / split / Path(series_dir).relative_to(Path(args.data_dir))
        series_output_dir.mkdir(parents=True, exist_ok=True)

        # Load the image series
        series_image = sitk.ReadImage(os.path.join(series_dir, "scan.nii.gz"))

        # Save the image series and mask
        sitk.WriteImage(series_image, os.path.join(series_output_dir, "image.nii.gz"))
        sitk.WriteImage(series_mask, os.path.join(series_output_dir, "mask.nii.gz"))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
