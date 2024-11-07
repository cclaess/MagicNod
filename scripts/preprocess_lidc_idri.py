import os
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

    # Get the list of directories containing the image series
    series_dirs = glob(os.path.join(args.data_dir, "LIDC-IDRI", "LIDC-IDRI-*", "*", "*"))

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
        # Only include nodules with at least 3 radiologist annotations
        # Then combine the masks of the nodules into a single mask for the series
        nodule_masks = {}
        for mask_file in mask_files:
            nodule_id = mask_file.split("nodule=")[1].split("-rad=")[0]
            if nodule_id not in nodule_masks:
                nodule_masks[nodule_id] = []
            nodule_masks[nodule_id].append(mask_file)
        
        # Create the output directory for the series
        series_output_dir = output_dir / Path(series_dir).relative_to(Path(args.data_dir))
        series_output_dir.mkdir(parents=True, exist_ok=True)

        # Combine the masks for all nodules
        series_mask = None
        for nodule_masks_list in nodule_masks.values():
            if len(nodule_masks_list) < 3:
                continue
            nodule_masks_list = [sitk.ReadImage(mask_file) for mask_file in nodule_masks_list]
            # Combine the masks of the same nodule using majority voting
            nodule_mask = sitk.BinaryThreshold(sitk.LabelVoting(nodule_masks_list), lowerThreshold=1)
            if series_mask is None:
                series_mask = nodule_mask
            else:
                series_mask = sitk.Or(series_mask, nodule_mask)

        # Load the image series
        series_image = sitk.ReadImage(os.path.join(series_dir, "scan.nii.gz"))

        # Save the image series and mask
        sitk.WriteImage(series_image, os.path.join(series_output_dir, "image.nii.gz"))
        sitk.WriteImage(series_mask, os.path.join(series_output_dir, "mask.nii.gz"))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
