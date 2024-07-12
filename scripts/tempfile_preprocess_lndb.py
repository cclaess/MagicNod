import os
import argparse
from glob import glob
from pathlib import Path

import SimpleITK as sitk


def main(args):

    scans = sorted(glob(os.path.join(args.input_dir, "*data*", "*.mhd")))
    masks = sorted(glob(os.path.join(args.input_dir, "masks", "*.mhd")))
    for scan in scans:

        subset = scan.split(os.sep)[-2]
        lndb_id = scan.split(os.sep)[-1].split("_")[0]

        sitk_scan = sitk.ReadImage(scan, sitk.sitkInt16)
        if subset.startswith("data"):
            masks_filtered = [mask for mask in masks if lndb_id in mask]
            sitk_masks = [sitk.ReadImage(mask, sitk.sitkUInt8) for mask in masks_filtered]
            sitk_masks = [sitk.Not(mask == 0) for mask in sitk_masks]

            # Combine the mask into one binary mask using majority voting
            sitk_mask = sitk.LabelVoting(sitk_masks)
            sitk_mask = sitk.Not(sitk_mask == 0)
            sitk_mask.CopyInformation(sitk_scan)
            sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)

            output_dir = Path(os.path.join(args.output_dir, "masks"))
            output_dir.mkdir(parents=True, exist_ok=True)

            ouput_filename = scan.split(os.sep)[-1][:-4] + ".nii.gz"
            sitk.WriteImage(sitk_mask, os.path.join(output_dir, ouput_filename))

        
        output_dir = Path(os.path.join(args.output_dir, subset))
        output_dir.mkdir(parents=True, exist_ok=True)

        ouput_filename = scan.split(os.sep)[-1][:-4] + ".nii.gz"
        sitk.WriteImage(sitk_scan, os.path.join(output_dir, ouput_filename))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', default='/path/to/lndb/folder', type=str)
    parser.add_argument('--output_dir', '-o', default='/path/to/output/folder', type=str)
    args = parser.parse_args()
    main(args)
