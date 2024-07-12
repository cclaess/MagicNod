import os
import argparse
from glob import glob
from pathlib import Path

import SimpleITK as sitk


def main(args):

    scans = sorted(glob(os.path.join(args.input_dir, "*data*", ".mhd")) + \
                   glob(os.path.join(args.input_dir, "masks", ".mhd")))
    
    print(scans)
    
    for scan in scans:

        sitk_scan = sitk.ReadImage(scan, sitk.sitkInt16)

        output_dir = Path(os.path.join(args.output_dir, scan.split(os.sep)[-2]))
        output_dir.mkdir(parents=True, exist_ok=True)

        ouput_filename = os.path.join(output_dir, scan.split(os.sep)[-1][:-4] + ".nii.gz")
        sitk.WriteImage(sitk_scan, os.path.join(output_dir, ouput_filename))
        print("Saved: ", os.path.join(output_dir, ouput_filename))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', default='/path/to/lndb/folder', type=str)
    parser.add_argument('--output_dir', '-o', default='/path/to/output/folder', type=str)
    args = parser.parse_args()
    main(args)
