import pylidc as pl


import os
import re
import argparse
from pathlib import Path

import SimpleITK as sitk


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', default='/path/to/output/folder', type=str)

    return parser


def main(args):

    scans = pl.query(pl.Scan)
    for scan in scans:

        nods = scan.cluster_annotations()
        print(nods)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
