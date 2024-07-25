import os
import random
import shutil
import argparse
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', default='/path/to/lidc/folder', type=str)
    parser.add_argument('--output_dir', '-o', default='/path/to/output/folder', type=str)

    return parser


def create_rgb_image_with_mask(slice_img, slice_msk_bbox):

    # Convert slice_msk_bbox to an RGB image
    red_channel = sitk.Image(slice_msk_bbox.GetSize(), sitk.sitkUInt8)
    red_channel = (red_channel + 1) * 255
    green_channel = sitk.Image(slice_msk_bbox.GetSize(), sitk.sitkUInt8)
    blue_channel = sitk.Image(slice_msk_bbox.GetSize(), sitk.sitkUInt8)
    slice_msk_rgb = sitk.Compose(red_channel, green_channel, blue_channel)
    slice_msk_rgb.CopyInformation(slice_img)
    slice_msk_rgb = sitk.Mask(slice_msk_rgb, slice_msk_bbox)

    # Prepare the slice image for masking
    slice_img = sitk.Clamp(slice_img, lowerBound=-1000, upperBound=1000)
    slice_img = sitk.RescaleIntensity(slice_img, 0, 255)
    slice_img = sitk.Cast(slice_img, sitk.sitkUInt8)

    # Convert the image to RGB
    slice_img_rgb = sitk.Compose(slice_img, slice_img, slice_img)
    slice_msk_rgb.CopyInformation(slice_img_rgb)

    # Apply the inverse of slice mask on the image and add the rgb mask
    slice_img_rgb_masked = sitk.Mask(slice_img_rgb, sitk.Cast(sitk.Not(slice_msk_bbox), sitk.sitkUInt8))
    slice_img_rgb_masked = slice_img_rgb_masked + slice_msk_rgb

    return slice_img, slice_img_rgb_masked


def zip_directory(dir_path, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=dir_path)
                zipf.write(file_path, arcname=arcname)


def main(args):

    nodules_df = pd.read_csv(os.path.join(args.input_dir, "data.csv"))
    scan_ids = nodules_df["ScanID"].unique()
    grouped_df = nodules_df.groupby(["ScanID"])

    for scan_id in scan_ids:
        current_df = grouped_df.get_group((scan_id,))

        # get the study_id and file_name from the `ScanPath` of the first row
        patient_id = current_df["Patient"].iloc[0]
        moment = current_df["ScanPath"].iloc[0].split('/')[-3]

        # create the output directories
        output_base_path = Path(os.path.join(args.output_dir, patient_id, moment))
        out_dirs = {
            "NoduleMasked": output_base_path / "NoduleMasked",
            "NoduleScan": output_base_path / "NoduleScan",
            "NoduleMask": output_base_path / "NoduleMask",
            "RandomMasked": output_base_path / "RandomMasked",
            "RandomScan": output_base_path / "RandomScan",
            "RandomMask": output_base_path / "RandomMask"
        }

        # Flag to skip processing if all directories are already zipped and processed
        skip_processing = True
        for dir_name, dir_path in out_dirs.items():
            zip_file_path = output_base_path / f'{dir_name}.zip'
            if dir_path.exists() and any(dir_path.iterdir()):
                print(f"Directory {dir_path} already exists and is not empty, zipping...")
                zip_directory(dir_path, zip_file_path)
                shutil.rmtree(dir_path)  # Remove the directory after zipping
            elif zip_file_path.exists():
                print(f"Zip file {zip_file_path} already exists, skipping...")
                continue
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                skip_processing = False

        if skip_processing:
            print(f"Patient {patient_id} has already been processed")
            continue

        # load image and mask
        sitk_img = sitk.ReadImage(os.path.join(args.input_dir, current_df["ScanPath"].iloc[0]), sitk.sitkInt16)
        sitk_msk = sitk.ReadImage(os.path.join(args.input_dir, current_df["AnnotationPath"].iloc[0]), sitk.sitkUInt8)

        # get the size of the XY plane of the scan
        size = sitk_img.GetSize()[:2]

        # create a list for all remaining slices without nodules
        remaining_slices = range(sitk_img.GetSize()[2])

        # loop over all nodules
        nodules = current_df["NoduleID"].unique()
        for i in nodules:
            nodule = current_df[current_df["NoduleID"] == i]

            # check if the nodule is unique
            if len(nodule) > 1:
                print(f"Patient {patient_id} has multiple nodules with the same ID {i}, we only consider the first one")
                nodule = nodule.iloc[0]

            # get the associated slices of the nodule
            z_start = nodule["ZNoduleStart"].item()
            z_end = nodule["ZNoduleSize"].item() + z_start

            # if slices still in remaining_slices, remove them
            remaining_slices = [z for z in remaining_slices if z not in range(z_start, z_end)]

            # get the bounding box of the nodule and add a margin of 5 pixels
            bbox = (nodule["XNoduleStart"].item(), nodule["YNoduleStart"].item(), 
                    nodule["XNoduleSize"].item(), nodule["YNoduleSize"].item())
            bbox = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 10, bbox[3] + 10)

            # create a mask with the bounding box of the nodule
            slice_msk_bbox = sitk.Image(size, sitk.sitkUInt8)
            slice_msk_bbox[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = 1

            # loop over the slices of the nodule and save the masked and unmasked images
            for z in range(z_start, z_end):

                # get the axial slice of the image and mask it
                slice_img = sitk_img[:, :, z]
                slice_msk = sitk_msk[:, :, z]
                slice_msk_bbox.CopyInformation(slice_img)
                slice_img, slice_img_rgb_masked = create_rgb_image_with_mask(slice_img, slice_msk_bbox)	

                # Rescale the mask for saving
                slice_msk = sitk.Cast(slice_msk * 255, sitk.sitkUInt8)

                sitk.WriteImage(slice_img_rgb_masked, out_dirs["NoduleMasked"] / f'{scan_id}-nodule={i:03}-slice={z:03}.png')
                sitk.WriteImage(slice_img,  out_dirs["NoduleScan"] / f'{scan_id}-nodule={i:03}-slice={z:03}.png')
                sitk.WriteImage(slice_msk, out_dirs["NoduleMask"] / f'{scan_id}-nodule={i:03}-slice={z:03}.png')
        
        # loop over the remaining slices and save them with random masks
        for z in remaining_slices:

            # get the axial slice of the image
            slice_img = sitk_img[:, :, z]
            slice_msk = sitk_msk[:, :, z]

            # iterate for 10 times on every slice
            for i in range(10):
                
                # create a random bbox between 10 and 100 pixels
                bbox_size_x, bbox_size_y = random.randint(10, 100), random.randint(10, 100)
                bbox_start_x, bbox_start_y = random.randint(0, size[0] - bbox_size_x), random.randint(0, size[1] - bbox_size_y)
                bbox = (bbox_start_x, bbox_start_y, bbox_size_x, bbox_size_y)

                # create a mask with the bounding box of the nodule
                slice_msk_bbox = sitk.Image(size, sitk.sitkUInt8)
                slice_msk_bbox[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = 1
                slice_msk_bbox.CopyInformation(slice_img)

                # get the axial slice of the image and mask it
                slice_img, slice_img_rgb_masked = create_rgb_image_with_mask(slice_img, slice_msk_bbox)

                # Rescale the mask for saving
                slice_msk = sitk.Cast(slice_msk * 255, sitk.sitkUInt8)

                sitk.WriteImage(slice_img_rgb_masked, out_dirs["RandomMasked"] / f'{scan_id}-random={i:03}-slice={z:03}.png')
                sitk.WriteImage(slice_img, out_dirs["RandomScan"] / f'{scan_id}-random={i:03}-slice={z:03}.png')
                sitk.WriteImage(slice_msk, out_dirs["RandomMask"] / f'{scan_id}-random={i:03}-slice={z:03}.png')
        
        # Create zip files for each output directory
        for dir_name, dir_path in out_dirs.items():
            zip_file_path = output_base_path / f'{dir_name}.zip'
            zip_directory(dir_path, zip_file_path)
            shutil.rmtree(dir_path)  # Remove the directory after zipping
                

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
