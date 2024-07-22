import os
import random
import argparse
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


def main(args):

    nodules_df = pd.read_csv(os.path.join(args.input_dir, "data.csv"))
    patients = nodules_df["Patient"].unique()

    grouped_df = nodules_df.groupby(["Patient"])
    for patient_id in patients:

        current_df = grouped_df.get_group(patient_id)

        # get the study_id and file_name from the `ScanPath` of the first row
        study_id = current_df["ScanPath"].iloc[0].split(os.sep)[-3]
        file_name = current_df["ScanPath"].iloc[0].split(os.sep)[-1][:-7]

        # check if the patient has not yet been processed
        if os.path.exists(os.path.join(args.output_dir, patient_id, study_id)) and \
           len(glob(os.path.join(args.output_dir, patient_id, study_id, "*", "*.png"))) > 0:
            print(f"Patient {patient_id} has already been processed")
            continue

        # create the output directories
        out_dir_nodule_masked = Path(os.path.join(args.output_dir, patient_id, study_id, "NoduleMasked"))
        out_dir_nodule_scan = Path(os.path.join(args.output_dir, patient_id, study_id, "NoduleScan"))
        out_dir_nodule_mask = Path(os.path.join(args.output_dir, patient_id, study_id, "NoduleMask"))
        out_dir_random_masked = Path(os.path.join(args.output_dir, patient_id, study_id, "RandomMasked"))
        out_dir_random_scan = Path(os.path.join(args.output_dir, patient_id, study_id, "RandomScan"))
        out_dir_random_mask = Path(os.path.join(args.output_dir, patient_id, study_id, "RandomMask"))

        out_dir_nodule_masked.mkdir(parents=True, exist_ok=True)
        out_dir_nodule_scan.mkdir(parents=True, exist_ok=True)
        out_dir_nodule_mask.mkdir(parents=True, exist_ok=True)
        out_dir_random_masked.mkdir(parents=True, exist_ok=True)
        out_dir_random_scan.mkdir(parents=True, exist_ok=True)
        out_dir_random_mask.mkdir(parents=True, exist_ok=True)

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

                sitk.WriteImage(slice_img_rgb_masked, os.path.join(out_dir_nodule_masked, f'{file_name}-nodule={i:03}-slice={z:03}.png'))
                sitk.WriteImage(slice_img, os.path.join(out_dir_nodule_scan, f'{file_name}-nodule={i:03}-slice={z:03}.png'))
                sitk.WriteImage(slice_msk, os.path.join(out_dir_nodule_mask, f'{file_name}-nodule={i:03}-slice={z:03}.png'))
        
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

                sitk.WriteImage(slice_img_rgb_masked, os.path.join(out_dir_random_masked, f'{file_name}-random={i:03}-slice={z:03}.png'))
                sitk.WriteImage(slice_img, os.path.join(out_dir_random_scan, f'{file_name}-random={i:03}-slice={z:03}.png'))
                sitk.WriteImage(slice_msk, os.path.join(out_dir_random_mask, f'{file_name}-random={i:03}-slice={z:03}.png'))
                

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
