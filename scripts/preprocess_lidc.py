import os
import argparse
from glob import glob
from pathlib import Path

import SimpleITK as sitk


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', default='/path/to/lidc/folder', type=str)
    parser.add_argument('--output_dir', '-o', default='/path/to/output/folder', type=str)

    return parser


def main(args):

    # initialize the label statistics filter
    label_statistic = sitk.LabelIntensityStatisticsImageFilter()

    # get all scans and masks
    scan_paths = sorted(glob(os.path.join(args.input_dir, "*", "*", "Scan", "*.nii.gz")))
    mask_paths = sorted(glob(os.path.join(args.input_dir, "*", "*", "Mask", "*.nii.gz")))

    for scan_path, mask_path in zip(scan_paths, mask_paths):

        # get the patient id, study id and file name
        patient_id = scan_path.split('/')[-4]
        study_id = scan_path.split('/')[-3]
        file_name = scan_path.split('/')[-1][:-7]

        print(f"Processing {patient_id}...")

        # load image and mask
        sitk_img = sitk.ReadImage(scan_path, sitk.sitkInt16)
        sitk_msk = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # get image spacing
        img_spacing = sitk_img.GetSpacing()
        msk_spacing = sitk_msk.GetSpacing()

        # check if mask and image have the same spacing
        if img_spacing != msk_spacing:
            sitk_msk.SetSpacing(img_spacing)
            print(f"Spacing of mask and image is not the same for {scan_path} and {mask_path}. Setting mask spacing to image spacing.")

        # set the label to 1
        sitk_msk = sitk.Not(sitk_msk == 0)
        sitk_msk = sitk.Cast(sitk_msk, sitk.sitkUInt8)

        # save all slices of image that contain nodules with and without black squares covering the nodules
        for z in range(sitk_img.GetSize()[2]):
            slice_img = sitk_img[:, :, z]
            slice_msk = sitk_msk[:, :, z]
            
            if sitk.GetArrayFromImage(slice_msk).sum() > 0:

                # Create a mask with different labels for each connected component
                slice_msk_labeled = sitk.ConnectedComponent(slice_msk)
                slice_msk_labeled.CopyInformation(slice_msk)

                # Create a new mask with bounding boxes covering the regions
                label_statistic.Execute(slice_msk_labeled, slice_msk_labeled)
                slice_msk_bbox = sitk.Image(slice_msk.GetSize(), sitk.sitkUInt8)
                slice_msk_bbox_rev = sitk.Image(slice_msk.GetSize(), sitk.sitkUInt8)
                slice_msk_bbox_rev = slice_msk_bbox_rev + 1

                for label in label_statistic.GetLabels():
                    # Get bounding box and add 5 pixels to every side of the bounding box
                    bbox = label_statistic.GetBoundingBox(label)
                    bbox = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 10, bbox[3] + 10)

                    slice_msk_bbox[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = 1
                    slice_msk_bbox_rev[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = 0

                # Convert slice_msk_bbox to an RGB image
                red_channel = sitk.Cast(slice_msk_bbox * 255, sitk.sitkUInt8)
                green_channel = sitk.Image(slice_msk_bbox.GetSize(), sitk.sitkUInt8)
                blue_channel = sitk.Image(slice_msk_bbox.GetSize(), sitk.sitkUInt8)
                slice_msk_bbox_rgb = sitk.Compose(red_channel, green_channel, blue_channel)

                # Prepare the slice image for masking
                slice_img = sitk.Clamp(slice_img, lowerBound=-1000, upperBound=1000)
                slice_img = sitk.RescaleIntensity(slice_img, 0, 255)
                slice_img = sitk.Cast(slice_img, sitk.sitkUInt8)

                # Convert the image to RGB
                slice_img_rgb = sitk.Compose(slice_img, slice_img, slice_img)
                slice_msk_bbox_rev.CopyInformation(slice_img_rgb)
                slice_msk_bbox_rgb.CopyInformation(slice_img_rgb)

                # Apply the red mask on the image
                slice_img_rgb_masked = sitk.Mask(slice_img_rgb, sitk.Cast(slice_msk_bbox_rev, sitk.sitkUInt8))
                slice_img_rgb = slice_img_rgb + slice_msk_bbox_rgb
                slice_img_rgb_masked = slice_img_rgb_masked + slice_msk_bbox_rgb

                # Rescale the mask for saving
                slice_msk = sitk.Cast(slice_msk * 255, sitk.sitkUInt8)

                out_dir_input = Path(os.path.join(args.output_dir, patient_id, study_id, "Masked"))
                out_dir_output = Path(os.path.join(args.output_dir, patient_id, study_id, "Scan"))
                out_dir_mask = Path(os.path.join(args.output_dir, patient_id, study_id, "Mask"))

                out_dir_input.mkdir(parents=True, exist_ok=True)
                out_dir_output.mkdir(parents=True, exist_ok=True)
                out_dir_mask.mkdir(parents=True, exist_ok=True)

                sitk.WriteImage(slice_img_rgb_masked, os.path.join(out_dir_input, f'{file_name}_{z}.png'))
                sitk.WriteImage(slice_img, os.path.join(out_dir_output, f'{file_name}_{z}.png'))
                sitk.WriteImage(slice_msk, os.path.join(out_dir_mask, f'{file_name}_{z}.png'))


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
