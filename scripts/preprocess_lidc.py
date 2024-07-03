import os
import re
import argparse
from pathlib import Path

import SimpleITK as sitk


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', defualt='/path/to/lidc/folder', type=str)
    parser.add_argument('--output_dir', '-o', default='/path/to/output/folder', type=str)


def load_dicom_volume(f, pixelID=None):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(f)
    reader.SetFileNames(dicom_names)
    if pixelID is not None:
        reader.SetOutputPixelType(pixelID)

    image = reader.Execute()
    return image


def main(args):

    # initialize the label statistics filter
    label_statistic = sitk.LabelIntensityStatisticsImageFilter()

    # check if input folder exist and is a directory
    assert os.path.exists(args.input_dir), f"Input directory `{args.input_dir}` does not exist."
    assert os.path.isdir(args.input_dir), f"Input directory `{args.input_dir}` is not a directory."

    # check if input folder has subdirectories `images`and `masks`
    assert 'images' in os.listdir(args.input_dir), f"Input directory is expected to contain subfolders `images` and `masks`."
    assert 'masks' in os.listdir(args.input_dir), f"Input directory is expected to contain subfolders `images` and `masks`."

    # get all subjects
    subjects = [im for im in os.listdir(os.path.join(args.input_dir, 'images')) if im.startswith('LIDC-IDRI-')]
    for subject in subjects:

        # get all studies for each subject
        study_ids = os.listdir(os.path.join(args.input_dir, 'images', subject))
        for study_id in study_ids:

            # check if there is any segmentation for `study_id`
            if not os.path.exists(os.path.join(args.input_dir, 'masks', subject, study_id)):
                continue
            
            # get all series for each study and select only the first series
            series_ids = os.listdir(os.path.join(args.input_dir, 'images', subject, study_id))
            series_id = series_ids[0]

            # load image and retrieve spacing, origin and size
            sitk_img = load_dicom_volume(os.path.join(args.input_dir, 'images', subject, study_id, series_id), sitk.sitkUInt16)
            img_spacing = sitk_img.GetSpacing()
            img_origin = sitk_img.GetOrigin()
            img_size = sitk_img.GetSize()

            # load all nodules for the study and bundle nodule segmentations from different annotators
            nodule_dict = {}
            nodules = [nod for nod in os.listdir(os.path.join(args.input_dir, 'masks', subject, study_id)) if 'Segmentation of Nodule' in nod]
            for nodule in nodules:
                
                # get nodule number
                nodule_num = re.split(r"\W(?=Nodule|- Annotation)", nodule)[1].split()[1]

                # load mask and retrieve origin and size
                sitk_msk = load_dicom_volume(os.path.join(args.input_dir, 'masks', subject, study_id, nodule), sitk.sitkUInt8)
                msk_origin = sitk_msk.GetOrigin()
                msk_size = sitk_msk.GetSize()
                if len(msk_size) == 4 and msk_size[3] == 1:
                    sitk_msk = sitk_msk[..., 0]
                
                # check if mask and image have the same spacing
                sitk_msk.SetSpacing(img_spacing)

                # set the label to 1
                sitk_msk = sitk.Not(sitk_msk == 0)
                sitk_msk = sitk.Cast(sitk_msk, sitk.sitkUInt8)

                # calculate padding for mask
                upadding = [0, 0, round(max(img_origin[2], msk_origin[2]) - min(img_origin[2], msk_origin[2]) / img_spacing[2])]
                lpadding = [0, 0, img_size[2] - msk_size[2] - upadding[2]]

                # pad mask
                sitk_msk = sitk.ConstantPad(sitk_msk, upadding, lpadding, 0)
                sitk_msk.CopyInformation(sitk_img)

                # put masks in dictionary to make combined mask of annotators later
                if nodule_num not in nodule_dict.keys():
                    nodule_dict.update({nodule_num: [sitk_msk]})
                else:
                    nodule_dict[nodule_num].append(sitk_msk)
            
            # make consensus mask of annotators for every nodule by using majority voting
            for nodule_num in nodule_dict.keys():
                consensus_mask = sitk.LabelVoting(nodule_dict[nodule_num])
                consensus_mask = sitk.Not(consensus_mask == 0)
                consensus_mask = sitk.Cast(consensus_mask, sitk.sitkUInt8)
                nodule_dict[nodule_num] = consensus_mask
            
            # combine all nodule masks to get the final mask
            combined_mask = sitk.Image(img_size, sitk.sitkUInt8)
            for nodule_num in nodule_dict.keys():
                combined_mask = sitk.Or(combined_mask, nodule_dict[nodule_num])

            # save the axial slices of the combined mask and image if and only if the slice covers one or more nodules as png
            # also save the axial image slices with black squares covering the nodules as inputs for the pix2pix model
            for z in range(img_size[2]):
                slice_img = sitk_img[:, :, z]
                slice_msk = combined_mask[:, :, z]
                if sitk.GetArrayFromImage(slice_msk).sum() > 0:
                    # create a new mask with black squares covering the classes 1
                    slice_msk_labeled = sitk.ConnectedComponent(slice_msk)
                    label_statistic.Execute(slice_msk, slice_msk_labeled)
                    slice_msk_bbox = sitk.Image(slice_msk.GetSize(), sitk.sitkUInt8)
                    slice_msk_bbox.CopyInformation(slice_msk)
                    for label in label_statistic.GetLabels():
                        bbox = label_statistic.GetBoundingBox(label)
                        slice_msk_bbox[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1
                    
                    # mask the input image with black squares
                    slice_img_bbox = sitk.Mask(slice_img, slice_msk_bbox)

                    out_dir_input = Path(os.path.join(args.output_dir, 'input', subject, study_id))
                    out_dir_output = Path(os.path.join(args.output_dir, 'output', subject, study_id))

                    out_dir_input.mkdir(parents=True, exist_ok=True)
                    out_dir_output.mkdir(parents=True, exist_ok=True)

                    sitk.WriteImage(slice_img_bbox, os.path.join(out_dir_input, f'{z}.png'))
                    sitk.WriteImage(slice_img, os.path.join(out_dir_output, f'{z}.png'))


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
