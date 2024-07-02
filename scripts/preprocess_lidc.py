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

    subjects = [im for im in os.listdir(os.path.join(args.input_dir, 'images')) if im.startswith('LIDC-IDRI-')]
    for subject in subjects:

        study_ids = os.listdir(os.path.join(args.input_dir, 'images', subject))
        for study_id in study_ids:

            # check if there is any segmentation for `study_id`
            if not os.path.exists(os.path.join(args.input_dir, 'masks', subject, study_id)):
                continue

            series_ids = os.listdir(os.path.join(args.input_dir, 'images', subject, study_id))
            series_id = series_ids[0]

            sitk_img = load_dicom_volume(os.path.join(args.input_dir, 'images', subject, study_id, series_id), sitk.sitkUInt16)
            img_spacing = sitk_img.GetSpacing()
            img_origin = sitk_img.GetOrigin()
            img_size = sitk_img.GetSize()

            nodule_dict = {}
            nodules = [nod for nod in os.listdir(os.path.join(args.input_dir, 'masks', subject, study_id)) if 'Segmentation of Nodule' in nod]
            for nodule in nodules:

                nodule_num = re.split(r"\W(?=Nodule|- Annotation)", nodule)[1].split()[1]

                sitk_msk = load_dicom_volume(os.path.join(args.input_dir, 'masks', subject, study_id, nodule), sitk.sitkUInt8)
                if len(sitk_msk.GetSize()) == 4 and sitk.msk.GetSize()[3] == 1:
                    sitk_msk = sitk_msk[..., 0]
                
                sitk_msk.SetSpacing(img_spacing)
                sitk_msk = sitk.Not(sitk_msk == 0)

                msk_origin = sitk_msk.GetOrigin()
                msk_size = sitk_msk.GetSize()

                upadding = [0, 0, round(max(img_origin[2], msk_origin[2]) - min(img_origin[2], msk_origin[2]) / img_spacing[2])]
                lpadding = [0, 0, img_size[2] - msk_size[2] - upadding[2]]

                sitk_msk = sitk.ConstantPad(sitk_msk, upadding, lpadding, 0)
                sitk_msk.CopyInformation(sitk_img)

                # put masks in dictionary to make combined mask later
                if nodule_num not in nodule_dict.keys():
                    nodule_dict.update({nodule_num: [sitk_msk]})
                else:
                    nodule_dict[nodule_num].append(sitk_msk)
            

                label_statistic.Execute(sitk_msk, sitk_msk)
                bbox = label_statistic.GetBoundingBox(1)

                for slice_num in range(bbox[2], bbox[2] + bbox[5]):

                    sitk_img_slice = sitk_img[..., slice_num]
                    sitk_msk_slice = sitk_msk[..., slice_num]
                    sitk_masked = sitk.Mask(sitk_img_slice, sitk_msk_slice)

                    img_save_path = Path(os.path.join(args.output_dir, 'images', subject, study_id, ''))





if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
