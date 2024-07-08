import os
import csv
import argparse
from pathlib import Path

import pylidc as pl
import SimpleITK as sitk


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', default='/path/to/output/folder', type=str)

    return parser


def main(args):

    scans = pl.query(pl.Scan)
    for scan in scans:

        nods = scan.cluster_annotations()
        annotations = pl.query(pl.Annotation).filter(pl.Annotation.scan_id == scan.id)

        sitk_image = sitk.GetImageFromArray(scan.to_volume().transpose(2, 0, 1))
        sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
        sitk_image.SetSpacing((scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing))
        
        sitk_label = sitk.Image(sitk_image.GetSize(), sitk.sitkUInt8)
        sitk_label.CopyInformation(sitk_image)

        meta_path = os.path.join(args.output_dir, "meta", f'{scan.id}.csv')
        with open(meta_path, "w", newline='') as f:
            writer = csv.writer(f, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "id", "bbox_x", "bbox_y", "bbox_z",
                "subtlety", 
                "internalStructure", 
                "calcification", 
                "sphericity",
                "margin",
                "lobulation",
                "spiculation",
                "texture",
                "malignancy",
            ])

            for n in nods:
                for a in n:
                    ann = annotations.filter(pl.Annotation.id == a.id).first()
                    bbox = ann.bbox()

                    # make mask and pad back to image size using bbox
                    image_size = sitk_image.GetSize()
                    lpadding = [int(bbox[1].start), int(bbox[0].start), int(bbox[2].start)]
                    upadding = [int(image_size[1] - bbox[1].stop), int(image_size[0] - bbox[0].stop), int(image_size[2] - bbox[2].stop)]
                    mask = sitk.GetImageFromArray(ann.boolean_mask().transpose(2, 0, 1).astype(int))
                    mask = sitk.Cast(mask, sitk.sitkUInt8)
                    mask = sitk.ConstantPad(mask, upadding, lpadding, 0)
                    mask.SetSpacing(sitk_image.GetSpacing())
                    mask.SetOrigin(sitk_image.GetOrigin())

                    sitk_label = sitk.Or(sitk_label, mask)

                    writer.writerow([
                        a.id, 
                        f"{bbox[0].start}:{bbox[0].stop}",
                        f"{bbox[1].start}:{bbox[1].stop}",
                        f"{bbox[2].start}:{bbox[2].stop}",
                        a.subtlety, 
                        a.internalStructure, 
                        a.calcification, 
                        a.sphericity,
                        a.margin,
                        a.lobulation,
                        a.spiculation,
                        a.texture,
                        a.malignancy,
                    ])
            
        image_path = os.path.join(args.output_dir, "images", f'{scan.id}.nii.gz')
        label_path = os.path.join(args.output_dir, "labels", f'{scan.id}.nii.gz')

        sitk.WriteImage(sitk_image, image_path)
        sitk.WriteImage(sitk_label, label_path)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    Path(os.path.join(args.output_dir, "meta")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "images")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "labels")).mkdir(parents=True, exist_ok=True)

    main(args)
