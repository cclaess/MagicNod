import os
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def get_args_parser():
    """
    Get the argument parser for the script.

    Returns:
    - argparse.ArgumentParser: Argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Preprocess LNDb dataset")
    
    # Experiment and data configuration
    parser.add_argument("--data-dir", default=r"C:\Users\20173869\Downloads\LNDb", type=str, help="Path to the data directory")
    parser.add_argument("--output-dir", default=r"C:\Users\20173869\Downloads\LNDb-processed", type=str, help="Path to the output directory")
    
    return parser


def calculate_nodule_center_and_size(bbox):
    center = [(bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2, (bbox[2] + bbox[5]) / 2]
    size = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
    return center, size


def nodules_are_same(nodule1, nodule2, threshold_distance=10):
    center1, size1 = calculate_nodule_center_and_size(nodule1)
    center2, size2 = calculate_nodule_center_and_size(nodule2)

    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    size_diff = np.linalg.norm(np.array(size1) - np.array(size2))
    
    return distance < threshold_distance  # and size_diff < threshold_distance


def main(args):

    # Get the list of files containing the image series
    image_paths = glob(os.path.join(args.data_dir, "*data*", "*.mhd"))

    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize a label statistics filter
    label_stats = sitk.LabelStatisticsImageFilter()

    # Loop over the series directories
    for image_path in image_paths:

        # Read the image series
        image = sitk.ReadImage(image_path)

        # Subject ID is the filename of the image series
        subject_id = Path(image_path).stem

        # Load the segmentation files of different annotators
        mask_paths = glob(os.path.join(args.data_dir, "masks", f"{subject_id}*.mhd"))

        annot_masks = [sitk.ReadImage(mask_path) for mask_path in mask_paths]
        avail_labels = []
        for annot_mask in annot_masks:
            label_stats.Execute(image, annot_mask)
            labels = label_stats.GetLabels()
            avail_labels.append([label for label in labels if label != 0])
        
        annot_bboxes = []
        for annot_mask, labels in zip(annot_masks, avail_labels):
            label_stats.Execute(image, annot_mask)
            annot_bboxes.append([{"bbox": label_stats.GetBoundingBox(label), "label": label} for label in labels])
        
        # Match the nodules based on the overlap of the bounding boxes
        detected_nodules = {}
        for annot_idx, bboxes in enumerate(annot_bboxes):
            for bbox in bboxes:
                matched = False
                for _, values in detected_nodules.items():
                    if any(nodules_are_same(bbox["bbox"], existing_bbox) for existing_bbox in values['bboxes']):
                        values['annotators'].append(annot_idx)
                        values['bboxes'].append(bbox["bbox"])
                        values['labels'].append(bbox["label"])
                        matched = True
                        break
                if not matched:
                    detected_nodules[len(detected_nodules)] = {'annotators': [annot_idx], 'bboxes': [bbox["bbox"]], 'labels': [bbox["label"]]}

        # Filter nodules detected by at least 2 annotators
        valid_nodules = {k: v for k, v in detected_nodules.items() if len(v['annotators']) >= 2}

        # Create final binary mask using majority voting
        combined_mask = sitk.Image(annot_masks[0].GetSize(), sitk.sitkUInt8)
        combined_mask.CopyInformation(annot_masks[0])

        for nodule_id, data in valid_nodules.items():
            # Extract relevant masks
            nodule_masks = [annot_masks[annot_idx] for annot_idx in data['annotators']]
            
            # Mask the current nodule using the bounding box
            nodule_masks = [sitk.Equal(m, data['labels'][idx]) for idx, m in enumerate(nodule_masks)]

            # Combine the masks using majority voting
            nodule_mask = sitk.BinaryThreshold(sitk.LabelVoting(nodule_masks), lowerThreshold=1)

            # Add the nodule mask to the combined mask
            combined_mask = sitk.Or(combined_mask, nodule_mask)

        # Save the combined mask
        mask_output_path = output_dir / Path(mask_paths[0]).parent.relative_to(args.data_dir) / f"{subject_id}.nii.gz"
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)

        image_output_path = output_dir / Path(image_path).relative_to(args.data_dir).with_suffix(".nii.gz")
        image_output_path.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(combined_mask, str(mask_output_path))
        sitk.WriteImage(image, str(image_output_path))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
