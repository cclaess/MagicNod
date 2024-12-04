import os
import argparse
from glob import glob
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    ToTensord,
)
from monai.utils import set_determinism
from generative.networks.nets import VQVAE
from tqdm import tqdm
from scipy.ndimage import label, find_objects
from torchvision.utils import make_grid
from PIL import Image

from magicnod.transforms import FilterSlicesByMaskFuncd


def get_args_parser():
    """
    Get the argument parser for the inference script.

    Returns:
    - argparse.ArgumentParser: Argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Inference for nodule segmentation and saving slices")
    parser.add_argument("--data-dir", required=True, type=str, help="Path to the data directory")
    parser.add_argument("--output-dir", required=True, type=str, help="Path to save the output slices")

    # Model configuration
    parser.add_argument("--model-path", required=True, type=str, help="Path to the trained model")
    parser.add_argument("--num-channels", type=int, nargs="+", default=(256, 512, 512), help="Number of channels in the model")
    parser.add_argument("--num-res-channels", type=int, nargs="+", default=(256, 512, 512), help="Number of channels in the residual blocks")
    parser.add_argument("--num-res-layers", type=int, default=3, help="Number of residual layers in the model")
    parser.add_argument("--downsample-parameters", type=int, nargs=4, default=(2, 4, 1, 1), help="Parameters for the downsampling layers")
    parser.add_argument("--upsample-parameters", type=int, nargs=5, default=(2, 4, 1, 1, 0), help="Parameters for the upsampling layers")
    parser.add_argument("--num-embeddings", type=int, default=256, help="Number of embeddings in the VQ-VAE")
    parser.add_argument("--embedding-dim", type=int, default=632, help="Dimension of the embeddings in the VQ-VAE")

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser


def mask_with_circle(binary_mask):
    """
    Replace each connected component in the binary mask with a circle of minimal size that covers the whole 
    component.

    Args:
        binary_mask (torch.Tensor): Binary mask tensor of shape (1, height, width).

    Returns:
        torch.Tensor: Modified mask of the same shape as the input, where connected components
                        are replaced by circles.
    """
    assert binary_mask.ndim == 3, "Input mask must have shape (1, height, width)"
    assert binary_mask.size(0) == 1, "The first dimension of the mask must be 1"

    # Convert to numpy array for processing
    binary_mask_np = binary_mask.squeeze(0).cpu().numpy()  # Shape: (height, width)
    nodules_info = []
    nodules_mask = []

    labeled_mask, _ = label(binary_mask_np)
    slices = find_objects(labeled_mask)

    for s in slices:
        if s is not None:
            min_row, max_row = s[0].start, s[0].stop
            min_col, max_col = s[1].start, s[1].stop

            # Get the center of the bounding box
            center_row = (min_row + max_row) // 2
            center_col = (min_col + max_col) // 2
            radius = max(max_row - min_row, max_col - min_col) // 2
            nodules_info.append((center_row, center_col, radius))

            # Create a circle mask
            result_mask_np = np.ones_like(binary_mask_np)
            y, x = np.ogrid[:binary_mask_np.shape[0], :binary_mask_np.shape[1]]
            circle = (x - center_col) ** 2 + (y - center_row) ** 2 <= (radius + 3) ** 2  # Add a margin of 3 pixels
            result_mask_np = np.logical_not(np.logical_and(result_mask_np, circle))
            nodules_mask.append(result_mask_np)

    # Convert the result back to a torch tensor with same dtype and device as the input
    # Add the channel dimension back
    nodules_mask = [torch.tensor(
            mask, 
            dtype=binary_mask.dtype, 
            device=binary_mask.device
        ).unsqueeze(0) for mask in nodules_mask]

    return nodules_mask, nodules_info


def create_circular_average_kernel(size, radius):
    """
    Create a circular averaging kernel in PyTorch.

    Args:
        size (int): The width and height of the kernel (assumes square kernel).
        radius (float): The radius of the circle to average within.

    Returns:
        torch.Tensor: A 2D kernel with the circle averaging mask.
    """
    # Create a grid of coordinates centered at the middle
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = (size - 1) / 2
    distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Create the circular mask
    mask = (distance <= radius).float()

    # Normalize to ensure the sum of the kernel equals 1
    kernel = mask / mask.sum()

    # Add a batch and channel dimension
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    return kernel


def map_pixel_to_original(p_new, affine_new, affine_orig):
    """
    Map a pixel coordinate from the resampled space back to the original space,
    rounding to the nearest pixel.

    Args:
        p_new (tuple): Pixel coordinate in the resampled space (x, y, z).
        affine_new (numpy.ndarray): New affine matrix after resampling (4x4).
        affine_orig (numpy.ndarray): Original affine matrix before resampling (4x4).

    Returns:
        numpy.ndarray: Corresponding coordinate in the original space, rounded to the nearest pixel.
    """
    # Convert the pixel coordinate to homogeneous form
    p_new_h = np.array([*p_new, 1])  # (x, y, z, 1)

    # Compute the world coordinate in the new space
    p_world = affine_new @ p_new_h

    # Transform the world coordinate back to the original pixel space
    p_orig_h = np.linalg.inv(affine_orig) @ p_world

    # Return only the first three coordinates (x, y, z), rounded to the nearest pixel
    return np.round(p_orig_h[:3]).astype(int)


def main(args):
    # Set seed for reproducibility
    set_determinism(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the excel file with the annotations and remove nodules with <2 annotations
    annotations = pd.read_csv(Path(args.data_dir) / "annotations.csv")
    nodule_counts = annotations.groupby(["PatientID", "NoduleID"]).size()
    valid_nodules = nodule_counts[nodule_counts >= 2].reset_index()
    annotations = pd.merge(annotations, valid_nodules, on=["PatientID", "NoduleID"], how="inner") 

    # Load data paths
    train_paths = [{"image": p, "mask": p.replace("image", "mask")} for p in glob(
        os.path.join(args.data_dir, "train", "**", "image.nii.gz"), recursive=True)]
    valid_paths = [{"image": p, "mask": p.replace("image", "mask")} for p in glob(
        os.path.join(args.data_dir, "valid", "**", "image.nii.gz"), recursive=True)]
    paths = train_paths + valid_paths

    # Define the transforms
    transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(384, 384, 128)),
        FilterSlicesByMaskFuncd(
            keys=["image", "mask"], 
            mask_key="mask", 
            mask_filter_func=lambda x: x.sum(dim=(0, 1, 2)) > 0,
            slice_dim=3,
        ),
        ToTensord(keys=["image", "mask"]),
    ])

    # Initialize the model
    model = VQVAE(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        num_channels=args.num_channels,
        num_res_channels=args.num_res_channels,
        num_res_layers=args.num_res_layers,
        downsample_parameters=args.downsample_parameters,
        upsample_parameters=args.upsample_parameters,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    )

    # Load the trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model_path, map_location=device)
    model.to(device).load_state_dict(state_dict)
    model.eval()

    # Loop over the data
    with torch.no_grad():
        for path in tqdm(paths, desc="Processing batches"):
            
            # Load the data
            data = transforms(path)

            # Check if the data list is empty
            if not data:
                continue

            # Get the original paths
            orig_path = Path(data[0]["image"].meta["filename_or_obj"])

            # Extract the subject ID from the path
            subject_id = orig_path.parent.parent.parent.name

            # Get the nodule annotations from the subject
            subject_annotations = annotations[annotations["PatientID"] == subject_id]
            assert len(subject_annotations) > 0, f"No annotations found for subject {subject_id}"

            # Get a list of the nodules with their maximum xy bounding box
            nodules = []
            for _, row in subject_annotations.iterrows():
                # Check if nodule ID already exists
                if row["NoduleID"] in [n["NoduleID"] for n in nodules]:
                    idx = [n["NoduleID"] for n in nodules].index(row["NoduleID"])
                    nodules[idx]["BBoxX"] = (
                        min(nodules[idx]["BBoxX"][0], row["BBoxMinX"]), 
                        max(nodules[idx]["BBoxX"][1], row["BBoxMaxX"])
                    )
                    nodules[idx]["BBoxY"] = (
                        min(nodules[idx]["BBoxY"][0], row["BBoxMinY"]), 
                        max(nodules[idx]["BBoxY"][1], row["BBoxMaxY"])
                    )
                else:
                    nodules.append({
                        "NoduleID": row["NoduleID"],
                        "BBoxX": (row["BBoxMinX"], row["BBoxMaxX"]),
                        "BBoxY": (row["BBoxMinY"], row["BBoxMaxY"]),
                    })
                
            print(nodules)


            for slice_idx, slice_data in enumerate(data):
                image = slice_data["image"]
                mask = slice_data["mask"]

                # Create rectangular mask
                nodules_mask, nodules_info = mask_with_circle(mask)

                # Send the data to the device
                image = image.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)
                nodules_mask = [m.unsqueeze(0).to(device) for m in nodules_mask]

                for nodule_info, nodule_mask in zip(nodules_info, nodules_mask):
                    
                    x, y, _ = map_pixel_to_original((*nodule_info[:2], 0), mask.affine.numpy(), mask.meta["original_affine"])

                    # Match the nodule info with the bounding box
                    nod_id = None
                    for nodule in nodules:
                        if nodule["BBoxX"][0] <= x <= nodule["BBoxX"][1] and \
                            nodule["BBoxY"][0] <= y <= nodule["BBoxY"][1]:
                            nod_id = nodule["NoduleID"]
                            break
                    assert nod_id, "Nodule not found in the annotations"

                    # Combine image and mask as input
                    masked_image = image.clone()
                    masked_image = masked_image * nodule_mask  # Apply mask to image
                    inversed_mask = nodule_mask * -1 + 1  # Inverse the mask
                    input_image = torch.cat([masked_image, inversed_mask], dim=1)

                    # Generate reconstructed image
                    reconstructed_image, _ = model(input_image)

                    # Cut and paste the reconstructed image within the mask region back to the original image
                    # Use a gaussian weighting around the edges to blend the images
                    
                    # Get bounding box of the mask
                    smooth_mask = torch.zeros_like(mask)
                    smooth_mask = inversed_mask.clone()

                    # Apply convolutional filter to mask to create a smooth transition
                    kernel = create_circular_average_kernel(7, 3).to(device)
                    smooth_mask = torch.nn.functional.conv2d(smooth_mask, kernel, padding=3)

                    # Cut and paste the reconstructed image within the mask region back to the original image
                    combined_image = image * (smooth_mask * -1 + 1) + reconstructed_image * smooth_mask

                    # Make grid to save later
                    grid_image = make_grid(
                        torch.cat([
                            image.permute(0, 1, 3, 2), 
                            masked_image.permute(0, 1, 3, 2),
                            reconstructed_image.permute(0, 1, 3, 2),
                            smooth_mask.permute(0, 1, 3, 2),
                            combined_image.permute(0, 1, 3, 2)
                        ], dim=0), 
                        nrow=5,
                        normalize=True,
                        value_range=(0, 1),
                    )

                    # Remove redundant dimensions
                    image_array = image.squeeze(0).permute(2, 1, 0).cpu().numpy()
                    mask_array = mask.squeeze(0).permute(2, 1, 0).cpu().numpy()
                    reconstructed_image_array = reconstructed_image.squeeze(0).permute(2, 1, 0).cpu().numpy()
                    combined_image_array = combined_image.squeeze(0).permute(2, 1, 0).cpu().numpy()

                    # Normalize images for saving
                    image_array = (image_array * 255).astype(np.uint8)  # Undo normalization
                    mask_array = (mask_array * 255).astype(np.uint8)
                    reconstructed_image_array = (reconstructed_image_array.clip(0., 1.) * 255).astype(np.uint8)
                    combined_image_array = (combined_image_array.clip(0., 1.) * 255).astype(np.uint8)

                    # Save the individual slices as tiff images
                    save_dir = output_dir / Path(orig_path).relative_to(args.data_dir).parent
                    save_dir.mkdir(parents=True, exist_ok=True)

                    image_name = f"image_slice={slice_idx:04}_nod={nod_id}.png"
                    mask_name = f"mask_slice=_{slice_idx:04}_nod={nod_id}.png"
                    recon_name = f"recon_slice={slice_idx:04}_nod={nod_id}.png"
                    combined_name = f"combined_slice={slice_idx:04}_nod={nod_id}.png"
                    grid_name = f"grid_slice={slice_idx:04}_nod={nod_id}.png"

                    Image.fromarray(image_array).save(save_dir / image_name)
                    Image.fromarray(mask_array).save(save_dir / mask_name)
                    Image.fromarray(reconstructed_image_array).save(save_dir / recon_name)
                    Image.fromarray(combined_image_array).save(save_dir / combined_name)

                    # Save the grid image as PNG
                    grid_image_array = grid_image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
                    Image.fromarray(grid_image_array).save(save_dir / grid_name)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
