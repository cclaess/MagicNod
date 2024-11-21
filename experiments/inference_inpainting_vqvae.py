import os
import argparse
from glob import glob
from pathlib import Path
from tifffile import imsave

import torch
import numpy as np
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
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from generative.networks.nets import VQVAE
from tqdm import tqdm
from scipy.ndimage import label, find_objects

from magicnod.transforms import FilterSlicesByMaskFuncd


def mask_with_bounding_boxes(binary_mask):
    """
    Replace each connected component in the binary mask with its bounding box.

    Args:
        binary_mask (torch.Tensor): Binary mask tensor of shape (batch, 1, height, width).

    Returns:
        torch.Tensor: Modified mask of the same shape as the input, where connected components
                      are replaced by their bounding boxes.
    """
    assert binary_mask.ndim == 4, "Input mask must have shape (batch, 1, height, width)"
    assert binary_mask.size(1) == 1, "The second dimension of the mask must be 1"

    # Convert to numpy array for processing
    binary_mask_np = binary_mask.squeeze(1).cpu().numpy()  # Shape: (batch, height, width)

    # Initialize a new mask to store the result
    result_mask_np = np.ones_like(binary_mask_np)

    for batch_idx, mask in enumerate(binary_mask_np):
        labeled_mask, _ = label(mask)  # Label connected components
        slices = find_objects(labeled_mask)  # Find bounding box slices for each component

        for s in slices:
            if s is not None:  # Valid slice
                min_row, max_row = s[0].start, s[0].stop
                min_col, max_col = s[1].start, s[1].stop

                # Fill the bounding box region in the result mask
                result_mask_np[batch_idx, min_row:max_row, min_col:max_col] = 0

    # Convert the result back to a torch tensor with same dtype and device as the input
    result_mask = torch.tensor(result_mask_np, dtype=binary_mask.dtype, device=binary_mask.device)
    result_mask = result_mask.unsqueeze(1)  # Add the channel dimension back
    
    return result_mask


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
    parser.add_argument("--num-channels", type=int, nargs="+", default=(256, 512), help="Number of channels in the model")
    parser.add_argument("--num-res-channels", type=int, default=512, help="Number of channels in the residual blocks")
    parser.add_argument("--num-res-layers", type=int, default=2, help="Number of residual layers in the model")
    parser.add_argument("--downsample-parameters", type=int, nargs=4, default=(2, 4, 1, 1), help="Parameters for the downsampling layers")
    parser.add_argument("--upsample-parameters", type=int, nargs=5, default=(2, 4, 1, 1, 0), help="Parameters for the upsampling layers")
    parser.add_argument("--num-embeddings", type=int, default=256, help="Number of embeddings in the VQ-VAE")
    parser.add_argument("--embedding-dim", type=int, default=632, help="Dimension of the embeddings in the VQ-VAE")

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser


def main(args):
    # Set seed for reproducibility
    set_determinism(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data paths
    valid_paths = [{"image": p, "mask": p.replace("image", "mask")} for p in glob(
        os.path.join(args.data_dir, "valid", "**", "image.nii.gz"), recursive=True)]

    # Define the transforms
    valid_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(384, 384, 128)),
        ToTensord(keys=["image", "mask"]),
        FilterSlicesByMaskFuncd(
            keys=["image", "mask"], 
            mask_key="mask", 
            mask_filter_func=lambda x: x.sum(dim=(0, 1, 2)) > 0,
            slice_dim=3,
        ),
    ])

    # Create the data loader
    val_data = Dataset(data=valid_paths, transform=valid_transforms)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)

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
    model.load_state_dict(state_dict)
    model.eval()

    # Loop over the data
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing batches")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Create rectangular masks
            rect_masks = mask_with_bounding_boxes(masks)

            # Combine image and mask as input
            masked_images = images.clone()
            masked_images = masked_images * rect_masks  # Apply mask to image
            inversed_masks = rect_masks * -1 + 1  # Inverse the mask
            input_images = torch.cat([masked_images, inversed_masks], dim=1)

            # Generate reconstructed images
            reconstructed_images, _ = model(input_images)

            # Save the results
            for slice_num, (orig_image, mask, recon_image) in enumerate(
                zip(images, rect_masks, reconstructed_images)):

                # Get the original image path from metatensor
                orig_image = orig_image[0].cpu().numpy()
                mask = mask[0].cpu().numpy()
                recon_image = recon_image[0].cpu().numpy()

                # Normalize images for saving
                orig_image = (orig_image * 2000 - 1000).astype(np.int16)  # Undo normalization
                mask = mask.astype(np.uint8)
                recon_image = (recon_image * 2000 - 1000).astype(np.int16)

                # Save the individual slices as tiff images
                save_dir = output_dir / f"batch_{batch_idx}"
                save_dir.mkdir(parents=True, exist_ok=True)

                imsave(save_dir / f"{slice_num:04}_original.tiff", orig_image)
                imsave(save_dir / f"{slice_num:04}_mask.tiff", mask)
                imsave(save_dir / f"{slice_num:04}_reconstruction.tiff", recon_image)

                # Save the combined image
                combined_image = np.concatenate([orig_image, mask, recon_image], axis=1)
                imsave(save_dir / f"{slice_num:04}_combined.tiff", combined_image)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
