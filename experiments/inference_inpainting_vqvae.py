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
from monai.utils import set_determinism
from generative.networks.nets import VQVAE
from tqdm import tqdm
from scipy.ndimage import label, find_objects
from torchvision.utils import make_grid
from PIL import Image

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
        ToTensord(keys=["image", "mask"]),
        FilterSlicesByMaskFuncd(
            keys=["image", "mask"], 
            mask_key="mask", 
            mask_filter_func=lambda x: x.sum(dim=(0, 1, 2)) > 0,
            slice_dim=3,
        ),
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

            # Get the original paths
            orig_path = Path(data[0]["image"].meta["filename_or_obj"])


            for slice_idx, slice_data in enumerate(data):
                image = slice_data["image"].unsqueeze(0).to(device)
                mask = slice_data["mask"].unsqueeze(0).to(device)

                # Create rectangular mask
                rect_mask = mask_with_bounding_boxes(mask)

                # Combine image and mask as input
                masked_image = image.clone()
                masked_image = masked_image * rect_mask  # Apply mask to image
                inversed_mask = rect_mask * -1 + 1  # Inverse the mask
                input_image = torch.cat([masked_image, inversed_mask], dim=1)

                # Generate reconstructed image
                reconstructed_image, _ = model(input_image)

                # Make grid to save later
                grid_image = make_grid(
                    torch.cat([image, rect_mask, reconstructed_image], dim=0), 
                    nrow=3,
                    normalize=True,
                    value_range=(0, 1),
                )

                # Get the original image path from metatensor
                image = image.squeeze(0).cpu().numpy()
                mask = mask.squeeze(0).cpu().numpy()
                reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy()

                # Normalize images for saving
                image = (image * 2000 - 1000).astype(np.int16)  # Undo normalization
                mask = mask.astype(np.uint8)
                reconstructed_image = (reconstructed_image * 2000 - 1000).astype(np.int16)

                # Save the individual slices as tiff images
                save_dir = output_dir / Path(orig_path).relative_to(args.data_dir).parent
                save_dir.mkdir(parents=True, exist_ok=True)

                image_name = str(orig_path).replace(".nii.gz", f"_slice_{slice_idx:04}.tiff")
                mask_name = str(orig_path).replace("image.nii.gz", f"mask_slice_{slice_idx:04}.tiff")
                recon_name = str(orig_path).replace("image.nii.gz", f"recon_slice_{slice_idx:04}.tiff")
                grid_name = str(orig_path).replace("image.nii.gz", f"grid_{slice_idx:04}.png")

                imsave(save_dir / image_name, image)
                imsave(save_dir / mask_name, mask)
                imsave(save_dir / recon_name, reconstructed_image)

                # Save the grid image as PNG
                grid_image = grid_image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
                grid_image = Image.fromarray(grid_image)
                grid_image.save(save_dir / grid_name)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
