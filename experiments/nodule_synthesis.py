import argparse
from glob import glob
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

# from diffusers.utils import load_image
# from lungmask import LMInferer
from scipy.ndimage import label, find_objects
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
from generative.networks.nets import VQVAE

from magicnod.transforms import FilterSlicesByMaskFuncd


def get_args_parser():

    parser = argparse.ArgumentParser(description="Inferring with InstructPix2Pix")

    parser.add_argument(
        "--model-path-vqvae", type=str, required=True, help="Path to the VQ-VAE model"
    )

    parser.add_argument(
        "--num-channels",
        type=int,
        nargs="+",
        default=(256, 512, 512),
        help="Number of channels in the model",
    )
    parser.add_argument(
        "--num-res-channels",
        type=int,
        nargs="+",
        default=(256, 512, 512),
        help="Number of channels in the residual blocks",
    )
    parser.add_argument(
        "--num-res-layers",
        type=int,
        default=3,
        help="Number of residual layers in the model",
    )
    parser.add_argument(
        "--downsample-parameters",
        type=int,
        nargs=4,
        default=(2, 4, 1, 1),
        help="Parameters for the downsampling layers",
    )
    parser.add_argument(
        "--upsample-parameters",
        type=int,
        nargs=5,
        default=(2, 4, 1, 1, 0),
        help="Parameters for the upsampling layers",
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        default=256,
        help="Number of embeddings in the VQ-VAE",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=632,
        help="Dimension of the embeddings in the VQ-VAE",
    )

    parser.add_argument(
        "--model-path-pix2pix",
        type=str,
        required=True,
        help="Path to the InstructPix2Pix model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Put a lung nodule in the masked region.",
        help="Prompt for the model",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Path to the images folder"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--image-guidance-scale", type=float, default=2.5, help="Image guidance scale"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Guidance scale"
    )

    parser.add_argument("--output-dir", type=str, help="Path to the output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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
    circle_masks = []
    binarized_masks = []

    labeled_mask, _ = label(binary_mask_np)
    slices = find_objects(labeled_mask)

    for label_id, s in enumerate(slices, start=1):
        if s is not None:
            min_row, max_row = s[0].start, s[0].stop
            min_col, max_col = s[1].start, s[1].stop

            # Get the center of the bounding box
            center_row = (min_row + max_row) // 2
            center_col = (min_col + max_col) // 2
            radius = max(max_row - min_row, max_col - min_col) // 2
            nodules_info.append((center_row, center_col, radius))

            # Create a circular mask
            result_mask_np = np.zeros_like(binary_mask_np, dtype=bool)
            y, x = np.ogrid[: binary_mask_np.shape[0], : binary_mask_np.shape[1]]
            circle = (x - center_col) ** 2 + (y - center_row) ** 2 <= (
                radius + 3
            ) ** 2  # Add a margin of 3 pixels
            result_mask_np[circle] = True
            circle_masks.append(result_mask_np)

            # Create a binarized mask for this component
            binarized_mask_np = labeled_mask == label_id
            binarized_masks.append(binarized_mask_np)

    # Convert the result back to a torch tensor with same dtype and device as the input
    # Add the channel dimension back
    circle_masks = [
        torch.tensor(
            mask, dtype=binary_mask.dtype, device=binary_mask.device
        ).unsqueeze(0)
        for mask in circle_masks
    ]
    binarized_masks = [
        torch.tensor(
            mask, dtype=binary_mask.dtype, device=binary_mask.device
        ).unsqueeze(0)
        for mask in binarized_masks
    ]

    return circle_masks, binarized_masks, nodules_info


def main(args):

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    state_dict = torch.load(args.model_path_vqvae, map_location=device)
    model.to(device).load_state_dict(state_dict)
    model.eval()

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_path_pix2pix, torch_dtype=torch.float16
    ).to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    # lungmask_inferer = LMInferer(modelname='LTRCLobes', fillmodel='R231')

    data_dir = Path(args.data_dir)
    image_paths = sorted(
        glob(str(data_dir / "*data*" / "*.nii.gz")),
        key=lambda x: Path(x).stem,
    )
    mask_paths = sorted(
        glob(str(data_dir / "*masks" / "*.nii.gz")),
        key=lambda x: Path(x).stem,
    )
    data_paths = [
        {"image": image_path, "mask": mask_path}
        for image_path, mask_path in zip(image_paths, mask_paths)
    ]

    # Define the transforms
    transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1300, a_max=200, b_min=-1.0, b_max=1.0
            ),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(0.688, 0.688, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(512, 512, -1)),
            FilterSlicesByMaskFuncd(
                keys=["image", "mask"],
                mask_key="mask",
                mask_filter_func=lambda x: (x > 0).sum(dim=(0, 1, 2)) > 8,
                slice_dim=3,
            ),
            ToTensord(keys=["image", "mask"]),
        ]
    )

    for data_path in data_paths:

        data = transforms(data_path)
        if data is None:
            continue

        for idx, slice_data in enumerate(data):

            image = slice_data["image"]
            mask = slice_data["mask"]

            # Create circles around masked areas
            circular_nodule_mask, nodule_mask, _ = mask_with_circle(mask)

            # Skip when more than one nodule is present
            if len(circular_nodule_mask) > 1:
                continue
            circular_nodule_mask = circular_nodule_mask[0]
            nodule_mask = nodule_mask[0]

            # Send the data to the device
            image = image.unsqueeze(0).to(device)
            nodule_mask = nodule_mask.unsqueeze(0).to(device)
            circular_nodule_mask = circular_nodule_mask.unsqueeze(0).to(device)

            # Combine image and mask as input
            masked_image = image.clone()
            masked_image = masked_image * torch.abs(
                circular_nodule_mask - 1
            )  # Apply inversed mask to image
            input_image = torch.cat(
                [masked_image, circular_nodule_mask], dim=1
            )  # Concatenate along the channel dimension

            # Generate reconstructed image
            recon_image, _ = model(input_image)

            # Get bounding box of the mask
            smooth_mask = circular_nodule_mask.clone()

            # Apply convolutional filter to mask to create a smooth transition
            kernel = (torch.ones((7, 7)) * (
                1 / 49
            )).unsqueeze(0).unsqueeze(0).to(smooth_mask.device)  # create_circular_average_kernel(7, 3).to(device)
            smooth_mask = torch.nn.functional.conv2d(smooth_mask, kernel, padding=3)

            # Cut and paste the reconstructed image within the mask region back to the original image
            combined_image = image * (smooth_mask - 1) + recon_image * smooth_mask

            image_array = (
                image.squeeze(0)
                .permute(2, 1, 0)
                .flip(dims=(0, 1))
                .detach()
                .cpu()
                .numpy()
                .repeat(3, axis=-1)
            )
            # mask_array = (
            #     nodule_mask.squeeze(0)
            #     .permute(2, 1, 0)
            #     .flip(dims=(0, 1))
            #     .detach()
            #     .cpu()
            #     .numpy()
            #     .repeat(3, axis=-1)
            # )
            round_mask_array = (
                circular_nodule_mask.squeeze(0)
                .permute(2, 1, 0)
                .flip(dims=(0, 1))
                .detach()
                .cpu()
                .numpy()
                .repeat(3, axis=-1)
            )
            # recon_array = (
            #     recon_image.squeeze(0)
            #     .permute(2, 1, 0)
            #     .flip(dims=(0, 1))
            #     .detach()
            #     .cpu()
            #     .numpy()
            #     .repeat(3, axis=-1)
            # )
            combined_array = (
                combined_image.squeeze(0)
                .permute(2, 1, 0)
                .flip(dims=(0, 1))
                .detach()
                .cpu()
                .numpy()
                .repeat(3, axis=-1)
            )

            # Forward image through the diffusion model
            edited_image_array = pipeline(
                args.prompt,
                image=combined_array,
                num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0].numpy()

            # retrieve bounding boxes of nodule
            x, y, w, h = cv2.boundingRect(round_mask_array[..., 0].astype(np.uint8))
            cv2.rectangle(
                image_array, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 1
            )
            cv2.rectangle(
                edited_image_array,
                (x - 5, y - 5),
                (x + w + 5, y + h + 5),
                (0, 255, 0),
                1,
            )

            image_pil = Image.fromarray(image_array)
            edited_image_pil = Image.fromarray(edited_image_array)

            save_dir = Path(data_path["image"]).relative_to(data_dir).parent
            save_dir = Path(args.output_dir) / save_dir

            image_pil.save(save_dir / f"image_slice={idx}.png")
            edited_image_pil.save(save_dir / f"edited_image_slice={idx}.png")


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
