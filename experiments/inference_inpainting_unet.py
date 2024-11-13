import os
import argparse
from pathlib import Path
from glob import glob

import cv2
import torch
import numpy as np
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityRanged, 
    Orientationd, 
    Spacingd, 
    ResizeWithPadOrCropd, 
    ToTensord,
)

from magicnod.transforms import FilterSlicesByMaskFuncd


def get_args_parser():
    parser = argparse.ArgumentParser(description="Inference script for UNet model on CT images")
    parser.add_argument("--model-path", required=True, type=str, help="Path to the saved best model checkpoint")
    parser.add_argument("--data-dir", required=True, type=str, help="Directory with input images")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save output images")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256, 512], 
                        help="Number of channels in each layer")
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 2, 2, 2], help="Strides in each layer")
    parser.add_argument("--num-res-units", type=int, default=2, help="Number of residual units in the model")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    return parser


def load_model(model_path: str, channels: list, strides: list, num_res_units: int):
    model = UNet(
        spatial_dims=2, 
        in_channels=3, 
        out_channels=1, 
        channels=channels, 
        strides=strides, 
        num_res_units=num_res_units
    )

    state_dict = torch.load(model_path)
    state_dict = {k.replace("module.model.", "model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    return model


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = load_model(args.model_path, args.channels, args.strides, args.num_res_units).to(device)
    model.eval()

    # Define the transforms for data loading
    transforms = transforms.Compose([
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
            mask_filter_func=lambda mask: mask.sum(dim=(0, 1, 2)) == 0,  # Keep slices without mask
            slice_dim=3,
        ),
    ])

    # Load the data
    data_paths = [{"image": p, "mask": p.replace("image", "mask")} for p in glob(
        os.path.join(args.data_dir, "valid", "**", "image.nii.gz"), recursive=True)]
    
    dataset = Dataset(data=data_paths, transform=transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run inference
    for batch in loader:
        
        # Mask part of the input using CutOut with a rectangle of 32 x 32 pixels
        mask = torch.ones_like(batch["image"])
        mask[:, :, 176:208, 176:208] = 0
        batch["masked"] = batch["image"] * mask

        # Forward pass
        with torch.no_grad():
            output = model(batch["masked"].to(device))

        # Save the outputs as PNG images of a grid of the original, masked and inpainted slices
        for i in range(output.shape[0]):
            image = batch["image"][i].cpu().numpy()
            masked = batch["masked"][i].cpu().numpy()
            inpainted = output[i].cpu().numpy()

            image = (image * 255).astype("uint8")
            masked = (masked * 255).astype("uint8")
            inpainted = (inpainted * 255).astype("uint8")

            # Create a grid of images
            grid = np.concatenate([image, masked, inpainted], axis=1)
            grid_path = os.path.join(args.output_dir, f"output_{i}.png")
            cv2.imwrite(grid_path, grid)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
