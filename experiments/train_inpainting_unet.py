import os
import argparse
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import monai.data as data
import monai.transforms as transforms
from monai.networks.nets import UNet
from monai.utils import set_determinism
from accelerate import Accelerator
from accelerate.utils import set_seed

from magicnod.transforms import FilterSlicesByMaskFuncd


class DebugLoadImagedWrapper(transforms.LoadImaged):
    def __call__(self, data):
        try:
            return super().__call__(data)
        except Exception as e:
            print(f"Error loading image: {data['image']}")
            raise e


def get_args_parser():
    """
    Get the argument parser for the script.

    Returns:
    - argparse.ArgumentParser: Argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Train UNet model for inpainting of axial CT slices")
    
    # Experiment and data configuration
    parser.add_argument("--experiment-name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--data-dir", required=True, type=str, help="Path to the data directory")

    # Model and training configuration
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256, 512], 
                        help="Number of channels in each layer")
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 2, 2, 2], help="Strides in each layer")
    parser.add_argument("--num-res-units", type=int, default=2, help="Number of residual units in the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loading")

    # Misc
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    
    return parser


def main(args):

    # Initialize accelerator for distributed training
    accelerator = Accelerator(log_with="wandb" if args.wandb else None)
    if args.wandb:
        accelerator.init_trackers(
            project_name="inpainting-unet",
            config=vars(args),
            init_kwargs={"name": args.experiment_name}
        )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    set_determinism(args.seed)

    # Create the checkpoint directory
    checkpoint_dir = Path("checkpoints") / args.experiment_name
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the data
    train_paths = [{"image": p, "mask": p.replace("image", "mask")} for p in glob(
        os.path.join(args.data_dir, "train", "**", "image.nii.gz"), recursive=True)]
    valid_paths = [{"image": p, "mask": p.replace("image", "mask")} for p in glob(
        os.path.join(args.data_dir, "valid", "**", "image.nii.gz"), recursive=True)]
    
    # Define the transforms
    train_transforms = transforms.Compose([
        DebugLoadImagedWrapper(keys=["image", "mask"]),
        transforms.EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        transforms.Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
        transforms.ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(384, 384, 128)),
        transforms.ToTensord(keys=["image", "mask"]),
        FilterSlicesByMaskFuncd(
            keys=["image", "mask"],
            mask_key="mask",
            mask_filter_func=lambda mask: mask.sum(dim=(0, 1, 2)) == 0,  # Keep slices without mask
            slice_dim=3,
        ),
    ])

    valid_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image", "mask"]),
        transforms.EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        transforms.Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
        transforms.ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(384, 384, 128)),
        transforms.ToTensord(keys=["image", "mask"]),
        FilterSlicesByMaskFuncd(
            keys=["image", "mask"],
            mask_key="mask",
            mask_filter_func=lambda mask: mask.sum(dim=(0, 1, 2)) == 0,  # Keep slices without mask
            slice_dim=3,
        ),
    ])

    # Create the data loaders
    train_data = data.Dataset(data=train_paths, transform=train_transforms)
    train_data = data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers)
    val_data = data.Dataset(data=valid_paths, transform=valid_transforms)
    val_data = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize the model
    model = UNet(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=1, 
        channels=args.channels, 
        strides=args.strides, 
        num_res_units=args.num_res_units
    )
    
    # Initialize the optimizer and lr scheduler
    lr = args.lr * (args.batch_size * accelerator.num_processes / 32)  # scale the learning rate
    num_training_steps = len(train_data) * args.epochs // accelerator.num_processes

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Initialize the l2 loss function
    loss_function = nn.MSELoss()

    # Prepare the model for training
    model, optimizer, train_data, val_data = accelerator.prepare(
        model, optimizer, train_data, val_data
    )

    # Training loop
    global_step = 0
    min_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_vars = {
            "total_loss": torch.tensor([0.0], device=accelerator.device), 
            "num_batches": torch.tensor([0], device=accelerator.device)
        }

        # Training epoch
        for i, batch in enumerate(train_data):

            # Mask part of the input using CutOut with a rectangle of 16-128 x 16-128 pixels
            random_mask = torch.ones_like(batch["image"])
            mask_size = torch.randint(16, 128, (2,))
            x = torch.randint(0, 384 - mask_size[0], (1,))
            y = torch.randint(0, 384 - mask_size[1], (1,))
            random_mask[:, :, x:x + mask_size[0], y:y + mask_size[1]] = 0
            batch["masked"] = batch["image"] * random_mask

            # Forward pass
            optimizer.zero_grad()
            pred = model(batch["masked"])

            # Compute loss
            loss = loss_function(pred, batch["image"])

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()

            # Log the loss, lr and epoch
            logs = {
                "loss / step": loss.item(), 
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
            accelerator.log(logs, step=global_step)

            # Update epoch variables
            epoch_vars["total_loss"] += loss.item()
            epoch_vars["num_batches"] += 1
            global_step += 1

            # Update the lr scheduler
            lr_scheduler.step()
        
        # Log the average loss for the epoch
        accelerator.wait_for_everyone()
        epoch_vars = {k: accelerator.reduce(v, reduction="sum") for k, v in epoch_vars.items()}
        accelerator.log({
            "loss / epoch": epoch_vars["total_loss"] / epoch_vars["num_batches"]
        }, step=global_step - 1)

        # Validation epoch
        model.eval()
        epoch_vars = {
            "total_loss": torch.tensor([0.0], device=accelerator.device), 
            "num_batches": torch.tensor([0], device=accelerator.device)
        }
        for i, batch in enumerate(val_data):

            # Forward pass
            with torch.no_grad():
                pred = model(batch["masked"])

            # Compute loss
            loss = loss_function(pred, batch["image"])

            # Update epoch variables
            epoch_vars["total_loss"] += loss.item()
            epoch_vars["num_batches"] += 1

        # Log the average loss for the epoch
        accelerator.wait_for_everyone()
        epoch_vars = {k: accelerator.reduce(v, reduction="sum") for k, v in epoch_vars.items()}
        val_loss = epoch_vars["total_loss"] / epoch_vars["num_batches"]
        accelerator.log({
            "val_loss": val_loss,
        }, step=global_step - 1)

        # Save the model checkpoint if the validation loss is lower
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            accelerator.save(model.state_dict(), checkpoint_dir / "best_model.pth")
    
    # Save the final model
    accelerator.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    accelerator.end_training()

    
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
