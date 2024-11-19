import os
import argparse
from glob import glob
from pathlib import Path

import wandb
import torch
from torch.nn import L1Loss, BatchNorm2d, InstanceNorm2d
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
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
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import VQVAE, PatchDiscriminator
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import make_grid


from magicnod.transforms import FilterSlicesByMaskFuncd


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
    parser.add_argument("--num-channels", type=int, nargs="+", default=(256, 512), help="Number of channels in the model")
    parser.add_argument("--num-res-channels", type=int, default=512, help="Number of channels in the residual blocks")
    parser.add_argument("--num-res-layers", type=int, default=2, help="Number of residual layers in the model")
    parser.add_argument("--downsample-parameters", type=int, nargs=4, default=(2, 4, 1, 1), help="Parameters for the downsampling layers")
    parser.add_argument("--upsample-parameters", type=int, nargs=5, default=(2, 4, 1, 1, 0), help="Parameters for the upsampling layers")
    parser.add_argument("--num-embeddings", type=int, default=256, help="Number of embeddings in the VQ-VAE")
    parser.add_argument("--embedding-dim", type=int, default=632, help="Dimension of the embeddings in the VQ-VAE")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--lr-g", type=float, default=1e-4, help="Learning rate for the generator")
    parser.add_argument("--lr-d", type=float, default=5e-4, help="Learning rate for the discriminator")
    parser.add_argument("--perceptual-weight", type=float, default=0.001, help="Weight for the perceptual loss")
    parser.add_argument("--adv-weight", type=float, default=0.01, help="Weight for the adversarial loss")

    # Misc
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loading")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    
    return parser


def replace_bn_with_in(module):
    """
    Replace all BatchNorm2d layers with InstanceNorm2d layers in a module.

    Args:
    - module (torch.nn.Module): Module to replace the BatchNorm2d layers with InstanceNorm2d layers.

    Returns:
    - None
    """
    # Iterate through the children of the module
    for name, child in module.named_children():
        if isinstance(child, BatchNorm2d):
            # Replace BatchNorm2d with InstanceNorm2d
            setattr(module, name, InstanceNorm2d(child.num_features))
        else:
            # Recursively apply to child modules
            replace_bn_with_in(child)

def init_weights(m):
    """
    Initialize the weights of a module using Kaiming normal initialization.

    Args:
    - m (torch.nn.Module): Module to initialize the weights.

    Returns:
    - None
    """
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)


def generate_random_masks(batch_size, image_size, min_size=16, max_size=128):
    """
    Generate random mask coordinates and sizes for a batch of images.
    
    Args:
    - batch_size (int): Number of images in the batch.
    - image_size (int): Size of the images in the batch.
    - min_size (int): Minimum size of the mask.
    - max_size (int): Maximum size of the mask.

    Returns:
    - List(dict): coordinates and sizes of the masks for each image in the batch.
    """
    mask_params = []
    for _ in range(batch_size):
        mask_width, mask_height = torch.randint(min_size, max_size, (2,)).tolist()
        x = torch.randint(0, image_size - mask_width, (1,)).item()
        y = torch.randint(0, image_size - mask_height, (1,)).item()
        mask_params.append({"x": x, "y": y, "width": mask_width, "height": mask_height})
    return mask_params


def main(args):

    # Initialize accelerator for distributed training
    accelerator = Accelerator(log_with="wandb" if args.wandb else None)
    if args.wandb:
        accelerator.init_trackers(
            project_name="inpainting-vqvae",
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
    train_transforms = Compose([
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
            num_slices=16,
        ),
    ])

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
            mask_filter_func=lambda mask: mask.sum(dim=(0, 1, 2)) == 0,  # Keep slices without mask
            slice_dim=3,
            num_slices=16,
        ),
    ])

    # Create the data loaders
    train_data = Dataset(data=train_paths, transform=train_transforms)
    train_data = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers)
    val_data = Dataset(data=valid_paths, transform=valid_transforms)
    val_data = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Initialize the models
    model = VQVAE(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=args.num_channels,
        num_res_channels=args.num_res_channels,
        num_res_layers=args.num_res_layers,
        downsample_parameters=args.downsample_parameters,
        upsample_parameters=args.upsample_parameters,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    )

    discriminator = PatchDiscriminator(
        spatial_dims=2,
        in_channels=1,
        num_layers_d=6,
        num_channels=64,
    )

    # Change all batch norm layers to instance norm layers for the discriminator
    replace_bn_with_in(discriminator)

    # Initialize the weights of the models
    model.apply(init_weights)
    discriminator.apply(init_weights)

    # Define the loss functions
    perceptual_loss = PerceptualLoss(
        spatial_dims=2,
        network_type="alex",
    )
    perceptual_loss.to(accelerator.device)  # move the loss function to the device for distributed training

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    
    # Initialize the optimizers and lr schedulers
    lr_g = args.lr_g * (args.batch_size * 16 * accelerator.num_processes / 256)  # scale the learning rate
    lr_d = args.lr_d * (args.batch_size * 16 * accelerator.num_processes / 256)  # scale the learning rate

    N = len(train_data) * args.epochs // accelerator.num_processes  # number of training steps

    optimizer_g = torch.optim.AdamW(params=model.parameters(), lr=lr_g)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=lr_d)

    # Use a consine annealing learning rate schedule with linear warm-up
    lr_scheduler_g = SequentialLR(
        optimizer_g, 
        [
            LinearLR(optimizer_g, start_factor=0.01, end_factor=1.0, total_iters=N // 10),
            CosineAnnealingLR(optimizer_g, T_max=N - N // 10),
        ],
        milestones=[N // 10],
    )
    lr_scheduler_d = SequentialLR(
        optimizer_d, 
        [
            LinearLR(optimizer_d, start_factor=0.01, end_factor=1.0, total_iters=N // 10),
            CosineAnnealingLR(optimizer_d, T_max=N - N // 10),
        ],
        milestones=[N // 10],
    )
    # lr_scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_g, T_0=N)
    # lr_scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_d, T_0=N)

    # Prepare the models, data and optimizers for distributed training
    model, discriminator, optimizer_g, optimizer_d, train_data, val_data = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_data, val_data
    )

    # Define global training and validation variables
    global_step = 0
    min_val_loss = float("inf")

    # Loop over the epochs
    for epoch in range(args.epochs):

        # Set the models to training mode and reset the epoch variables
        model.train()
        discriminator.train()
        epoch_vars = {
            "epoch_loss": torch.tensor([0.0], device=accelerator.device),
            "gen_epoch_loss": torch.tensor([0.0], device=accelerator.device),
            "disc_epoch_loss": torch.tensor([0.0], device=accelerator.device),
            "num_batches": torch.tensor([0], device=accelerator.device)
        }

        # Training epoch
        for batch in train_data:

            images = batch["image"]
            images.to(accelerator.device)  # explicitly move the data to the device because of the cloning below

            mask_params = generate_random_masks(images.size(0), 384)
            mask = torch.ones_like(images)  # create a mask tensor with ones
            masked_images = images.clone()  # deep-copy to avoid in-place operations

            for j, params in enumerate(mask_params):
                x, y, w, h = params["x"], params["y"], params["width"], params["height"]
                mask[j, :, x:x + w, y:y + h] = 0
                masked_images[j] = masked_images[j] * mask[j]

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)

            reconstruction, quantization_loss = model(masked_images)
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = recons_loss + quantization_loss + args.perceptual_weight * p_loss + args.adv_weight * generator_loss

            accelerator.backward(loss_g)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip the gradients
            optimizer_g.step()

            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = 0.5 * (loss_d_fake + loss_d_real)

            loss_d = args.adv_weight * discriminator_loss

            accelerator.backward(loss_d)
            accelerator.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # clip the gradients
            optimizer_d.step()

            # Log the losses, learning rates and epoch
            logs = {
                "recons_loss / train / step": recons_loss.item(),
                "gen_loss / train / step": generator_loss.item(),
                "disc_loss / train / step": discriminator_loss.item(),
                "lr_g": optimizer_g.param_groups[0]["lr"],
                "lr_d": optimizer_d.param_groups[0]["lr"],
                "epoch": epoch,
            }
            accelerator.log(logs, step=global_step)

            # Update epoch variables
            epoch_vars["epoch_loss"] += recons_loss.item()
            epoch_vars["gen_epoch_loss"] += generator_loss.item()
            epoch_vars["disc_epoch_loss"] += discriminator_loss.item()
            epoch_vars["num_batches"] += 1
            global_step += 1

            # Update the learning rates
            lr_scheduler_g.step()
            lr_scheduler_d.step()
        
        # Log the average losses for the epoch
        accelerator.wait_for_everyone()
        epoch_vars = {k: accelerator.reduce(v, reduction="sum") for k, v in epoch_vars.items()}
        logs = {
            "recons_loss / train / epoch": epoch_vars["epoch_loss"] / epoch_vars["num_batches"],
            "gen_loss / train / epoch": epoch_vars["gen_epoch_loss"] / epoch_vars["num_batches"],
            "disc_loss / train / epoch": epoch_vars["disc_epoch_loss"] / epoch_vars["num_batches"],
        }
        accelerator.log(logs, step=global_step - 1)

        # Set the model to evaluation mode and reset the epoch variables
        model.eval()
        epoch_vars = {
            "epoch_loss": torch.tensor([0.0], device=accelerator.device), 
            "num_batches": torch.tensor([0], device=accelerator.device)
        }

        # Validation epoch
        with torch.no_grad():
            for step, batch in enumerate(val_data):

                images = batch["image"]
                images.to(accelerator.device)  # explicitly move the data to the device because of the cloning below

                mask = torch.ones_like(images)  # create a mask tensor with ones
                mask[:, :, 176:208, 176:208] = 0  # mask a 32 x 32 region in the center of the image
                masked_images = images.clone()  # deep-copy to avoid in-place operations
                masked_images = masked_images * mask

                reconstruction, quantization_loss = model(images=masked_images)

                recons_loss = l1_loss(reconstruction.float(), images.float())

                # Get the first sample from the first batch for visualization
                if step == 0 and accelerator.is_main_process:

                    # Create wandb image with the original image, the masked image and the reconstruction
                    image_grid = make_grid(
                        torch.cat([images[:1], masked_images[:1], reconstruction[:1]]),
                        nrow=3,
                        normalize=True,
                        scale_each=True,
                    )
                    image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
                    accelerator.log({"reconstruction": [wandb.Image(image_grid)]}, step=global_step - 1)
                
                # update epoch variables
                epoch_vars["epoch_loss"] += recons_loss.item()
                epoch_vars["num_batches"] += 1

        # Log the average loss for the epoch
        accelerator.wait_for_everyone()
        epoch_vars = {k: accelerator.reduce(v, reduction="sum") for k, v in epoch_vars.items()}
        val_loss = epoch_vars["epoch_loss"] / epoch_vars["num_batches"]
        accelerator.log({
            "recons_loss / valid / epoch": val_loss,
        }, step=global_step - 1)

        # Save the model checkpoint if the validation loss is lower
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            accelerator.save(model.state_dict(), checkpoint_dir / "best_model.pth")
    
    # Save the final model
    accelerator.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    accelerator.end_training()

    
if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
