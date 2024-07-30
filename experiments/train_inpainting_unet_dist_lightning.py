import os
import random
import argparse
import zipfile
from io import BytesIO
from glob import glob

import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from monai.networks.nets import UNet
from monai.losses import SSIMLoss
from monai import transforms


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train an inpainting model on the LIDC dataset')
    parser.add_argument('--data_dir', type=str, help='Path to the preprocessed LIDC data directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for the model weights')
    parser.add_argument('--experiment', type=str, default='experiment', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'wmse', 'ssim'], help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'], help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'step', 'plateau', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for the dataloader')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--accelerator', type=str, default='cuda', help='Divice accelerator to use')
    return parser


class LIDCInpaintingDataset(Dataset):
    def __init__(self, data_dir, split="train", val_split=0.2, transform=None, seed=42):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        assert split in ['train', 'val'], 'Split must be either "train" or "val"'
        assert os.path.exists(data_dir), f'Data directory {data_dir} does not exist'
        input_zips = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomMasked.zip')))
        output_zips = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomScan.zip')))

        input_files, output_files = [], []
        for input_zip_path, output_zip_path in zip(input_zips, output_zips):
            with zipfile.ZipFile(input_zip_path, 'r') as input_zip, zipfile.ZipFile(output_zip_path, 'r') as output_zip:
                input_names = [f for f in input_zip.namelist() if f.endswith('.png')]
                output_names = [f for f in output_zip.namelist() if f.endswith('.png')]
                input_files.extend([(input_zip_path, name) for name in input_names])
                output_files.extend([(output_zip_path, name) for name in output_names])

        patient_ids = [f[0].split(os.sep)[-3] for f in input_files]
        unique_ids = list(set(patient_ids))
        random.Random(seed).shuffle(unique_ids)
        n_val = int(val_split * len(unique_ids))
        patient_ids_split = unique_ids[:n_val] if split == 'val' else unique_ids[n_val:]
        self.input_files = [f for f, p in zip(input_files, patient_ids) if p in patient_ids_split]
        self.output_files = [f for f, p in zip(output_files, patient_ids) if p in patient_ids_split]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_zip_path, input_name = self.input_files[idx]
        output_zip_path, output_name = self.output_files[idx]

        with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
            input_data = input_zip.read(input_name)
        with zipfile.ZipFile(output_zip_path, 'r') as output_zip:
            output_data = output_zip.read(output_name)

        input_img = np.array(Image.open(BytesIO(input_data)))
        output_img = np.array(Image.open(BytesIO(output_data)))[..., np.newaxis]

        mask = (input_img[..., 0] == 255) & (input_img[..., 1] == 0) & (input_img[..., 2] == 0)
        mask = mask.astype(np.uint8) * 255 - 25
        mask = mask[..., np.newaxis]
        mask += 25

        data = {'input': input_img, 'output': output_img, 'mask': mask}

        if self.transform:
            data = self.transform(data)

        return data
    

class WeightedMSELoss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if input.size() != target.size():
            raise ValueError(f"Target size ({target.size()}) must be the same as input size ({input.size()})")
        if input.size() != mask.size():
            raise ValueError(f"Mask size ({mask.size()}) must be the same as input size ({input.size()})")

        # Compute the squared differences
        loss = (input - target) ** 2

        # Apply the mask
        loss = loss * mask

        # Reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            # Calculate the mean weighted squared error
            sum_weighted_loss = torch.sum(loss)
            # Count of the non-zero elements in the mask
            num_elements = torch.sum(mask != 0).float()
            return sum_weighted_loss / num_elements if num_elements > 0 else 0
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class LIDCInpaintingModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = UNet(spatial_dims=2, in_channels=3, out_channels=1, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2)
        self.criterion = nn.MSELoss() if args.loss == 'mse' else WeightedMSELoss() if args.loss == 'wmse' \
                          else SSIMLoss(spatial_dims=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = {
            'sgd': lambda: torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay),
            'adam': lambda: torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay),
            'adamw': lambda: torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        }[self.args.optimizer]()
        
        scheduler = {
            'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.1 * self.args.epochs), gamma=0.1),
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0)
        }.get(self.args.scheduler, None)
        
        if scheduler:
            return [optimizer], [scheduler]
        return [optimizer]

    def training_step(self, batch, batch_idx):
        input_img, output_img, mask_img = batch
        pred_img = self(input_img)
        loss = self.criterion(pred_img, output_img, mask_img) if self.args.loss == 'wmse' else \
            self.criterion(pred_img, output_img)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_img, output_img, mask_img = batch
        pred_img = self(input_img)
        loss = self.criterion(pred_img, output_img, mask_img) if self.args.loss == 'wmse' else \
            self.criterion(pred_img, output_img)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Logging a few examples to W&B
        if batch_idx < 5:
            input_img = transforms.ScaleIntensityRange(0, 1, 0, 255)(input_img[0])
            output_img = transforms.ScaleIntensityRange(0, 1, 0, 255)(output_img[0])
            pred_img = transforms.ScaleIntensityRange(0, 1, 0, 255)(pred_img[0])

            self.logger.experiment.log({
                'input': wandb.Image(input_img.detach().cpu().permute(1, 2, 0), caption="Input Image"),
                'output': wandb.Image(output_img.detach().cpu().permute(1, 2, 0), caption="Output Image"),
                'prediction': wandb.Image(pred_img.detach().cpu().permute(1, 2, 0), caption="Predicted Image")
            })
        return loss

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensord(keys=['input', 'output', 'mask']),
            transforms.EnsureChannelFirstd(keys=['input', 'output', 'mask'], channel_dim=-1),
            transforms.ScaleIntensityRanged(keys=['input', 'output'], a_min=0, a_max=255, b_min=0., b_max=1., clip=True)
        ])
        train_dataset = LIDCInpaintingDataset(
            self.args.data_dir, 
            split="train", 
            val_split=self.args.val_split, 
            transform=transform, 
            seed=self.args.seed
        )
        return DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensord(keys=['input', 'output', 'mask']),
            transforms.EnsureChannelFirstd(keys=['input', 'output', 'mask'], channel_dim=-1),
            transforms.ScaleIntensityRanged(keys=['input', 'output'], a_min=0, a_max=255, b_min=0., b_max=1., clip=True)
        ])
        val_dataset = LIDCInpaintingDataset(
            self.args.data_dir,
            split="val",
            val_split=self.args.val_split,
            transform=transform,
            seed=self.args.seed
        )
        return DataLoader(val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True)


def main(args):
    seed_everything(args.seed)
    
    wandb_logger = WandbLogger(project='lidc-inpainting', name=args.experiment)
    
    model = LIDCInpaintingModel(args)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    trainer = Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator=args.accelerator,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        sync_batchnorm=True,
    )
    
    trainer.fit(model)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
