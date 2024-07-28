import os
import random
import argparse
import zipfile
from io import BytesIO
from glob import glob

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from monai.networks.nets import UNet
from monai.losses import SSIMLoss
from tqdm import tqdm

import utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train an inpainting model on the LIDC dataset')

    parser.add_argument('--data_dir', type=str, help='Path to the preptrained LIDC data directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for the model weights')
    parser.add_argument('--experiment', type=str, default='experiment', help='Name of the experiment')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ssim'], help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'], 
                        help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'step', 'plateau', 'cosine'], 
                        help='Learning rate scheduler')

    # misc
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for the dataloader')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training. Please do not set this manually.')

    return parser


def main(args):
    # initialize wandb, random seeds, and reproducibility
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(seed=args.seed)
    torch.backends.cudnn.benchmark = True
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    if utils.is_main_process():
        wandb.init(project='lidc-inpainting', name=args.experiment, config=args)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_dataset = LIDCInpaintingDataset(
        args.data_dir, 
        split="train", 
        val_split=args.val_split, 
        transform=transform, 
        seed=args.seed
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = LIDCInpaintingDataset(
        args.data_dir,
        split="val",
        val_split=args.val_split,
        transform=transform,
        seed=args.seed
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset, 
        sampler=val_sampler,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    model = UNet(spatial_dims=2, in_channels=3, out_channels=1, 
                 channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2)
    model.cuda()

    # synchoronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = nn.MSELoss().cuda() if args.loss == 'mse' else SSIMLoss(spatial_dims=2).cuda()

    params_groups = utils.get_params_groups(model)
    optimizer = {
        'sgd': lambda: optim.SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
        'adam': lambda: optim.Adam(params_groups, lr=args.lr, weight_decay=args.weight_decay),
        'adamw': lambda: optim.AdamW(params_groups, lr=args.lr, weight_decay=args.weight_decay)
    }[args.optimizer]()
    
    num_steps = len(train_dataloader) * args.epochs
    scheduler = {
        'step': lambda: optim.lr_scheduler.StepLR(optimizer, step_size=num_steps // 5, gamma=0.1),
        'plateau': lambda: optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
        'cosine': lambda: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0)
    }.get(args.scheduler, lambda: None)()

    to_restore = {'epoch': 0, 'global_step': 0, 'best_val_loss': float('inf')}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, 'model.pth'), 
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        run_variables=to_restore,
    )
    start_epoch = to_restore['epoch']
    global_step = to_restore['global_step']
    best_val_loss = to_restore['best_val_loss']

    for epoch in range(start_epoch, args.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, optimizer, scheduler, train_dataloader, epoch, global_step, args)
        wandb.log({f'Training / {k} / Epoch': v for k, v in train_stats.items()}, step=global_step)

        val_stats = evaluate(model, criterion, val_dataloader, epoch, global_step, args)
        wandb.log({f'Validation / {k} / Epoch': v for k, v in val_stats.items()}, step=global_step)

        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch + 1,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'args': args
        }

        if utils.is_main_process():
            if val_stats['Loss'] < best_val_loss:
                best_val_loss = val_stats['Loss']
                save_dict['best_val_loss'] = best_val_loss

                utils.save_on_master(
                    save_dict, 
                    os.path.join(args.output_dir, 'model-best.pth')
                )
            
            utils.save_on_master(
                save_dict, 
                os.path.join(args.output_dir, 'model.pth')
            )


def train_one_epoch(model, criterion, optimizer, scheduler, dataloader, epoch, global_step, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="\t")
    header = 'Epoch: [{}/{}]'.format(epoch + 1, args.epochs)

    for input_img, output_img in metric_logger.log_every(dataloader, 10, header):
        input_img, output_img = input_img.cuda(), output_img.cuda()

        optimizer.zero_grad()

        pred_img = model(input_img)
        loss = criterion(pred_img, output_img)

        loss.backward()
        optimizer.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(LR=optimizer.param_groups[0]['lr'])
        if utils.is_main_process():
            wandb.log({
                'Training / Loss / Step': loss.item(), 
                'Training / LR / Step': optimizer.param_groups[0]['lr']
            }, step=global_step)
        
        global_step += 1
        scheduler.step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, criterion, dataloader, epoch, global_step, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="\t")
    header = 'Validation Epoch: [{}/{}]'.format(epoch + 1, args.epochs)

    wandb_images = []
    with torch.no_grad():
        for i, (input_img, output_img) in enumerate(metric_logger.log_every(dataloader, 10, header)):
            input_img, output_img = input_img.cuda(), output_img.cuda()

            pred_img = model(input_img)
            loss = criterion(pred_img, output_img)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())

            if utils.is_main_process and i < 3:
                wandb_images.append(wandb.Image(
                    input_img[0].cpu().detach().numpy().transpose(1, 2, 0), 
                    caption="Input Image"
                ))
                wandb_images.append(wandb.Image(
                    output_img[0].cpu().detach().numpy().transpose(1, 2, 0), 
                    caption="Output Image"
                ))
                wandb_images.append(wandb.Image(
                    pred_img[0].cpu().detach().numpy().transpose(1, 2, 0), 
                    caption="Predicted Image"
                ))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if utils.is_main_process():
        wandb.log({'Validation / Example Images': wandb_images}, step=global_step)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LIDCInpaintingDataset(Dataset):
    def __init__(self, data_dir, split="train", val_split=0.2, transform=None, seed=42):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        assert split in ['train', 'val'], 'Split must be either "train" or "val"'
        assert os.path.exists(data_dir), f'Data directory {data_dir} does not exist'

        input_zips = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomMasked.zip')))
        output_zips = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomScan.zip')))

        # extract the zip files
        input_files = []
        output_files = []
        for input_zip_path, output_zip_path in zip(input_zips, output_zips):
            with zipfile.ZipFile(input_zip_path, 'r') as input_zip, zipfile.ZipFile(output_zip_path, 'r') as output_zip:
                input_names = [f for f in input_zip.namelist() if f.endswith('.png')]
                output_names = [f for f in output_zip.namelist() if f.endswith('.png')]
                input_files.extend([(input_zip_path, name) for name in input_names])
                output_files.extend([(output_zip_path, name) for name in output_names])

        # make split based on patient id
        patient_ids = [f[0].split(os.sep)[-3] for f in input_files]
        unique_ids = list(set(patient_ids))
        
        # shuffle the patient ids
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

        input_img = Image.open(BytesIO(input_data))
        output_img = Image.open(BytesIO(output_data))

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        return input_img, output_img



if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)