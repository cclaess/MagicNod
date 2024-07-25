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


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train an inpainting model on the LIDC dataset')

    parser.add_argument('--data_dir', type=str, help='Path to the preptrained LIDC data directory')
    parser.add_argument('--out_dir', type=str, help='Output directory for the model weights')
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
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

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


def main(args):
    # initialize wandb
    wandb.init(project='lidc-inpainting', name=args.experiment, config=args)

    # set random seeds and determinism for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = LIDCInpaintingDataset(
        args.data_dir,
        split="val",
        val_split=args.val_split,
        transform=transform,
        seed=args.seed
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(spatial_dims=2, in_channels=3, out_channels=1, 
                 channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2)
    model.to(device)

    criterion = nn.MSELoss() if args.loss == 'mse' else SSIMLoss(spatial_dims=2)
    optimizer = {
        'sgd': lambda: optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
        'adam': lambda: optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        'adamw': lambda: optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    }[args.optimizer]()
    
    scheduler = {
        'step': lambda: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        'plateau': lambda: optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
        'cosine': lambda: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    }.get(args.scheduler, lambda: None)()

    best_val_loss = float('inf')
    save_path = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for i, (input_img, output_img) in enumerate(train_dataloader):
            input_img, output_img = input_img.to(device), output_img.to(device)

            optimizer.zero_grad()

            pred_img = model(input_img)
            loss = criterion(pred_img, output_img)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
            
            if scheduler:
                if args.scheduler == 'plateau':
                    scheduler.step(running_loss / len(train_dataloader))
                else:
                    scheduler.step()
        
        avg_train_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch}, Training Loss: {avg_train_loss}')
        wandb.log({'Training Loss': avg_train_loss})
        
        model.eval()
        val_loss = 0.0
        val_images = []

        with torch.no_grad():
            for i, (input_img, output_img) in enumerate(val_dataloader):
                input_img, output_img = input_img.to(device), output_img.to(device)

                pred_img = model(input_img)
                loss = criterion(pred_img, output_img)

                val_loss += loss.item()
                val_images.append(wandb.Image(input_img[0].cpu().detach().numpy(), caption="Input Image"))
                val_images.append(wandb.Image(output_img[0].cpu().detach().numpy(), caption="Output Image"))
                val_images.append(wandb.Image(pred_img[0].cpu().detach().numpy(), caption="Predicted Image"))

        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch {epoch}, Validation Loss: {avg_val_loss / len(val_dataloader)}')
        wandb.log({'Validation Loss': avg_val_loss, 'Example Images': val_images})

        # save the model if the validation loss has decreased
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss

            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            
            if save_path:
                os.remove(save_path)

            save_path = os.path.join(args.out_dir, args.experiment, f'model-epoch={epoch}-val_loss={val_loss}.pth')
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)