import os
import argparse
from glob import glob

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

    # training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ssim'], help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'], 
                        help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'step', 'plateau', 'cosine'], 
                        help='Learning rate scheduler')

    # misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser


class LIDCInpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        print("Input dir: ", os.path.join(data_dir, '*', '*', 'RandomMasked',  '*.png'))

        self.input_files = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomMasked',  '*.png')))
        self.output_files = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomScan',  '*.png')))

        print("Num input files: ", len(self.input_files))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_files[idx])
        output_img = Image.open(self.output_files[idx])

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        return input_img, output_img


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    dataset = LIDCInpaintingDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet(spatial_dims=2, in_channels=3, out_channels=3, channels=(32, 64, 128, 256, 512))
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


    for epoch in range(args.epochs):
        model.train()

        for i, (input_img, output_img) in enumerate(dataloader):
            input_img, output_img = input_img.to(device), output_img.to(device)

            optimizer.zero_grad()

            pred_img = model(input_img)
            loss = criterion(pred_img, output_img)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
            
            if scheduler:
                if args.scheduler == 'plateau':
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'inpainting_model.pth'))


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)