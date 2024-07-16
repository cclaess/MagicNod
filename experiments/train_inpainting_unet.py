import os
import argparse
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train an inpainting model on the LIDC dataset')

    parser.add_argument('--data_dir', type=str, help='Path to the preptrained LIDC data directory')
    parser.add_argument('--out_dir', type=str, help='Output directory for the model weights')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser


class LIDCInpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.input_files = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomMasked',  '*.png')))
        self.output_files = sorted(glob(os.path.join(data_dir, '*', '*', 'RandomScan',  '*.png')))

    def __len__(self):
        return len(self.image_files)

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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = LIDCInpaintingDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet2DModel(in_channels=3, out_channels=3)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
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

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'inpainting_model.pth'))


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)