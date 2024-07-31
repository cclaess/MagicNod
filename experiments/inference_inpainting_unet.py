import os
import zipfile
import argparse
from glob import glob
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from monai.networks.nets import UNet


def get_args_parser():
    parser = argparse.ArgumentParser(description="Run inference on a trained UNet model")
    parser.add_argument("--model", required=True, type=str, help="Path to the trained model")
    parser.add_argument("--input", required=True, type=str, help="Path to the input images")
    parser.add_argument("--output", required=True, type=str, help="Path to the output images")
    return parser


def main(args):

    model = UNet(
        spatial_dims=2, 
        in_channels=3, 
        out_channels=1, 
        channels=(32, 64, 128, 256, 512), 
        strides=(2, 2, 2, 2), 
        num_res_units=2
    )

    ckpt = torch.load(args.model)
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("model.model.", "model."): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
    
    # process all images in the input directory and save in similar folder structure in output directory
    zip_files = glob(os.path.join(args.input, '**', 'NoduleMasked.zip'), recursive=True)
    print("number of zip files: ", len(zip_files))
    image_files = []
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip:
                names = [f for f in zip.namelist() if f.endswith('.png')]
                image_files.extend([(zip_file, name) for name in names])

    print("number of image files: ", len(image_files))
    for zip_file, image_file in image_files:
        with zipfile.ZipFile(zip_file, 'r') as zip:
            data = zip.read(image_file)
        
        # read the image and convert to tensor
        img = Image.open(BytesIO(data))
        img = ToTensor()(img).unsqueeze(0).float() / 255.0

        if torch.cuda.is_available():
            img = img.cuda()

        # run inference
        with torch.no_grad():
            out = model(img).detach().cpu().squeeze(0) * 255.0
        
        # overwrite the original image with the inpainted image only for the region where the input is red
        img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = out.numpy().transpose(1, 2, 0)

        # convert out to 3 channel image
        out = np.stack([out] * 3, axis=-1)

        print(img.shape, out.shape)
        out[img[0, :, :] != 1] = img[img[0, :, :] != 1]

        # save the output image in same directror structure as input
        output_path = os.path.join(args.output, os.path.relpath(zip_file, args.input).replace('.zip', ''), image_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_image(out, output_path)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
