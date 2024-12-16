import os
import argparse
from glob import glob
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
# from lungmask import LMInferer


def get_args_parser():

    parser = argparse.ArgumentParser(description='Inferring with InstructPix2Pix')

    parser.add_argument('--model-path', type=str, help='Path to the model')

    parser.add_argument('--prompt', type=str, help='Prompt for the model')
    parser.add_argument('--data-dir', type=str, help='Path to the images folder')
    parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--image-guidance-scale', type=float, default=2.5, help='Image guidance scale')
    parser.add_argument('--guidance-scale', type=float, default=7.5, help='Guidance scale')

    parser.add_argument('--output-dir', type=str, help='Path to the output folder')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')


    return parser


def main(args):

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")
    generator = torch.Generator("cuda").manual_seed(args.seed)

    data_dir = Path(args.data_dir)

    image_paths = sorted(glob(
        str(data_dir / "valid" / "**" / "combined_mask_slice=*_nod=*.png"),
        recursive=True,
    ))
    mask_paths = sorted(glob(
        str(data_dir / "valid" / "**" / "mask_slice=*_nod=*.png"),
        recursive=True,
    ))

    for image_path, mask_path in zip(image_paths, mask_paths):

        image = load_image(image_path)
        mask = load_image(mask_path)

        edited_image = pipeline(
        args.prompt,
        image=image,
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
        ).images[0]

        # retreive bounding boxes of the nodules using the mask where mask > 0
        mask_array = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(binary_mask)

        # Change bbox to a square with the same middle point
        max_side = max(w, h)
        x = x + w // 2 - max_side // 2
        y = y + h // 2 - max_side // 2
        w = h = max_side
        
        # draw bounding boxes on the edited image
        edited_image = np.array(edited_image)
        cv2.rectangle(edited_image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 1)
        edited_image = Image.fromarray(edited_image)

        save_path = Path(image_path.replace(args.data_dir, args.output_dir))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving edited image to {save_path}")
        edited_image.save(save_path)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
