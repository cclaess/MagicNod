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


def get_args_parser():

    parser = argparse.ArgumentParser(description='Inferring with InstructPix2Pix')

    parser.add_argument('--model_path', type=str, help='Path to the model')

    parser.add_argument('--prompt', type=str, help='Prompt for the model')
    parser.add_argument('--data_path', type=str, help='Path to the images folder')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--image_guidance_scale', type=float, default=2.5, help='Image guidance scale')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')

    parser.add_argument('--output_path', type=str, help='Path to the output folder')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')


    return parser


def main(args):

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")
    generator = torch.Generator("cuda").manual_seed(args.seed)

    image_paths = sorted(glob(os.path.join(args.data_path, '**', 'Masked', '*.png'), recursive=True))
    mask_paths = sorted(glob(os.path.join(args.data_path, '**', 'Mask', '*.png'), recursive=True))

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
        mask_array = np.array(mask)
        _, binary_mask = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY)
        num_labels, labels_im = cv2.connectedComponents(binary_mask)
        bboxes = []
        for i in range(1, num_labels):
            component_mask = (labels_im == i).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(component_mask)
            bboxes.append((x, y, w, h))
        
        # draw bounding boxes on the edited image
        edited_image = np.array(edited_image)
        for x, y, w, h in bboxes:
            cv2.rectangle(edited_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        edited_image = Image.fromarray(edited_image)

        save_path = Path(image_path.replace(args.data_path, args.output_path))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        edited_image.save(save_path)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
