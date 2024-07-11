import os
import json
import random
import argparse


PROMPTS = [
    "Put a pulmonary nodule in the red squares",
    "Insert a lung nodule within each red square.",
    "Place pulmonary nodules in the marked red areas.",
    "Position nodules inside the red squares.",
    "Embed nodules into the specified red squares.",
    "Integrate pulmonary nodules into the red squares.",
    "Populate red squares with pulmonary nodules.",
    "Affix pulmonary nodules to the red square regions.",
    "Situate nodules within the designated red areas.",
    "Deposit lung nodules within each red square.",
    "Arrange pulmonary nodules within the red squares.",
    "Locate nodules precisely in the red square locations.",
    "Embed nodules in the red square zones.",
    "Install pulmonary nodules in the red squares.",
    "Place a lung nodule in each red square.",
    "Position a nodule in every red square.",
    "Insert nodules within the outlined red squares.",
    "Affix nodules in the specified red squares.",
    "Distribute nodules among the red square areas.",
    "Embed a nodule in each marked red square.",
    "Introduce lung nodules into the red square regions.",
    "Set pulmonary nodules inside the outlined red squares.",
    "Position nodules accurately within the red squares.",
    "Implant nodules into the designated red areas.",
    "Place a pulmonary nodule in each of the red squares.",
    "Populate the red squares with lung nodules.",
    "Place nodules in the designated red square locations.",
    "Integrate nodules into the marked red areas.",
    "Embed nodules within each red square.",
    "Place lung nodules within the marked red squares.",
    "Situate nodules within each specified red square.",
    "Embed nodules in the red square outlines.",
    "Insert pulmonary nodules into the red squares.",
    "Position nodules in the designated red square zones.",
    "Install a nodule in each outlined red square.",
    "Affix pulmonary nodules within the red square boundaries.",
    "Deposit nodules into the red square regions.",
    "Place a pulmonary nodule within each red square.",
    "Populate each red square with lung nodules.",
    "Integrate nodules into the red square outlines.",
    "Embed nodules in the red square spaces.",
    "Position nodules within the red square markings.",
    "Situate nodules in the designated red square areas.",
    "Install nodules within the red squares as marked.",
    "Place lung nodules within the red square boundaries.",
    "Integrate pulmonary nodules into the red square regions.",
    "Affix nodules within each red square.",
    "Position a nodule within each designated red square.",
    "Embed nodules in the marked red square locations.",
    "Populate the red square zones with pulmonary nodules.",
    "Set lung nodules in the outlined red square areas.",
    "Place nodules in the red square spaces.",
    "Introduce nodules into the red square markings.",
    "Embed nodules in the red square areas.",
    "Install nodules within the red square outlines.",
    "Place a nodule in each red square.",
    "Position nodules in the red square regions.",
    "Affix nodules within the red square boundaries.",
    "Deposit nodules in the red square locations.",
    "Populate the red squares with nodules.",
]


def main(args):

    assert os.path.exists(args.dataset_dir), "Dataset directory does not exist"
    assert os.path.exists(os.path.join(args.dataset_dir, "original_images")), "Original images directory does not exist"
    assert os.path.exists(os.path.join(args.dataset_dir, "edited_images")), "Edited images directory does not exist"

    images = os.listdir(args.dataset_dir, "original_images")
    entries = []
    for image in images:
        entries.append({
            "prompt": random.choice(PROMPTS),
            "oriningal_image": os.path.join(args.dataset_dir, "original_images", image),
            "edited_image": os.path.join(args.dataset_dir, "edited_images", image),
        })

    with open(os.path.join(args.dataset_dir, "train.jsonl"), "w") as f:
        for entry in entries:
            f.write(json.dumps(entry))
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    args = parser.parse_args()

    main(args)
