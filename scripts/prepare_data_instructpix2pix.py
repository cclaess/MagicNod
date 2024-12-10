import argparse
from glob import glob
from pathlib import Path

import pandas as pd


INSTRUCTIONS = [
    "Place a lung lesion in the masked area.",
    "Insert a pulmonary lesion into the masked zone.",
    "Add a lung lesion to the masked region.",
    "Position a pulmonary lesion in the designated area.",
    "Set a lung lesion in the highlighted zone.",
    "Drop a pulmonary lesion into the specified region.",
    "Embed a lung lesion in the selected mask.",
    "Apply a pulmonary lesion to the marked area.",
    "Locate a lung lesion in the defined region.",
    "Situate a pulmonary lesion in the outlined area.",
    "Introduce a lung lesion into the masked space.",
    "Place a pulmonary lesion in the targeted zone.",
    "Put a lung lesion in the designated mask.",
    "Add a pulmonary lesion into the marked area.",
    "Insert a lung lesion in the specified zone.",
    "Position a pulmonary lesion within the masked space.",
    "Drop a lung lesion into the outlined region.",
    "Embed a pulmonary lesion into the chosen area.",
    "Locate a lung lesion in the highlighted mask.",
    "Apply a pulmonary lesion to the delineated area.",
    "Set a lung lesion in the marked region.",
    "Plant a pulmonary lesion in the selected mask.",
    "Introduce a lung lesion into the specified zone.",
    "Situate a pulmonary lesion within the defined area.",
    "Add a lung lesion to the outlined mask.",
    "Place a pulmonary lesion into the masked zone.",
    "Embed a lung lesion into the targeted area.",
    "Drop a pulmonary lesion in the highlighted region.",
    "Position a lung lesion in the marked zone.",
    "Put a pulmonary lesion in the specified mask.",
    "Apply a lung lesion to the chosen area.",
    "Insert a pulmonary lesion into the masked region.",
    "Set a lung lesion into the defined zone.",
    "Place a pulmonary lesion into the outlined space.",
    "Introduce a lung lesion within the targeted region.",
    "Locate a pulmonary lesion in the chosen mask.",
    "Embed a lung lesion in the delineated zone.",
    "Drop a pulmonary lesion into the marked space.",
    "Add a lung lesion into the chosen region.",
    "Set a pulmonary lesion in the defined mask.",
    "Position a lung lesion into the outlined area.",
    "Insert a pulmonary lesion in the designated zone.",
    "Place a lung lesion into the chosen mask.",
    "Embed a pulmonary lesion in the masked area.",
    "Situate a lung lesion in the targeted mask.",
    "Drop a pulmonary lesion into the specified region.",
    "Add a lung lesion to the defined space.",
    "Apply a pulmonary lesion to the highlighted mask.",
    "Introduce a lung lesion in the marked region.",
    "Position a pulmonary lesion within the chosen area.",
    "Put a lung lesion into the defined mask.",
    "Set a pulmonary lesion into the specified space.",
    "Embed a lung nodule within the outlined zone.",
    "Place a pulmonary nodule in the selected region.",
    "Add a lung nodule into the marked mask.",
    "Locate a pulmonary nodule in the targeted area.",
    "Drop a lung nodule in the highlighted zone.",
    "Apply a pulmonary nodule to the specified mask.",
    "Insert a lung nodule into the chosen space.",
    "Position a pulmonary nodule within the masked region.",
    "Introduce a lung nodule into the defined area.",
    "Set a pulmonary nodule in the marked mask.",
    "Embed a lung nodule in the chosen zone.",
    "Add a pulmonary nodule into the highlighted space.",
    "Place a lung nodule into the defined mask.",
    "Drop a pulmonary nodule within the specified area.",
    "Put a lung nodule in the outlined region.",
    "Locate a pulmonary nodule into the marked space.",
    "Situate a lung nodule in the selected zone.",
    "Embed a pulmonary nodule into the defined area.",
    "Position a lung nodule into the highlighted mask.",
    "Introduce a pulmonary nodule in the specified region.",
    "Apply a lung nodule to the outlined zone.",
    "Drop a pulmonary nodule in the designated mask.",
    "Insert a lung nodule within the chosen region.",
    "Add a pulmonary nodule into the defined space.",
    "Place a lung nodule into the masked mask.",
    "Locate a pulmonary nodule in the chosen zone.",
    "Set a lung nodule in the highlighted area.",
    "Embed a pulmonary nodule into the specified mask.",
    "Situate a lung nodule within the marked zone.",
    "Add a pulmonary nodule in the chosen mask.",
    "Apply a lung nodule to the outlined region.",
    "Introduce a pulmonary nodule into the designated space.",
    "Position a lung nodule within the targeted area.",
    "Put a pulmonary nodule in the masked zone.",
    "Drop a lung nodule into the defined region.",
    "Embed a pulmonary nodule in the highlighted mask.",
    "Locate a lung nodule into the marked zone.",
    "Add a pulmonary nodule in the chosen space.",
    "Apply a lung nodule to the targeted zone.",
    "Set a pulmonary nodule within the masked region.",
    "Position a lung nodule in the highlighted zone.",
    "Introduce a pulmonary nodule in the delineated area.",
    "Embed a lung nodule into the outlined mask.",
    "Place a pulmonary nodule into the defined space.",
    "Drop a lung nodule within the specified mask.",
    "Insert a pulmonary nodule into the chosen region.",
    "Locate a lung nodule in the highlighted area.",
    "Add a pulmonary nodule into the masked zone.",
]


def get_args_parser():
    parser = argparse.ArgumentParser(description="Prepare data for InstructPix2Pix")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="path to the data directory"
    )
    parser.add_argument(
        "--annotations_csv",
        type=str,
        default="data/annotations.csv",
        help="path to the annotations csv file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/instructions.jsonl",
        help="path to the output jsonl file",
    )

    return parser


def main(args):

    # Create the data directory and output directory paths
    data_dir = Path(args.data_dir)
    output_path = Path(args.output_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the annotations csv file with the PatientIDs, NoduleIDs, and corresponding malignancy scores
    annotations = pd.read_csv(args.annotations_csv)

    # Get the paths to the original and edited images
    orig_paths = glob(
        data_dir / "train" / "**" / "combined_mask_slice=*_nod=*.png", recursive=True
    )
    edit_paths = glob(
        data_dir / "train" / "**" / "image_slice=*_nod=*.png", recursive=True
    )

    # Create a dictionary to store the original and edited image paths along with the instructions
    data = {"orig_path": [], "edit_path": [], "instruction": []}

    # Iterate through the original and edited image paths
    for orig_path, edit_path in zip(orig_paths, edit_paths):
        # Get the PatientID and NoduleID from the original image path
        patient_id = Path(orig_path).parent.parent.parent.name
        nodule_id = Path(orig_path).stem.split("_")[-1].split("=")[-1]

        # Get the malignancy scores of the different annotators for the corresponding PatientID and NoduleID
        malignancy = annotations[
            (annotations["PatientID"] == int(patient_id))
            & (annotations["NoduleID"] == int(nodule_id))
        ]["Malignancy"].values
        malignancy = sum(list(malignancy)) / len(list(malignancy))

        if malignancy >= 4.0:
            malignancy = 1
        elif malignancy < 2.0:
            malignancy = 0
        else:
            malignancy = -1

        for repeat in range(2):

            # Get a random instruction
            instruction = INSTRUCTIONS[
                hash(f"{patient_id}_{nodule_id}_{repeat}") % len(INSTRUCTIONS)
            ]

            # Add `benign` or `malignant` to `lung/pulmonary nodule/lesion` in instruction if malignancy is known
            if repeat == 0 and malignancy == 1:
                instruction = instruction.replace("lung", "malignant lung")
                instruction = instruction.replace("pulmonary", "malignant pulmonary")
            elif repeat == 0 and malignancy == 0:
                instruction = instruction.replace("lung", "benign lung")
                instruction = instruction.replace("pulmonary", "benign pulmonary")

            # Add the original and edited image paths along with the instruction to the dictionary
            data["orig_path"].append(orig_path)
            data["edit_path"].append(edit_path)
            data["instruction"].append(instruction)

        # Write the original and edited image paths along with the instruction to a jsonl file
        pd.DataFrame(data).to_json(
            output_path, orient="records", lines=True
        )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
