# Diffusion-based Lung Nodule Synthesis

This repository contains the code for the paper "Diffusion-based lung nodule synthesis for advanced evaluation of deep learning models", presented at SPIE Medical Imaging 2025 Conference in San Diego, CA, USA. The method presented in this paper synthesizes lung nodules in 2D computed tomography (CT) slices of the human chest.

![Graphical Summary](https://github.com/cclaess/MagicNod/blob/main/images/graphical_summary.png)

## Trained Encoders and Dataset
The trained diffusion model and VQ-VAE will be make publicly available upon publication. Additionally, we will release a dataset containing slices of the LIDC-IDRI dataset with synthetic nodules equally spaced throughout the lungs, avoiding any biases related to size and location of the nodules and thereby providing a benchmark for biases assessment on lung nodule detection models. You will be able to download them from the following links:

- [Trained Encoders](#) (link will be added upon publication)
- [Released Dataset](#) (link will be added upon publication)

## Add environment
Instructions for adding a .env file to the repository:
 
 1. Create a new file named `.env` in the root directory of your project.
 2. Open the `.env` file and add your environment variables in the format `KEY=VALUE`.
    For example:
    ```
    CONTAINER_PATH=your_docker_sif_file
    WANDB_API_KEY=your_WandB_api_key
    ```
 3. Save the `.env` file.
 4. To ensure that the `.env` file is not committed to version control, add it to your `.gitignore` file:
    ```
    # .gitignore
    .env
    ```

 > Note: Never share your `.env` file publicly or commit it to version control, as it may contain sensitive information.

 
## Citation
If you find this work useful for your own research, please cite as following:
```
Citation will be added upon publication
```