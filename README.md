# CAM-Seg: A Continuous-valued Embedding Approach for Semantic Image Generation

**Official PyTorch Implementation**

This is a PyTorch/GPU implementation of the paper [CAM-Seg: A Continuous-valued Embedding Approach for Semantic Image Generation](https://arxiv.org/abs/2503.15617)

```
@article{ahmed2025cam,
  title={CAM-Seg: A Continuous-valued Embedding Approach for Semantic Image Generation},
  author={Ahmed, Masud and Hasan, Zahid and Haque, Syed Arefinul and Faridee, Abu Zaher Md and Purushotham, Sanjay and You, Suya and Roy, Nirmalya},
  journal={arXiv preprint arXiv:2503.15617},
  year={2025}
}
```

## Abstract
Traditional transformer-based semantic segmentation relies on quantized embeddings. However, our analysis reveals that autoencoder accuracy on segmentation mask using quantized embeddings (e.g. VQ-VAE) is 8\% lower than continuous-valued embeddings  (e.g. KL-VAE). Motivated by this, we propose a continuous-valued embedding framework for semantic segmentation. By reformulating semantic mask generation as a continuous image-to-embedding diffusion process, our approach eliminates the need for discrete latent representations while preserving fine-grained spatial and semantic details. Our key contribution includes a diffusion-guided autoregressive transformer that learns a continuous semantic embedding space by modeling long-range dependencies in image features. Our framework contains a unified architecture combining a VAE encoder for continuous feature extraction, a diffusion-guided transformer for conditioned embedding generation, and a VAE decoder for semantic mask reconstruction. Our setting facilitates zero-shot domain adaptation capabilities enabled by the continuity of the embedding space. Experiments across diverse datasets (e.g., Cityscapes and domain-shifted variants) demonstrate state-of-the-art robustness to distribution shifts, including adverse weather (e.g., fog, snow) and viewpoint variations. Our model also exhibits strong noise resilience, achieving robust performance ($\approx$ 95\% AP compared to baseline) under gaussian noise, moderate motion blur, and moderate brightness/contrast variations, while experiencing only a moderate impact ($\approx$ 90\% AP compared to baseline) from 50\% salt and pepper noise, saturation and hue shifts.

## Result
Trained on Cityscape dataset and tested on SemanticKITTI, ACDC, CADEdgeTune dataset
<p align="center">
  <img src="demo/qualitative.png" width="720">
</p>

## Prerequisite
To install the docker environment, first edit the `docker_env/Makefile`:
```
IMAGE=img_name/dl-aio
CONTAINER=containter_name
AVAILABLE_GPUS='0,1,2,3'
LOCAL_JUPYTER_PORT=18888
LOCAL_TENSORBOARD_PORT=18006
PASSWORD=yourpassword
WORKSPACE=workspace_directory
```
- Edit the `img_name`, `containter_name`, `available_gpus`, `jupyter_port`, `tensorboard_port`, `password`, `workspace_directory`

1. For the first time run the following commands in terminal:
```
cd docker_env
make docker-build
make docker-run
```
2. or further use to docker environment
- To stop the environmnet: `make docker-stop`
- To resume the environmente: `make docker-resume`

For coding open a web browser ip_address:jupyter_port
```http://localhost:18888```

## Dataset
Four Dataset is used in the work
1. [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
2. [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_step.php)
3. [ACDC Dataset](https://acdc.vision.ee.ethz.ch/)
4. [CAD-EdgeTune Dataset](https://ieee-dataport.org/documents/cad-edgetune)

**Modify the trainlist and vallist file to edit train and test split**

### Dataset structure
- Cityscapes Dataset
```
|-CityScapes
|----leftImg8bit #contians the RGB images
|----gtFine #contains semantic segmentation labels
|----trainlist.txt #image list used for training
|----vallist.txt #image list used for testing
|----cityscape.yaml #configuration file for Cityscapes dataset
```

- ACDC Dataset
```
|-ACDC
|----rgb_anon #contians the RGB images
|----gt #contains semantic segmentation labels
|----vallist_fog.txt #image list used for testing fog data
|----vallist_rain.txt #image list used for testing rain data
|----vallist_snow.txt #image list used for testing snow data
|----acdc.yaml #configuration file for ACDC dataset
```

## Weights
To download the pretrained weights please visit [Hugging Face Repo](https://huggingface.co/mahmed10/CAM-Seg)
- **LDM model** Pretrained model from Rombach et al.'s Latent Diffusion Models is used [Link](https://huggingface.co/mahmed10/CAM-Seg/resolve/main/pretrained_models/vae/modelf16.ckpt)
- **MAR model** Following mar model is used
|Training Data|Model|Params|Link|
|-------------|-----|------|----|
|City | Mar-base| 217M|[link](https://huggingface.co/mahmed10/CAM-Seg/resolve/main/pretrained_models/mar/city768.16.pth)|


Download this weight files and organize as follow
```
|-pretrained_models
|----mar
|--------city768.16.pth
|----vae
|--------modelf16.ckpt
```

**Alternative code to automatically download pretrain weights**
```
import os
import requests

# Define URLs and file paths
files_to_download = {
    "https://huggingface.co/mahmed10/CAM-Seg/resolve/main/pretrained_models/vae/modelf16.ckpt":
        "pretrained_models/vae/modelf16.ckpt",
    "https://huggingface.co/mahmed10/CAM-Seg/resolve/main/pretrained_models/mar/city768.16.pth":
        "pretrained_models/mar/city768.16.pth"
}

for url, path in files_to_download.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {path}")
    else:
        print(f"Failed to download from {url}, status code {response.status_code}")
```

## Validation
Open the `validation.ipnyb` file

Edit the **Block 6** to select which dataset is to use for validation

Run all the blocks

## Training
**We will release the training code by May 10th**

## Acknowlegement
The code is developed on top following codework
1. [latent-diffusion](https://github.com/CompVis/latent-diffusion)
2. [mar](https://github.com/LTH14/mar)
