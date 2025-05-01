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
To install the docker environment, first edit the `docker_env/Makefile`
```
IMAGE=img_name/dl-aio
CONTAINER=containter_name
AVAILABLE_GPUS='0,1,2,3'
LOCAL_JUPYTER_PORT=18888
LOCAL_TENSORBOARD_PORT=18006
PASSWORD=yourpassword
WORKSPACE=workspace_directory
```
edit the `img_name`, `containter_name`, `available_gpus`, `jupyter_port`, `tensorboard_port`, `password`, `workspace_directory`

for the first time run the following commands in terminal
```
cd docker_env
make docker-build
make docker-run
```

for further use to docker environment

to stop the environmnet `make docker-stop`

to resume the environmente `make docker-resume` 

**We will release the code soon**
