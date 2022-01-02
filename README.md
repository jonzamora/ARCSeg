# ARCSeg - Supervised Semantic Segmentation Benchmarking Framework for Surgical Robotics 📹 🤖 🪡

---

The goal of ARCSeg is to produce accurate segmentation masks for images from laparascopic gallbladder removal surgeries (cholecystectomy)

To achieve this goal, we have conducted an extensive study of various semantic segmentation networks including:

- `UNet`
- `UNet++`
- `DeepLabV3+`
- `MANet`

For each of these networks, experiments are split between:
- Initializing the networks from scratch
- Initializing the networks from pretrained weights

When initializing the networks, we have found success in using the following encoders with pretrained weights:
- `ResNet18`
- `ResNet34`
- `ResNet50`
- `EfficientNet-B4`

The above ResNets and EfficientNet-B4 are all trained on ImageNet, and they serve as strong encoder backbones for our segmentation networks

## Training a Semantic Segmentation Network

To run a Semantic Segmentation experiment, you can run the following commands:

```bash
cd scripts
./trainSegNet.sh
```

The data for the experiments are organized as follows:
```plain
|__ src
	|__ data
		|__ datasets
                    |__ synapse
                        |__ train
                            |__ images
                            |__ groundtruth
                        |__ trainval
                            |__ images
                            |__ groundtruth
                        |__ test
                            |__ images
                            |__ groundtruth
                            
                    |__ cholec
                        |__ train
                            |__ images
                            |__ groundtruth
                        |__ trainval
                            |__ images
                            |__ groundtruth
                        |__ test
                            |__ images
                            |__ groundtruth
```
Here, the `synapse` dataset is provided to us by the hosts of the HeiChole 2021 competition, and `cholec` is provided to us from the following Kaggle data repository: https://www.kaggle.com/newslab/cholecseg8k. Currently, we use `cholec` as a pretraining dataset.


Prior to running your experiments, though, be aware of the following models:

```bash
model="smp_UNet++" 
# other acceptable options include:
# model="segnet"
# model="unet"
# model="resnet18_unet"
# model="smp_unet18"
# model="smp_deeplabv3+"
# model="smp_MANet"
```

If the model contains the `smp` prefix, this means the model comes from the [Segmentation Models in PyTorch](https://github.com/qubvel/segmentation_models.pytorch) package. Further, each `smp` model is initialized with a `resnet18` encoder trained on ImageNet by default. To implement your own parameters and tune the models, please look at the following [technical documentation](https://smp.readthedocs.io/en/latest/models.html)

Otherwise, `unet` is trained from scratch, and `segnet` is adopted from the following paper: [m2caiSeg: Semantic Segmentation of Laparoscopic Images using Convolutional Neural Networks](https://arxiv.org/abs/2008.10134). Additionally, our code repository is based on this paper's repository, which can be found [here](https://github.com/salmanmaq/segmentationNetworks).

Our repository has introduced many updates to the `m2caiSeg` paper's repository, and additionally, we have expanded the code to work with various models, as can be seen above.

If you have any questions, please file an issue, and we will address your inquiry as best and as soon as we can.

## Environment Setup

- **Conda Environment Configuration**
    - Using conda environment to containerize PyTorch and CUDA dependencies for ARCSeg experiments (assuming you have installed Miniconda or Anaconda)
    - Environment Creation Commands:
        ```
        conda create --name torch python=3.7
        conda activate torch
        conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
        conda install numpy matplotlib scikit-learn pandas tqdm jupyter scikit-image
        conda install -c conda-forge tensorboardx
        conda install -c conda-forge opencv
        conda install -c anaconda scipy=1.5.3
        ```
    - Note: With the above `torch` environment, all experiments were successfully run on `Ubuntu 21.04` with NVIDIA GeForce RTX 3080 Mobile / Max-Q 16GB VRAM GPU and `CUDA Version: 11.3` [Checked via `nvidia-smi`]
