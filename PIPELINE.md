## ARCNet Model Pipeline

---

### [Baseline Model] - Unsupervised Semantic Segmentation for m2caiseg

1. **Unsupervised Pre-Training to Learn Dataset Specific Features [`trainMiccaiRecon.sh`](scripts/trainMiccaiRecon.sh)**
    - Train over the entire `m2cai16-tool` training set (581,935 Frames) for Image Reconstruction over 1 epoch
        - Acquired `m2cai16-tool` training data via this form [[Link](https://docs.google.com/forms/d/1RIHj5aenrA37fVWHi3SHmDeIp9Iaz8W302P8dbwI3Po/viewform?edit_requested=true)] and from this Lab [[Link](http://camma.u-strasbg.fr/datasets)]
        - Used [`vid2jpg.py`](src/vid2jpg.py) to convert 1 video frame to jpg. Since all videos from `m2cai16-tool` were shot in 25 fps, the script was designed to generate 25 images for every 1 second of the input video. As a result, we get **581,935 jpg images for Training** and **313,455 jpg images for Testing**
        - To train the Image Reconstruction model, I ran [`trainMiccaiRecon.sh`](scripts/trainMiccaiRecon.sh) inside of my conda environment using `./trainMiccaiRecon.sh` (See details on Conda Environment Below)
    - **Conda Environment Configuration**
        - `conda activate pytorch` - Using conda environment to containerize PyTorch and CUDA dependencies for ReconNet + SegNet
        - Directions:
            ```
            conda create --name pytorch python=3.7
            conda activate pytorch
            conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
            conda install numpy matplotlib scikit-learn pandas tqdm jupyter scikit-image
            conda install -c conda-forge tensorboardx
            conda install -c conda-forge opencv
            conda install -c anaconda scipy=1.5.3
            ```
        - Note: All tests were run on `Ubuntu 21.04` with NVIDIA GeForce RTX 3080 Mobile / Max-Q 16GB VRAM GPU and `CUDA Version: 11.3` [Checked via `nvidia-smi`]
    - **Training Results**
        - [log_MiccaiRecon](logs/log_MiccaiRecon)
2. **Initialize Segmentation Network with Weights from the Reconstruction Network [`trainMiccaiSeg.sh`](scripts/trainMiccaiSeg.sh)**
    - **Data**
        - The data can be found at the following Kaggle Link -> [Link](https://www.kaggle.com/salmanmaq/m2caiseg)
    - To train the Segmentation Network without initializing its weights from the Pre-trained Image ReconNet, we run `./trainMiccaiSeg.sh`
    - The SegNet uses `batch size = 2` and is trained over 90 Epochs
    - **Training Results**
        - [log_MiccaiSeg](log_MiccaiSeg)
3. **Finetune the Segmentation Network [`finetuneMiccaiSeg.sh`](scripts/finetuneMiccaiSeg.sh)**
    - Batch Size = 2
    - 90 Epochs
4. **Run Segmentation + Classifier Network [`trainMiccaiSegPlusClass.sh`](scripts/trainMiccaiSegPlusClass.sh)**

5. **Evaluate Segmentation Predictions via [`evaluate.sh`](scripts/evaluate.sh)**

---

### [Proposed Model] - Semi-Supervised Semantic Segmentation for m2caiseg
1. ARCNet will aim to utilize a Semi-Supervised approach for improving the task of Semantic Segmantion for live surgical video datasets.
    * The ["Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation"](https://arxiv.org/abs/2012.10782) paper provides a successful framework for semi-supervised semantic segmentation that is enhanced by self-supervised monocular depth estimation from unlabeled image sequences. They transfer knowledge from features learned during self-supervised depth estimation to semantic segmentation.
    * Github Repo: [Link](https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth)
    * For pre-training, we will train a self-supervised depth estimation model on `m2cai16-tool` image sequences and transfer the pre-trained weights to our semantic segmentation model.
2. Create Image Sequences for Unlabeled Monocular Depth Estimation Model Training
    * The data preparation scripts for the image sequences can be found in the [SfMLearner](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) paper's `data/` directory in the following [Github Repo](https://github.com/tinghuiz/SfMLearner)
    * The image sequences will consist of 3-tuples in the format of `(prior frame, current frame, next frame)` since that is what the paper uses in their method.
3. To improve on the paper from [1](https://arxiv.org/abs/2012.10782), we will utilize a better self-supervised depth estimation model that was developed by Niantic Labs. The paper originally used the [Monodepth2](https://github.com/nianticlabs/monodepth2) model, but we will use the new-and-improved [manydepth](https://github.com/nianticlabs/manydepth) model (also developed by Niantic Labs) for our work.
    * We use this depth estimation model since it provides overall improvements for depth estimation and will ideally transfer these improvements to our semantic segmentation model.