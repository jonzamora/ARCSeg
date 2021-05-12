1. **Unsupervised Pre-Training to Learn Dataset Specific Features `trainMiccaiRecon.sh`**
    - Train over the entire `m2cai16-tool` training set (581,935 Frames) for Image Reconstruction over 1 epoch
        - Acquired `m2cai16-tool` training data via this form [[Link](https://docs.google.com/forms/d/1RIHj5aenrA37fVWHi3SHmDeIp9Iaz8W302P8dbwI3Po/viewform?edit_requested=true)] and from this Lab [[Link](http://camma.u-strasbg.fr/datasets)]
        - Used `vid2jpg.py` to convert 1 video frame to jpg. Since all videos from `m2cai16-tool` were shot in 25 fps, the script was designed to generate 25 images for every 1 second of the input video. As a result, we get **581,935 jpg images for Training** and **313,455 jpg images for Testing**
        - To train the Image Reconstruction model, I ran [`trainMiccaiRecon.sh`](trainMiccaiRecon.sh) inside of my conda environment using `./trainMiccaiRecon.sh` (See details on Conda Environment Below)
    - **Environment Specifics**
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
        - [log_MiccaiRecon](log_MiccaiRecon)
2. **Initialize Segmentation Network with Weights from the Reconstruction Network `trainMiccaiSeg.sh`**
    - **Data**
        - The data can be found at the following Kaggle Link -> [Link](https://www.kaggle.com/salmanmaq/m2caiseg)
    - To train the Segmentation Network without initializing its weights from the Pre-trained Image ReconNet, we run `./trainMiccaiSeg.sh`
    - The SegNet uses `batch size = 2` and is trained over 90 Epochs
    - **Training Results**
        - [log_MiccaiSeg](log_MiccaiSeg)
3. **Finetune the Segmentation Network `finetuneMiccaiSeg.sh`**
    - Batch Size = 2
    - 90 Epochs
4. **Run Segmentation + Classifier Network `trainMiccaiSegPlusClass.sh`**

--
* TODO: Evaluate Segmentation Predictions via `evaluate.sh`