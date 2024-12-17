<p align="center">
  <a href="https://phenonet.org/phenotools">
    <img src="./assets/logo.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">PhenoSR</h3>
  <p align="center">
Enhancing organ-level phenotyping with super-resolution RGB UAV imagery for large-scale breeding experiments
    <br />
     <a href="https://phenonet.org/phenosr">Offical Site</a>
    ·
     <a href="https://help.phenonet.org/">Help Documents</a>
    ·
      <a href="https://github.com/Jinlab-AiPhenomics/phenosr/issues">Report Bug</a>
</p>

### 📝 Overview

Ruinan Zhang<sup>1</sup>, Shichao Jin<sup>1,*</sup>, Yi Wang<sup>2</sup>, Jingrong Zang<sup>1</sup>, Yu Wang<sup>1</sup>, Ruofan Zhao<sup>1</sup>, Yanjun Su<sup>3</sup>, Jin Wu<sup>4</sup>, Xiao Wang<sup>1</sup>, Dong Jiang<sup>1</sup>

1 College of Agriculture, Plant Phenomics Research Centre, Academy for Advanced Interdisciplinary Studies, Collaborative Innovation Centre for Modern Crop Production co-sponsored by Province and Ministry, Nanjing Agricultural University, Nanjing 210095, China.

2 Institute of Remote Sensing and Geographic Information System, School of Earth and Space Sciences, Peking University, Beijing, China

3 State Key Laboratory of Vegetation and Environmental Change, Institute of Botany, Chinese Academy of Sciences, Beijing 100093, China

4 School of Biological Sciences and Institute for Climate and Carbon Neutrality, The University of Hong Kong, Pokfulam Road, Hong Kong, China

*Corresponding author

Shichao Jin, Associate professor

Head of AiPhenomics Lab, Nanjing Agricultural University

Senior Editor of Plant Phenomics

Email: jschaon@njau.edu.cn; Tel: +86 025 8439-6112

Address: No. 666, Binjiang Avenue, Jiangbei New District, Nanjing 210095, China

## ⚙️ Installation

### Install with Conda

1. Please install Anaconda firstly.
2. We recommend cloning the PhenoSR repository into a clear folder.

   ```python
   cd {your folder}
   git clone https://github.com/Jinlab-AiPhenomics/PhenoSR.git
   cd PhenoSR
   ```
3. Create a clear environment for PhenoSR and activate the environment.

   ```python
   conda create -n phenosr python=3.9
   conda activate phenosr
   ```
4. Install Python requirements.

   ```python
   pip install -r requirements.txt
   ```

## 🚀 Usage

> 🎉 Now you can inference  using the GUI software [PhenoTools](https://phenonet.org/phenotools).

### Inference

1. Download pre-trained model

```python
wget https://github.com/Jinlab-AiPhenomics/PhenoSR/releases/download/0.1.0/phenosr_20240831.pth -P weights
```

2. Usage

```python
python inference.py -i input -o output -m model_path 
  -h, --help            				show this help message and exit
  -t THRESHOLD, --threshold				se_score threshold
  -c CALCULATE, --calculate				calculate metrics
  -m MODEL_PATH, --model_path			model path
  -i INPUT, --input						input folder
  -o OUT, --out							output folder
```

### Training

##### 1. Dataset preparation

1. Construct segmentation dataset

   Please refer to the README.md in the [HRNet](https://github.com/bubbliiiing/hrnet-pytorch) repository.

2. Multiscale scaling

   ```python
   python scripts/generate_multiscale.py -i input -o output
     -h, --help            			show this help message and exit
     -i INPUT, --input					the path of high-resolution UAV imagery
     -o OUT, --out						output folder
   ```

3. Generate a txt for meta information

   ```python
   python scripts/generate_meta_info.py -i input -s save_path
     -h, --help            			show this help message and exit
     -i INPUT, --input					the path of high-resolution UAV imagery following multiscale scaling
     -s SAVE_PATH, --save_path			txt path to save meta info file
   ```

##### 2. Train HRNet

Please refer to the README.md in the [HRNet](https://github.com/bubbliiiing/hrnet-pytorch) repository.

##### 3. Train the PSNR-oriented PhenoSR

1. Modify the content in the option file `options/train_phenosr.yml`

   ```
   train:
     # the path of sr dataset
     dataroot_gt: ~
     # the path of the meta info file
     meta_info: ~
     
   network_g:
     # the number of classes in the segmentation model
     num_classes: ~
     # the path of the modified HRNet model
     seg_model_path: ~
     
   path:
     # Modify it to your storage path
     root_path: ~
   ```

2. Usage

   ```python
   python train.py -opt options/train_phenosr.yml
   ```

##### 4. Fintune  PhenoSR

1. Modify the content in the option file `options/finetune_phenosr.yml`

   ```
   train:
     # the path of sr dataset
     dataroot_gt: ~
     # the path of the meta info file
     meta_info: ~
     
   network_g:
     # the number of classes in the segmentation model
     num_classes: ~
     # the path of the modified HRNet model
     seg_model_path: ~
     
   path:
     # the path of the PSNR-oriented PhenoSR model
     pretrain_network_g: ~
     # Modify it to your storage path
     root_path: ~
   ```

2. Usage

   ```python
   python train.py -opt options/finetune_phenosr.yml
   ```

## 🙏 Acknowledgement

- BasicSR

## 📄 License

[GPL-3.0](LICENSE) © PhenoSR
