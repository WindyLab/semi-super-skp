<div align="center">
  <h1>Semi-Supervised Semantic Key Point Detection via Bundle Adjustment</h1>
<p align="center">
  <a href="https://shiyuzhao.westlake.edu.cn/2024IROSLiKai.pdf">
    <img src="https://img.shields.io/badge/Paper-blue?logo=googledocs&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://pan.baidu.com/s/1KyvN9--4radHq7ZZAiqnig?pwd=128y">
    <img src="https://img.shields.io/badge/Baidu Netdisk-blue?logo=dask&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

</div>

This repository contains the code and data of our paper: "Uncertainty-Aware Semi-Supervised Semantic Key Point Detection via Bundle Adjustment" submitted to **IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024**.

### Dataset Preparation

#### YCB-Video
https://rse-lab.cs.washington.edu/projects/posecnn/

#### Real-world data
For the real-world drone data used in the paper, you can download from:     
https://pan.baidu.com/s/1KyvN9--4radHq7ZZAiqnig?pwd=128y     
password: 128y    

### Code for data preprocessing
We currently provide code for data preprocessing.
#### Environment setup
```bash
conda env create -f environment.yaml
conda activate semi-super-skp
```

#### Generate heatmap labels
1. Change 'img_dir' in config_kpts.py to your custom path.
2. Run
```bash
python utils/generate_ycb_labels.py
python utils/generate_drone_labels.py
```

### Code for model and pose optimization
Code for this part will be released later.


### Others
This project is licensed under the [MIT License](LICENSE).    
If you have any questions, please contact likai [at] westlake [dot] edu [dot] cn
