

# Residual Cell Feature Refined Deep Multi-scale Network


<hr />


> **Abstract:** *—Bad weather conditions, such as fog and dust, significantly degrade outdoor images, making it difficult for computer vision applications to extract information efficiently. Applications like nighttime dehazing, daytime dehazing, deraining, underwater image enhancement, and low-light enhancement, need a simultaneous generation of haze-free images. To address the issues with image dehazing, we propose an end-to-end progressive multi-stage cascaded network named Residual Cell Feature Refined Deep Multi-scale Network (RCFRMS). RCFRMS consists of a feature extraction module that makes use of four dilated convolutional layers followed by an ARDN
network, it can find multi-scale features on hazy maps of different lengths. The performance of the proposed method is evaluated using PSNR, SSIM, FADE and, CIEDE2000 on different publicly available datasets. Experimental findings demonstrate that our approach quantitatively and, qualitatively outperforms state-of-the-art techniques on synthetic and real-world datasets.* 

## Network Architecture
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/69c0pQv.png" width="1600"> </td>
    <td> <img src = "https://i.imgur.com/JJAKXOi.png" width="1200"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of RCFRMS</b></p></td>
    <td><p align="center"> <b>Supervised Attention Module (SAM)</b></p></td>
  </tr>
</table>

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Quick Run

To test the pre-trained models of [Dehazing](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view?usp=sharing) on your own images, run 
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```
Here is an example to perform Dehazing:
```
python demo.py --task Dehazing --input_dir ./samples/input/ --result_dir ./samples/output/
```

## Training and Evaluation


## Results
Experiments are performed for different image processing tasks like image  dehazing. Images produced by RCFRMS can be downloaded from Google Drive links: [Dehazing](https://drive.google.com/drive/folders/12jgrGdIh_lfiSsXyo-QicQuZYcLXp9rP?usp=sharing)
<details>
  <summary> <strong>Image Dehazing</strong> (click to expand) </summary>
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/UIwmY13.png" width="1600"> </td>
    <td> <img src = "https://i.imgur.com/ecSlcEo.png" width="1200"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Dehazing on Synthetic Datasets.</b></p></td>
    <td><p align="center"><b>Dehazing on Real Dataset.</b></p></td>
  </tr>
</table></details>
