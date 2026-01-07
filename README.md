# Multi-Scale Pyramid Fusion with Overlap Density Attention Module for Crowd Counting


---

## Description
This repository presents MSPF, a novel crowd counting framework based on multi-scale pyramid fusion with an overlap density attention mechanism. Crowd counting is a challenging computer vision task due to severe occlusion, overlapping individuals, and large scale variations in highly dense scenes. To address these challenges, MSPF introduces a robust architecture composed of three specialized modules: a multi-scale pyramid for extracting features at different receptive field sizes, an overlap density attention module for emphasizing overlapping and non-overlapping crowd regions, and a feature enrichment module for adaptively refining and fusing the most relevant feature information. The model first encodes input images to capture multi-scale features, then applies attention to highlight critical overcrowded areas, followed by adaptive pyramid fusion and feature enrichment to enhance feature representation. Finally, MSPF generates an accurate crowd density map as output. Extensive evaluations on the Highly-Packed-Crowd dataset and four challenging benchmark datasets demonstrate that the proposed approach achieves superior performance in accuracy, efficiency, and robustness compared to state-of-the-art crowd counting methods. 

## Model Architecture

<p align="center">
  <img src="https://github.com/Avinash-Rohra/MSPF/blob/2fe3aced7b72d34c037e6499135161be316ed0b1/figures/Fig%202.jpg" width="700"><br>
  <em>Figure 1: Architecture of the proposed MSPF framework.</em>
</p>

# Getting Started
## Preparation
-  Prerequisites
    - Python ≥ 3.7
    - PyTorch ≥ 1.6
    - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.
-  Code
    - Clone this repo in the directory (```Root/MSPF```):
  
- Datasets
    - The HPC dataset is already available within this repository for ease of access and reproducibility
	- Download ShanghaiTech dataset from this [link](https://www.kaggle.com/datasets/tthien/shanghaitech/).
    - Download UCF-QNRF dataset from this [link](https://www.kaggle.com/datasets/faihajalamtopu/ucf-qnrf/).
    - Download UCF-CC-50 dataset from this [link](https://www.crcv.ucf.edu/data/ucf-cc-50/).
    - Download JHU-Crowd++ dataset from this [link](http://www.crowd-counting.com/).
    - Unzip ```*zip``` files in turns and place ```datasets*``` into the folder (```data_process```). Dataset folders: (HPC, SHHA, SHHB, QNRF, UCF_CC_50, JHU-Crowd++) 

  - Finally, the folder tree is below:
 ```
ProcessedData
|-- ShanghaiTech
|   |-- Part_A
|   |   |-- train
|   |   |   |-- images
|   |   |   |   |-- IMG_1.jpg
|   |   |   |   |-- IMG_2.jpg
|   |   |   |   |-- ...
|   |   |   |-- annotations
|   |   |   |   |-- IMG_1.mat
|   |   |   |   |-- IMG_2.mat
|   |   |   |   |-- ...
|   |   |-- test
|   |       |-- images
|   |       |-- annotations
|   |
|   |-- Part_B
|       |-- train
|       |-- test
|
|-- QNRF
|   |-- train
|   |   |-- images
|   |   |   |-- 0001.jpg
|   |   |   |-- 0002.jpg
|   |   |   |-- ...
|   |   |-- annotations
|   |       |-- 0001.mat
|   |       |-- 0002.mat
|   |       |-- ...
|   |
|   |-- test
|       |-- images
|       |-- annotations
|
|-- UCF_CC_50
|   |-- images
|   |   |-- 1.jpg
|   |   |-- 2.jpg
|   |   |-- ...
|   |-- annotations
|   |   |-- 1.mat
|   |   |-- 2.mat
|   |   |-- ...
|   |-- train.txt
|   |-- test.txt
|
|-- JHU_Crowd
    |-- train
    |   |-- images
    |   |   |-- 000001.jpg
    |   |   |-- 000002.jpg
    |   |   |-- ...
    |   |-- annotations
    |   |   |-- 000001.txt
    |   |   |-- 000002.txt
    |   |   |-- ...
    |
    |-- val
    |   |-- images
    |   |-- annotations
    |
    |-- test
        |-- images
        |-- annotations

 ```

## Training

python train.py 

     --img 400 \
     --batch 16 \
     --epochs 150 \
     --data {dataset.location}/data.yaml \
     --weights MSPF.pt \
     --name MSPF_results \
     --cache
  

   
## Testing and Submitting

- Modify some key parameters in ```test.py```: 
  - ```netName```.  
  -  ```model_path```.  
- Run ```python test.py```. Then the output file (```*_*_test.txt```) will be generated.

## Visualization on the val set
- Modify some key parameters in ```test.py```: 
  - ```test_list = 'val.txt'```
  - ```netName```.  
  -  ```model_path```.  
- Run ```python test.py```. Then the output file (```*_*_val.txt```) will be generated.
- Modify some key parameters in ```vis4val.py```: 
  - ```pred_file```.  
- Run  ```python vis4val.py```. 

# Performance

- Comparison results of MSPF with state-of-the-art methods on ShanghaiTech Part_A, Part_B, UCF_CC_50, UCF_QNRF, and JHU-Crowd++ datasets using MAE and MSE

| Method              | SHHA (MAE/MSE)  | SHHB (MAE/MSE) | CC_50 (MAE/MSE)   | QNRF (MAE/MSE)   | JHU (MAE/MSE)    |
| ------------------- | --------------- | -------------- | ----------------- | ---------------- | ---------------- |
| Zhang et al.        | 181.8 / 277.7   | 32.0 / 49.8    | 467.0 / 498.5     | – / –            | – / –            |
| MCNN                | 110.2 / 173.2   | 26.4 / 41.3    | 466.0 / 497.5     | 242.4 / 363.6    | – / –            |
| CSRNet              | 68.2 / 115.0    | 10.6 / 16.0    | 266.1 / 397.5     | 120.3 / 208.5    | – / –            |
| Multiscale-CNN      | 83.6 / 124.6    | 17.7 / 32.3    | – / –             | – / –            | – / –            |
| Wan et al.          | 61.3 / 95.4     | 7.3 / 11.7     | – / –             | 84.3 / 147.5     | – / –            |
| MNA                 | – / –           | – / –          | – / –             | – / –            | 67.7 / 258.5     |
| MSSRGN++            | 86.6 / 137.8    | – / –          | – / –             | 115.6 / 213.9    | – / –            |
| UOT                 | – / –           | – / –          | – / –             | – / –            | 60.5 / 252.7     |
| PaDNet              | 59.2 / 98.1     | 8.1 / 12.2     | 185.8 / 278.3     | – / –            | – / –            |
| DM-Count            | 59.7 / 95.7     | 7.4 / 11.8     | 211.0 / 291.5     | – / –            | – / –            |
| PFSNet              | – / –           | – / –          | – / –             | – / –            | 61.2 / 257.8     |
| MPCount             | 99.6 / 182.9    | 11.4 / 19.7    | – / –             | 165.6 / 290.4    | – / –            |
| MSFFNet             | 58.6 / 93.2     | 6.2 / 10.1     | 180.2 / 245.0     | 85.3 / 146.7     | 60.6 / 250.8     |
| **MSPF (Ours)**     | **52.4 / 87.2** | **5.9 / 10.2** | **178.1 / 240.3** | **84.4 / 145.5** | **59.2 / 250.9** |




- Estimated results on proposed Highly-Packed-Crowd dataset. The first column shows the overlapped input images, the second column shows the actual ground truth, and the third, fourth, and fifth column shows non-overlap estimation, overlapped estimation, and  final estimated density map, respectively.

<p align="center">
  <img src="https://github.com/Avinash-Rohra/MSPF/blob/7f512c3d9b3ac17626b55300d03b22ab58991843/figures/Figure%201.jpg" width="700"><br>
  <em>Figure 2: The estimated results on the Highly-Packed Crowd dataset.</em>
</p>







