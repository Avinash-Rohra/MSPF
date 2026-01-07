# Multi-Scale Pyramid Fusion with Overlap Density Attention Module for Crowd Counting


---

## Description
This repository presents MSPF, a novel crowd counting framework based on multi-scale pyramid fusion with an overlap density attention mechanism. Crowd counting is a challenging computer vision task due to severe occlusion, overlapping individuals, and large scale variations in highly dense scenes. To address these challenges, MSPF introduces a robust architecture composed of three specialized modules: a multi-scale pyramid for extracting features at different receptive field sizes, an overlap density attention module for emphasizing overlapping and non-overlapping crowd regions, and a feature enrichment module for adaptively refining and fusing the most relevant feature information. The model first encodes input images to capture multi-scale features, then applies attention to highlight critical overcrowded areas, followed by adaptive pyramid fusion and feature enrichment to enhance feature representation. Finally, MSPF generates an accurate crowd density map as output. Extensive evaluations on the Highly-Packed-Crowd dataset and four challenging benchmark datasets demonstrate that the proposed approach achieves superior performance in accuracy, efficiency, and robustness compared to state-of-the-art crowd counting methods. 

## Model Architecture

<p align="center">
  <img src="https://github.com/Avinash-Rohra/MSPF/blob/2fe3aced7b72d34c037e6499135161be316ed0b1/figures/Fig%202.jpg" width="700"><br>
  <em>Figure: Architecture of the proposed MSPF framework.</em>
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

| Method              | ShanghaiTech A |       | ShanghaiTech B |      | UCF_CC_50 |       | UCF_QNRF |       | JHU-CROWD++ |       |
| ------------------- | -------------- | ----- | -------------- | ---- | --------- | ----- | -------- | ----- | ----------- | ----- |
|                     | MAE            | MSE   | MAE            | MSE  | MAE       | MSE   | MAE      | MSE   | MAE         | MSE   |
| Zhang et al. [30]   | 181.8          | 277.7 | 32.0           | 49.8 | 467.0     | 498.5 | –        | –     | –           | –     |
| MCNN [1]            | 110.2          | 173.2 | 26.4           | 41.3 | 466.0     | 497.5 | 242.4    | 363.6 | –           | –     |
| CSRNet [21]         | 68.2           | 115.0 | 10.6           | 16.0 | 266.1     | 397.5 | 120.3    | 208.5 | –           | –     |
| Multiscale-CNN [31] | 83.6           | 124.6 | 17.7           | 32.3 | –         | –     | –        | –     | –           | –     |
| Wan et al. [32]     | 61.3           | 95.4  | 7.3            | 11.7 | –         | –     | 84.3     | 147.5 | –           | –     |
| CDCC [33]           | 76.3           | 144.2 | 11.4           | 17.1 | 336.5     | 486.1 | 134.3    | 240.3 | –           | –     |
| CLTR [34]           | 56.9           | 95.2  | 6.5            | 10.6 | –         | –     | 85.8     | 141.3 | –           | –     |
| DGCC [35]           | 121.8          | 203.1 | 12.6           | 24.6 | –         | –     | 119.4    | 216.6 | –           | –     |
| MNA [36]            | –              | –     | –              | –    | –         | –     | –        | –     | 67.7        | 258.5 |
| DMCNet [37]         | 58.4           | 84.5  | 8.6            | 13.6 | –         | –     | 96.5     | 163.9 | –           | –     |
| MSSRGN++ [38]       | 86.6           | 137.8 | –              | –    | –         | –     | 115.6    | 213.9 | –           | –     |
| TopoCount [39]      | –              | –     | –              | –    | –         | –     | –        | –     | 60.9        | 267.4 |
| UOT [40]            | –              | –     | –              | –    | –         | –     | –        | –     | 60.5        | 252.7 |
| PaDNet [41]         | 59.2           | 98.1  | 8.1            | 12.2 | 185.8     | 278.3 | –        | –     | –           |       |



- The cross-view estimated results on the WildTrack dataset. The first row displays synchronized crowd images with the same timestamp; the second row shows the actual ground truth; the third row presents the estimation results; and the last two rows show the BEV-plotted density maps of distinct individuals.

<p align="center">
  <img src="https://raw.githubusercontent.com/avibest1/MSF-CVHR/14ca06bc82c761a4c48c29a071f297d9c55c1047/figures/Figure%207.jpg" width="700"><br>
  <em>Figure: The cross-view estimated results on the WildTrack dataset.</em>
</p>







