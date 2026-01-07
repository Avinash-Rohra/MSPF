# Multi-Scale Feature Fusion with Cross-View Head Re-Identification Module for Crowd-Counting

---

This repo is the implementation of **Cross-View Head Re-Identification Module for Crowd-Counting**

## Description
The single and cross-view crowd counting in complex scenarios remains challenging due to severe occlusion, perspective distortion, and the problem of avoiding redundant counting across camera views. This paper presents a novel crowd counting network that efficiently addresses these challenges and accurately counts in single and cross-view complex scenes. The model integrates a hybrid attention module to capture precise spatial and channel features, a multi-scale feature fusion module to capture granular features and broader contextual details from highly dense and sparse crowd images, and a cross-view head re-identification module that integrates feature representation with appearance similarity and geometric weighted similarity that transforms the data and prioritize angular similarity over magnitude-based comparison, which is most essential when dealing with diverse camera viewpoints extracted features. By jointly modeling intra-view and inter-view, the model achieves superior capability in handling crowd counting in single-view and cross-view settings. This research conducts extensive experiments on single-view counting using the ShanghaiTech, UCF-CC50, UCF-QNRF, and JHU-Crowd++ benchmark datasets, as well as on cross-view counting using the WildTrack, MVOR, and MultiviewX datasets. The experimental results with ablation studies validate the model’s localization accuracy, robustness, and counting precision for both single-view and cross-view crowd counting. 

## Model Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/avibest1/MSF-CVHR/fa046fb80b17614fc74efe43674c96be0377938a/figures/Architecture.jpg" width="700"><br>
  <em>Figure: Architecture of the proposed MSF-CVHR framework.</em>
</p>

# Getting Started
## Preparation
-  Prerequisites
    - Python ≥ 3.7
    - PyTorch ≥ 1.6
    - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.
-  Code
    - Clone this repo in the directory (```Root/MSF-CVHR```):
  
- Datasets
    - Download Wildtrack dataset from this [link](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/).
    - Download MVOR dataset from this [link](https://camma.unistra.fr/datasets/).
    - Download MultiviewX dataset from this [link](https://opendatalab.com/OpenDataLab/MultiviewX/explore/main).
    - Unzip ```*zip``` files in turns and place ```datasets*``` into the folder (```dataset_prepare/Cross-View```). 

 
    - If you want to reproduce the results on Shanghai Tech Part A/B , UCF-QNRF, UCF_CC_50 and JHU-Crowd++ datasets for single view, Just place datasets into the folder (```dataset_prepare/Single-View```)   
  
  - Finally, the folder tree is below:
 ```
  -- ProcessedData
  |-- Cross-View
  |-- Wildtrack
	 |-- image_subsets
            |   |-- C1
			|   |-- 00000000.png
			|   |-- 00000005.png
			|   |-- ...
			|   |-- 00002000.png
            |   |-- C2
			|   |-- 00000000.png
			|   |-- 00000005.png
			|   |-- ...
			|   |-- 00002000.png
            ....................
            ....................
            |   |-- C7
			|   |-- 00000000.png
			|   |-- 00000005.png
			|   |-- ...
			|   |-- 00002000.png

     |-- annotations_positions
			|   |-- 00000000.json
			|   |-- 00000005.json
			|   |-- ...
			|   |-- 00001995.json
			|-- train.txt
			|-- val.txt
			|-- test.txt
			|-- val_gt_loc.txt

 ```

## Training

python train.py 

     --img 400 \
     --batch 16 \
     --epochs 150 \
     --data {dataset.location}/data.yaml \
     --weights MSF-CVHR.pt \
     --name MSF-CVHR_results \
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

- The validtion on Wildtrack, the table is shown as follows:

| Methods              | AP   | FPR-95 | P     | R     | ACC   | IPAA-100 | IPAA-90 | IPAA-80 |
|----------------------|------|--------|-------|-------|-------|----------|----------|----------|
| OSNET                | 16.81| 92.08  | 28.27 | 29.67 | 37.55 | 0.00     | 0.48     | 3.21     |
| MvMHAT               | 4.45 | 94.13  | 5.97  | 6.28  | 22.37 | 0.00     | 0.48     | 1.55     |
| OSNET+ESC            | 59.53| 15.33  | 78.12 | 79.08 | 82.12 | 26.43    | 39.40    | 66.68    |
| GNN_CCA              | 4.13 | 93.30  | –     | 0.00  | 36.82 | 0.00     | 2.14     | 14.41    |
| ASNet                | 73.40| 8.30   | –     | –     | –     | 32.10    | –        | –        |
| ViT-P3DE             | 70.45| 5.83   | 86.91 | 87.01 | 89.48 | 35.48    | 53.10    | 84.17    |
| MVA                  | 56.68| 11.18  | 92.31 | 94.34 | 91.73 | 54.64    | 65.60    | 86.55    |
| **MSF-CVHR (Ours)**  | **65.40**| **6.24** | **93.12** | **94.82** | **90.84** | **54.62** | **68.30** | **87.36** |


- The cross-view estimated results on the WildTrack dataset. The first row displays synchronized crowd images with the same timestamp; the second row shows the actual ground truth; the third row presents the estimation results; and the last two rows show the BEV-plotted density maps of distinct individuals.

<p align="center">
  <img src="https://raw.githubusercontent.com/avibest1/MSF-CVHR/14ca06bc82c761a4c48c29a071f297d9c55c1047/figures/Figure%207.jpg" width="700"><br>
  <em>Figure: The cross-view estimated results on the WildTrack dataset.</em>
</p>







