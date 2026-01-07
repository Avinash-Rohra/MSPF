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







