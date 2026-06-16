# VGTR: Vision-Guided Text Representation Learning for Text-based Person Re-Identification

## 1. Progect Overview
Text-based person retrieval (TBPR) aims to retrieve pedestrian images using natural language descriptions. Many previous studies have tend to focused on text enhancement and cross-modal alignment to improve retrieval performance. However, these strategies merely focus on improving text descriptions or optimizing cross-modal matching, while rich identity-related semantics embedded in pedestrian images have not been fully utilized in text representation learning. Intuitively, semantic information missing in text descriptions cannot be effectively supplemented through the visual modality, making it difficult to learn discriminative text representations, especially in cases where descriptions are short, vague, and semantically sparse. To address this issue, we propose a vision-guided text representation learning framework termed VGTR, which alleviates semantic sparsity by dynamically enriching textual representations with structured visual semantics. Specifically, we introduce a Semantic Slot Memory (SSM) module to progressively accumulate stable identity-aware visual semantic prototypes from pedestrian images. In addition, a Visual-Text Projection (VTP) module is designed to project visual features into a text-compatible semantic space, facilitating effective cross-modal semantic interaction. Through dynamic semantic retrieval and refinement, textual features can acquire complementary visual semantics, resulting in more informative and discriminative representations.
![示例图片](image/framework.png)

## 3. Key algorithm
  **Semantic Slot Memory (SSM) module:** The SSM organizes visual features into multiple semantic slots, where token-level information is softly aggregated and dynamically updated via a momentum mechanism, enabling structured and stable visual knowledge to guide text representation learning.
    
   **Visual-Text Projection module (VTP) module:** This module transforms visual representations into a space more compatible with textual embeddings, facilitating more reliable cross-modal interaction and memory retrieval.

## 4. Environment Setup
### Sofware Dependencies
```
Linux 6.8.0
Python 3.9.12
pytorch 2.6.0
torchvision 0.10.0
cuda 11.3
```
### Hardware Requirements
Nvidia L40 GPU with 48.00 GB
## 5. Installation and Usage
### Clone the repository
```bash
git clone https://github.com/junhaohe777/VGTR.git
```
### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```
### Training

```python
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name iira_decouple_features_id \
--img_aug \
--batch_size 128 \
--MLM \
--loss_names 'sdm+id+mlm' \
--dataset_name 'CUHK-PEDES' \
--num_epoch 60 \
--memory_size 32 \
--num_slots 4 \
--memory_alpha 0.01 \
--memory_momentum 0.9 \
--memory_warmup_epochs 5 \
--learnable_alpha true

```

## 6. Text-to-Image Person Retrieval Results
#### CUHK-PEDES dataset
![示例图片](image/CUHK-PEDES.png)

#### ICFG-PEDES dataset
![示例图片](image/ICFG-PEDES.png)

#### RSTPReid dataset
![示例图片](image/RSTPReid.png)


## 7. Acknowledgments
The code is based on [IRRA](https://github.com/anosorae/IRRA) licensed under Apache 2.0.

## 8. Citation
#### If you use this project's code,please cite our paper:
```bibtex
@article{He_2026_VGTR
  title={VGTR: Vision-Guided Semantic Memory for Text-Based Person Retrieval},
  author={He, Junhao and Zhang, Chengfang and Feng, Ziliang},
  journal={xxx},
  year={2026}
}
```
## 9. Contact Information
- **Email**: 2817881079@qq.com or chengfangzhang@scpolicec.edu.cn
