# Datasets and pretrained models

No data or weights are stored in this repository. On Kaggle, attach each dataset with **Add Input → Datasets** and the notebooks find it automatically. Each dataset keeps its own licence and terms of use — check them before reuse.

## Internal collection (training of the original system)

| Organ(s) | Dataset | Link |
|---|---|---|
| 15 organs | Multi-Cancer Multi-Image Dataset (Akmal) | https://www.kaggle.com/datasets/mohamedakmal1/multi-cancer-mulit-image-dataset |
| Gastric (8 tissue types) | HMU-GC-HE-30K | https://www.kaggle.com/datasets/mdabrarhossainzahin/gastric-cancer-data |

## External sources (multi-source training and external testing)

H = held out from all training (test only) · G = trains the gateway only · X = unlabelled extra gateway source

| Organ | Dataset | Role | Link |
|---|---|---|---|
| Blood | C-NMC 2019 (single-cell crops) | train/test by subject | https://www.kaggle.com/datasets/andrewmvd/leukemia-classification |
| Skin | PAD-UFES-20 (smartphone photos) | train/test by patient | https://www.kaggle.com/datasets/mahdavi1202/skin-cancer |
| Skin | ISIC 2018 Task 1 (with lesion masks) | X; localisation test | https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation |
| Cervical | LBC Pap-smear (Mendeley) | train/test | https://www.kaggle.com/datasets/blank1508/mendeley-lbc-cervical-cancer |
| Colon | CRC-VAL-HE-7K | train/test; stain reference | https://www.kaggle.com/datasets/imrankhan77/crc-val-he-7k |
| Ovarian | Ovarian cancer & subtypes (histopathology) | train/test | https://www.kaggle.com/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology |
| Breast | CBIS-DDSM (lesion crops) | train/test by patient | https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset |
| Kidney | KiTS23 2D slices (with tumour masks) | train/test by case; localisation | search Kaggle for `kits23-2d-kidney-tumor-segmentation` |
| Pancreas | MSD Task07 Pancreas | train/test by case (crops) | streamed by `00_data/02` and `00_data/03` (Hugging Face / MONAI mirror) |
| Pancreas | NIH Pancreas-CT (healthy) | **H** (specificity) | https://www.kaggle.com/datasets/tahsin/pancreasct-dataset |
| Thyroid | DDTI | **G** (TI-RADS proxy labels) | https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images |
| Thyroid | TN3K (with nodule masks) | X; localisation test | https://www.kaggle.com/datasets/tjahan/tn3k-thyroid-nodule-region-segmentation-dataset |
| Esophagus | EDD2020 (with lesion masks) | **G**; localisation test | https://www.kaggle.com/datasets/orvile/edd2020-endoscopy-detection-and-segmentation |
| Lung | LungHist700 | **H**; stain reference | https://www.kaggle.com/datasets/abdullahhasansajjad/lunghist700 |
| Oral | Oral histopathology (Rahman et al.) | **H** | https://www.kaggle.com/datasets/ashenafifasilkebede/dataset |
| Brain | PMRAM (brain MRI) | **H** | https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset |

## Nuclei and out-of-distribution tests

| Purpose | Dataset | Link |
|---|---|---|
| Nuclei (train / test) | PanNuke | https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification |
| Nuclei (unseen test) | MoNuSeg 2018 | https://www.kaggle.com/datasets/tuanledinh/monuseg2018 |
| OOD inputs | CIFAR-10 | https://www.kaggle.com/datasets/pankrzysiu/cifar10-python |
| OOD inputs | Chest X-ray (pneumonia) | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |

## Optional (earlier experiments, downloaded by `00_data/01_external_data_importer`)

GasHisSDB (figshare 15066147), FedBCa bladder MRI (Zenodo 10409145), NDB-UFES oral (Mendeley bbmmm4wgr8). Not used by the final system.

## Pretrained models

| Model | Used for | Source |
|---|---|---|
| ConvNeXt-Small, ImageNet-22k | gateway and all 16 specialists | `timm` model `convnext_small.fb_in22k_ft_in1k` |
| MedSAM (ViT-B) | lesion outlines (skin, thyroid) and localisation benchmark | Hugging Face `wanglab/medsam-vit-base` |
| HoVer-Net (PanNuke) | nuclei in reports; comparison with our model | TIAToolbox `hovernet_fast-pannuke` |
| Phikon-v2, BiomedCLIP, UNI2-h | foundation-model baselines (audit) | Hugging Face `owkin/phikon-v2`, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, `MahmoodLab/UNI2-h` (gated; needs `HF_TOKEN`) |

## Weights of MMAF-CD

The trained weights live in my private Kaggle datasets (`marcyoussef22/...`): the original models used by the audit notebooks, and the outputs of the rebuild notebooks (gateway ≈200 MB, 16 specialists in FP16). They are not in this repository because of size. To share them, publish those Kaggle datasets or upload them as GitHub Release assets / to the Hugging Face Hub.
