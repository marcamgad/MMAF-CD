# Audit of the original (submitted) system

Numbers from the audit notebooks in `notebooks/01_audit/`. They describe the system as submitted with the thesis, re-evaluated correctly.

## 1. Effect of the stain-normalisation bug (same test images)

Legacy = the submitted evaluation transform (H&E stain normaliser on every image); Fixed = resize + ImageNet normalisation, as in training.

| Organ | Modality | Macro-F1 legacy | Macro-F1 fixed | Routing legacy | Routing fixed |
|---|---|---|---|---|---|
| Lymphoma | H&E | 0.593 | **0.972** | 0.865 | 1.000 |
| Gastric | H&E | 0.604 | **0.811** | **0.382** | 0.965 |
| Breast | Mammography | 0.799 | **0.971** | 0.880 | 0.997 |
| Ovarian | H&E | 0.734 | 0.855 | 0.948 | 0.975 |
| Pancreatic | CT | 0.882 | 0.981 | 1.000 | 1.000 |
| Skin | Dermoscopy | 0.873 | 0.960 | 0.950 | 1.000 |
| Cervical | Cytology | 0.938 | 1.000 | 0.985 | 1.000 |
| Thyroid | Ultrasound | 0.523 | 0.584 | 0.919 | 1.000 |
| Brain | MRI | 0.967 | 0.995 | **0.770** | 1.000 |
| Kidney | CT | 0.982 | 1.000 | **0.615** | 1.000 |
| Esophageal | Endoscopy | 0.976 | 0.997 | 0.998 | 1.000 |
| Oral | H&E | 0.857 | 0.876 | 1.000 | 1.000 |
| Blood | Smear | 0.990 | 1.000 | 1.000 | 1.000 |
| Colon | H&E | 1.000 | 1.000 | 1.000 | 1.000 |
| Lung | H&E | 0.998 | 0.997 | 0.847 | 0.997 |
| Bladder | MRI | 0.494 | 0.451 | **0.741** | 1.000 |
| **Mean** | | 0.826 | **0.903** | 0.869 | **0.996** |

## 2. Routing on the full internal test split (37,068 images, corrected)

Top-1 accuracy **99.23%** (13 organs above 99.5%). At the deployed threshold 0.70: 97.7% dispatched, 0.31% misrouted without warning, 59.6% of routing errors caught.

## 3. Leakage (1,000 test images per organ)

Share of test images within 6 bits (D8 perceptual hash) of a training image: kidney 0.78, lung 0.64, colon 0.62, bladder 0.66, breast 0.60, brain 0.55, **pancreatic 0.99** (42.8 images per near-duplicate group ⇒ about 33 independent CT volumes, labelled per patient).

## 4. Does FiLM condition, or copy? (test macro-F1)

| Organ | Base | Gateway only | Late fusion | FiLM |
|---|---|---|---|---|
| Oral | 0.883 | 0.963 | 0.967 | **0.972** |
| Gastric | 0.825 | 0.608 | 0.724 | **0.836** |
| Breast | 0.959 | 0.921 | 0.940 | 0.960 |
| Ovarian | 0.864 | 0.630 | 0.789 | 0.862 |
| Thyroid | 0.569 | **0.850** | 0.848 | 0.835 |
| Bladder | 0.411 | 0.959 | **0.969** | 0.959 |

For bladder and thyroid the gateway alone matches FiLM: the FiLM model was reading the hint.

## 5. External validation of the original system (14 public sets)

Routing to the correct specialist collapsed for sources not seen in training: colon (CRC-VAL-HE-7K) **2.9%** (70% sent to gastric), lung (LungHist700) **0.2%**, blood (C-NMC) **0%**. Specialists held up on 4 of 14 sets (oral F1 0.965, brain 0.924, colon AUC 0.960, lung AUC 0.80); the pancreatic specialist called every healthy NIH pancreas a tumour (specificity 0.000).

## 6. Explanations

Independent saliency maps agreed weakly (mean |ρ| = 0.17; negative for 11 of 14 organs). The previously reported agreement of 0.56 came from same-layer comparisons and a fallback that selected on its own outcome. Lymphoma and pancreatic maps were flat (hook failure).

## 7. Foundation-model baselines (internal split, macro-F1)

Specialists beat a frozen BiomedCLIP linear probe by 11–19 points on skin, breast, bladder and thyroid, and beat a Phikon-v2 probe on gastric (0.836 vs 0.633) and ovarian (0.862 vs 0.790). Phikon-v2 won on lymphoma (0.998 vs 0.977).
