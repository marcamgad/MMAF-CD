# Run order

All notebooks were written for **Kaggle**. For each one: create a notebook from the `.ipynb`, attach the inputs listed, set the accelerator and Internet as shown, then **Save Version → Save & Run All**. When a notebook finishes, attach its **output** to the next one (Add Input → Your Work → Notebooks). Dataset links: [`DATASETS.md`](DATASETS.md).

"Original models" means my private Kaggle datasets with the weights of the submitted system (`marcyoussef22/gateway1`, `gateway2`, `bloodskinlymphoma`, `retrain`, `coloncervical`, `7models`, `breast`, `3model`, `gastric`, `filmv8`, `sota`, `segmentation`). They are needed only for the audit and for the original-vs-final comparisons.

## 0 · Data preparation

| Notebook | Accelerator | Inputs | Output |
|---|---|---|---|
| `00_data/01_external_data_importer` | CPU, Internet ON | none (downloads) | `mmaf-external-extra` (optional sets) |
| `00_data/02_pancreas_msd_slices` | CPU, Internet ON | none (streams MSD Task07) | `msd-pancreas-slices` |
| `00_data/03_pancreas_centred_crops` | CPU, Internet ON | NIH Pancreas-CT (`tahsin/pancreasct-dataset`) | `msd_pancreas_crops/` + `metadata.csv` |

## 1 · Audit of the original system (optional)

| Notebook | Time | Inputs | Measures |
|---|---|---|---|
| `01_audit/A_core_audit` | 3.5–4.5 h | original models, multi-cancer, gastric, PanNuke, CIFAR-10, chest X-ray | stain-normaliser effect, full-test routing, calibration, conformal, FiLM attribution, shortcut, sanity, OOD, CAM agreement |
| `01_audit/C1_leakage_and_external_validation` | 3–4 h | as A + all external datasets + A's output | near-duplicate leakage, group splits, external validation on 14 sets, stain robustness, energy OOD |
| `01_audit/C2_baselines_hovernet_medsam` | 3–5 h | as A + mask datasets (ISIC, KiTS23, TN3K, DDTI, EDD2020, MoNuSeg) + A's output; optional `HF_TOKEN` | foundation-model probes, HoVer-Net vs pretrained, CAM→MedSAM |
| `01_audit/C2b_baseline_fixes` | 40–60 min | as C2 | re-runs the two C2 parts that were invalid |

## 2 · Rebuild

| Notebook | Time | Inputs | Output |
|---|---|---|---|
| `02_rebuild/01_gateway_multisource` | ~7.7 h, GPU, Internet ON | original models, multi-cancer, gastric, PanNuke, all external datasets | split table `e_table.csv`, gateway, routing results |
| `02_rebuild/02_specialists` | 5–7 h, GPU (resumable) | everything from step 1 + its output | 16 specialists, specialist and robustness results |
| `02_rebuild/03_pancreatic_specialist` | 1–1.5 h, GPU | step 2's output + `00_data/03` output + CRC-VAL + LungHist700 | pancreatic specialist on crops |

Design notes for these three notebooks: [`DESIGN_NOTES_rebuild.md`](DESIGN_NOTES_rebuild.md).

## 3 · Evaluation

| Notebook | Time | Inputs | Output |
|---|---|---|---|
| `03_evaluation/01_end_to_end_and_guard` | 3–4 h, GPU, Internet ON | outputs of all rebuild notebooks + P1/P1b, original models + `segmentation`, multi-cancer, gastric, PanNuke, external sets, ISIC, TN3K, MoNuSeg | `results/end_to_end/` |
| `03_evaluation/02_explanation_benchmark` | 1.5–2 h, GPU ×2, Internet ON | rebuild outputs, external sets with masks | `results/explanation_benchmark/` |
| `03_evaluation/03_hovernet_pretrained` | 10 min, GPU | step 1's `pannuke.npz` and `monuseg.npz` | PQ of the pretrained HoVer-Net |

## 4 · Reliability and reports

| Notebook | Time | Inputs | Output |
|---|---|---|---|
| `04_reporting/01_reliability_and_reports` | 1–1.5 h, GPU ×2, Internet ON | as the explanation benchmark + PanNuke + MoNuSeg; optional `f2_results.json` | `results/reliability_and_reports/` (thresholds, reliability table, nuclei table, 32 HTML + FHIR reports) |

## Notes from running on Kaggle

- Attach only the inputs a notebook needs. A notebook with 28 inputs started slowly and failed when one dataset could not be mounted (`ERRORED_MOUNTING_DATASET`). If a mount fails, remove that optional input and run again; the later notebooks print a coverage table showing which test images were readable.
- Long runs stop themselves before the 12-hour limit; run the notebook again with its own output attached and it resumes from saved checkpoints.
- The split is a deterministic function of `md5(source | group)`, so every notebook sees the same train/validation/test split.
