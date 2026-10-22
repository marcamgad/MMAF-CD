# MMAF-CD v25 — Stage C2 + Stage E (model upgrade)

**Purpose of this package:** turn the Stage C1 findings into a better *model*, not just a more honest paper.
C1 showed the specialists are strong where the data is clean. The weak points are **routing on new data**, **stain/compression fragility** and **leaky splits**. Stage E fixes those three at the source: the data, the training protocol and the augmentation. Model size does not change.

| File | What it is |
|---|---|
| `MMAF_CD_v24_6_stageC2.ipynb` | Stage C2, unchanged from before: foundation-model baselines, official HoVer-Net, MedSAM. **Run it now, in parallel with E1.** |
| `MMAF_CD_v25_stageE1.ipynb` | Stage E1: new multi-source gateway (organ router) |
| `MMAF_CD_v25_stageE2.ipynb` | Stage E2: retrains all 16 specialists |
| this file | What was built, why, how to run it, how to read the output |

---

## 1. What C1 proved, and what Stage E changes

| C1 finding | Root cause | Stage E fix | Where |
|---|---|---|---|
| Gateway routes new sources wrongly: lung 0.2%, colon 2.9%, blood 0% | Every organ came from **one** dataset, so "organ" and "that dataset's look" were the same signal | Train on **several sources per organ**. Add a **within-organ source-adversarial head** that removes "which dataset" features while keeping organ features. | E1 |
| Colon patches sent to gastric 70% of the time | Colon and gastric tissue-type patches look identical, so a single route is impossible | **Route sets**: when the gateway is unsure, the image goes to every organ in a calibrated set. Coverage is guaranteed at 90% on data like the calibration data. | E1 |
| Lymphoma 0.98 → 0.52 under stain change and JPEG | No stain or compression augmentation in training | **HED stain jitter**, plus JPEG 25–90, blur, down-scaling and noise, applied to every specialist and the gateway | E1 + E2 |
| Pancreatic: every healthy CT called tumour | Training slices (chest, pelvis) carry the **patient** label, about 33 patients, 83% exact copies | Train only on **slice-level labels** from tumour masks (MSD Task07). The old data is kept for the gateway only. | E2 |
| Image-level splits (colon 2.1, kidney 4.9, pancreatic 43 images per original) | `train_test_split` on images | **Group splits**: near-duplicates (D8 pHash ≤ 6) and patients (where the dataset gives IDs) stay on one side | E0 |
| External labels are coarser than the specialist classes (C-NMC "ALL" vs early/pre/pro, LBC NILM vs 5 cell types, ovarian "serous" vs HGS/LGS) | Heterogeneous label spaces | **Set-NLL loss**: −log Σ p(c) over the allowed classes, so coarse labels train the fine model correctly | E1 + E2 |
| Bladder base specialist 0.41, thyroid 0.57; FiLM "gains" were the gateway's subclass head | Weak base training; FiLM copied the hint | Retrain both properly. FiLM is **replaced by validated late fusion** (see §5). | E2 |

---

## 2. The data protocol (stage E0, shared by E1 and E2)

One table holds every image: `path · organ · source · group · split · label_set · spec_ok · holdout · D8 pHash`. It is saved as `e_out/e_table.csv`. E2 reuses it when E1's output is attached, which saves about 20 minutes.

**Sources per organ**

| Organ | Internal | External, trained on 60% / tested on 30% | Gateway-only extra |
|---|---|---|---|
| blood | multi-cancer | C-NMC 2019 (grouped by subject ID) | — |
| skin | multi-cancer | PAD-UFES-20 (grouped by patient ID) | ISIC-2018 task-1 images |
| cervical | multi-cancer | LBC Pap (NILM → {mep, pab, sfi}; LSIL/HSIL/SCC → {dyk, koc}) | — |
| colon | multi-cancer | CRC-VAL-HE-7K (tum/norm) | — |
| breast | multi-cancer | **CBIS-DDSM lesion crops** (grouped by patient) instead of shrunk full mammograms | — |
| kidney | multi-cancer | KiTS23 2D (grouped by case) | — |
| ovarian | multi-cancer | Kasture subtypes (serous → {HGS, LGS}) | — |
| pancreatic | multi-cancer (**gateway only**) | NIH healthy CT (by volume) + **MSD Task07** if attached | — |
| thyroid | multi-cancer | DDTI (**gateway only**: TI-RADS labels are proxies) | TN3K images |
| esophageal | multi-cancer | EDD2020 (**gateway only**: Barrett's ≠ "benign") | — |
| gastric, lymphoma, bladder | internal only | — | — |
| **oral, brain_2d, lung** | multi-cancer | **HOLDOUT**: Rahman oral, PMRAM brain, LungHist700 are **never trained on** | — |

**Rules**

- **Internal split: 70/15/15 by group.** A group is a connected set of D8 near-duplicates, meaning flipped or rotated copies. If one group swallows more than 10% of a source, the link threshold tightens automatically from 6 to 4, 2 and then 0, and the log says so.
- **External split: 60/10/30 by group.** A group is the patient or case ID where the dataset has one, and a pHash group otherwise.
- **Holdout sets are test-only.** They are the true external test in the paper: data from a new hospital that no model has seen.
- **Cross-source de-duplication.** Any validation or test image within 6 bits of *any* training image of the same organ is removed. The log prints how many, per source.
- **The split is deterministic** (md5 of `source|group`), so E1, E2 and the later Stage F see identical splits.

> **Trade-off, stated plainly:** training on 60% of an external set means that set's 30% test portion is *in-distribution* for the new models. Only the three HOLDOUT sets measure true generalisation. The paper reports both columns separately. To make another set a true external test, add its key to `HOLDOUT_EXTERNAL` in the first cell.

---

## 3. Stage E1: the gateway

**Architecture:** the same `HierarchicalMedViT` class as before, so every downstream cell loads it unchanged. It has a MAP layer, a ConvNeXt-S backbone (fresh ImageNet-22k weights, not the old gateway, because the old one may have seen the new test images), an organ head and 16 subclass heads.

**What's new**

1. **Balanced sampling.** Every organ gets equal weight, and every source gets equal weight inside its organ, so a 30k-image source cannot drown out a 400-image source.
2. **Losses:**
   - organ cross-entropy (label smoothing 0.1);
   - 0.5 × subclass set-NLL;
   - **source-adversarial loss.** A small head behind a gradient-reversal layer tries to predict the source, but only among the **sources of the same organ**. Its strength ramps from 0 to 0.3. The gateway therefore gets no reward for recognising "C-NMC style" versus "multi-cancer style", while organ information is untouched. A plain source-adversary would also erase organ identity, because each organ has its own sources.
3. **Robustness augmentation**, the same as E2 (§4).
4. **Schedule:** 2 epochs with the backbone frozen, then 12 epochs of full fine-tuning. Each epoch is 36k images; learning rate 6e-5 for the backbone and 6e-4 for the heads, cosine schedule, AMP.
5. **Model selection** uses the **mean per-(organ, source) validation accuracy**, not pooled accuracy, so the checkpoint that routes *every* source well wins.
6. **After training:**
   - an organ temperature T is fitted on validation;
   - a **route-set conformal q̂** (LAC, α = 0.10) is fitted on validation. The route set is every organ with p ≥ 1 − q̂. It contains the true organ at least 90% of the time on data like the calibration data, and is usually a single organ.

**E1b: the evaluation (old vs new on identical images)**

| Column | Meaning |
|---|---|
| `acc` | top-1 organ accuracy |
| `flag` | share with max p < θ = 0.70: not dispatched, sent to a human |
| `silent` | share with max p ≥ 0.70 **and** wrong: the dangerous case |
| `set cov` / `|S|` | route-set coverage and mean size |

- Sets evaluated: internal test, each external test portion, the **HOLDOUT** sets, and the ISIC and TN3K extras.
- Old-gateway numbers on the *internal* test set are optimistic, because it was trained on an image-level split.
- **Dataset-identification probe.** For each organ with two or more sources, a linear probe on the gateway's 256-d features tries to predict the source (5-fold balanced accuracy). *Old high → new near chance* is direct evidence the shortcut was removed. That is a clean, novel figure for the paper.

**Outputs:** `e_out/gateway_v25.pth` (includes `organ_T`, `route_qhat`, `theta`, training log), `e1_results.json`, `tab_routing_v25.tex`, `e_table.csv` and the zip.

---

## 3b. E1 result (run 1, 7.7 h) and the v25.1 changes it caused

| Finding | Severity | What changed |
|---|---|---|
| Seen external sources now route at 99–100 % (blood 0→1.00, colon 0.03→1.00, breast 0.41→1.00, kidney 0.40→1.00) | ✅ win | — |
| Unseen LungHist700 still 0 % correct, 95.6 % misrouted **without a flag** | 🔴 critical | new cell **E2a: source-novelty guard** (Mahalanobis distance to every (organ, source) cluster, Ledoit–Wolf covariance, 95 % val threshold). It reports, per unseen set, how many silent misroutes the guard turns into "review" flags, and where the misrouted images go |
| Source-adversarial head did nothing (dataset-ID probe ≈ unchanged) | 🟠 major | kept, reported as a negative result; the guard uses the leftover source signal instead of fighting it |
| Route conformal q̂ = 0.003 → every set is a singleton | 🟡 minor | reported honestly; the guard, not the route set, carries the abstention |
| 66 min lost searching `/kaggle/input` 5 levels deep; MSD slices not attached | 🟠 major | search now only walks small attached inputs (≤ 4 levels) and never the image datasets |
| E2 would have crashed after training pancreatic (robustness test looked for an `internal` pancreatic test set that no longer exists) | 🔴 critical (caught in mock run) | robustness falls back to the largest test source |

E2 now runs: **guard (≈ 10–15 min) → pancreatic (MSD) → other 15 specialists**. MSD Task07 is treated as unseen by the gateway (it was added after E1), so it is a fourth hold-out for the guard.

## 4. Stage E2: the specialists

**For each organ**, in this order: pancreatic, lymphoma, skin, breast, kidney, blood, cervical, colon, lung, ovarian, bladder, thyroid, esophageal, gastric, oral, brain.

- **Backbone:** ImageNet-22k ConvNeXt-S for all organs. The old set mixed ViT-S, EfficientViT, EffNet-B2/B3 and ConvNeXt-T. One strong backbone makes the comparison clean.
- **Data:** group-split internal data plus the 60% training portion of the organ's external set(s), `spec_ok` only.
- **Sampling:** balanced over source × label-set.
- **Loss:** 0.95 × set-NLL + 0.05 × uniform term (label smoothing that also works for coarse labels).
- **Augmentation:**

| Applied to | Augmentation |
|---|---|
| H&E, smears, cytology | HED stain jitter (±25% per stain channel, Tellez et al. 2019), rot90 + flips |
| Colour photos (skin, endoscopy) | hue and saturation jitter |
| All organs | random resized crop, brightness/contrast/gamma, blur, 0.3–0.8× down-scaling, Gaussian noise, JPEG quality 25–90 |
| CT (pancreas, kidney) | rot90 as well, so scanner orientation cannot become a label cue |

- **Schedule:** 12 epochs of clip(2 × n_train, 3,000, 8,000) samples. Learning rate 1e-4 for the backbone and 1e-3 for the head, 200-step warm-up then cosine, early stop after 4 epochs without improvement.
- **Selection:** the **mean per-source validation balanced accuracy**.
- **Calibration:** temperature (fitted with set-NLL) and LAC q̂ on validation.
- **Late fusion:** if E1's gateway is attached, `(1−w)·specialist + w·gateway-subclass` with w ∈ {0, 0.25, 0.5}, chosen on validation. Because the gateway never trained on those validation images, there is no hint leakage.
- **Checkpoint:** `e_out/spec_v25/<organ>.pth` in fp16. It has the same keys as your old checkpoints (`model_state_dict`, `class_names`, `backbone_name`, `dropout`, `val_f1`) plus `temperature`, `qhat_lac` and `fusion_w`, so the existing loader reads it.

**Evaluation, printed per organ (old vs new on identical images):**

- balanced set-accuracy, macro-F1 on fine labels, AUC for binary splits, prediction-set coverage and size;
- per test source: `internal` (old marked †, it may have seen them), each external test portion, and the `HOLDOUT` set;
- **robustness** on 300 internal test images: clean, re-staining to two external labs (Reinhard to CRC-VAL and LungHist700 colour statistics; H&E, smear and cytology only), JPEG quality 30, Gaussian blur σ = 2, and 0.5× down-scaling.

**Pancreatic.** It is retrained **only if MSD Task07 Pancreas is attached** — easiest via **Stage P1** (`MMAF_CD_stageP1_msd_pancreas.ipynb`), which builds the `msd-pancreas-slices` dataset that E1/E2 find automatically; otherwise: `imagesTr/` and `labelsTr/` with `.nii.gz` files, found automatically, or set `PANC_TUMOUR_ROOT`. Slices are labelled "tumor" when the mask has 50 or more tumour pixels (up to 6 per case) and "normal" for pancreas-only slices (3 per case), plus the NIH healthy slices. Without MSD, the organ is skipped and reported as `skipped_no_tumour_labels`. **Keep the old pancreatic specialist out of every headline number either way.**

**Resumable.** A time guard stops before Kaggle's 12-hour limit. If it stops early, attach the run's output and run E2 again: finished organs load their checkpoints and are only re-evaluated. Set `E2_ORGANS = [...]` to run a subset.

**Outputs:** `e_out/spec_v25/*.pth`, `e2_results.json`, `tab_specialists_v25.tex`, `tab_robust_v25.tex` and the zip.

---

## 5. What is deliberately **not** in the new system

- **FiLM.** The Stage A audit showed its large gains (bladder, thyroid) came from the gateway's subclass head. Honest FiLM would need hints from a gateway that never saw the training images, meaning cross-fitting, which halves the data. Validated late fusion gets the useful part, a second opinion, without that cost. The FiLM attribution test stays in the paper **as an audit finding about v1**.
- **More layers or a bigger backbone.** C1 showed the failures come from data, not capacity. Internal F1 was already 0.96.
- **Stain normalisation at test time.** That was the v1 bug. Robustness now comes from training-time augmentation.

---

## 6. How to run (order, inputs, time)

| # | Notebook | Attach | GPU time |
|---|---|---|---|
| 1 | **C2** `MMAF_CD_v24_6_stageC2.ipynb` | models + training data + the mask datasets (ISIC-2018 task 1, KiTS23-2D, TN3K, DDTI, EDD2020, MoNuSeg) + `mmaf-stagea-results`; optional Kaggle secret `HF_TOKEN` for UNI2-h | 3–5 h |
| 2 | **E1** `MMAF_CD_v25_stageE1.ipynb` | models + training data + **all 20 external datasets** (same as C1) + `mmaf-stagea-results`; optional MSD pancreas; **Internet ON** | 3.5–5 h |
| 3 | **E2** `MMAF_CD_v25_stageE2.ipynb` | everything from E1 + **E1's output** (Add Input → Your Work → Notebooks → E1) | 5–7 h |

- **C2 and E1 are independent.** Run them at the same time if Kaggle lets you run two GPU sessions.
- **E2 also runs without E1** (no late fusion, table rebuilt), but with E1 attached it is faster and complete.
- **Do not run Stage D or Stage B.** D is replaced by E1 and E2. B (figures) runs once, on the final models, in Stage F.

**Pancreas data:** search Kaggle for "Task07 Pancreas" or "medical segmentation decathlon pancreas" and attach one that contains `imagesTr/pancreas_*.nii.gz` and `labelsTr/`. If none exists, pancreatic is dropped. That is the honest outcome.

**Send me:** the full log of each run, plus `mmaf_v25_E1.zip` / `mmaf_v25_E2.zip`. Without the checkpoints is fine for analysis; the results JSON and tables are what I read.

---

## 7. How to read the results: what "good" looks like

| Check | Where | Good | Bad, and what it means |
|---|---|---|---|
| New routing on external test portions | E1b | ≥ 0.95, silent misroute < 1% | Low: the source-adversary is too weak, so raise `ADV_MAX` |
| **New routing on HOLDOUT** (LungHist700, PMRAM, Rahman) | E1b | lung: 0.2% → **> 0.6**; brain and oral stay > 0.9 | Lung still ≈ 0: an unseen H&E source is still not recognised. Report it and use route sets. |
| Route-set coverage | E1b | ≈ 0.90 on in-distribution sets; \|S\| ≈ 1.0–1.3 | Coverage ≪ 0.9 on HOLDOUT is expected under shift. Say so in the paper. |
| Dataset-ID probe | E1b | new ≪ old, near chance | new ≈ old: shortcut still present |
| Specialist on external test portions | E2 | skin 0.21 → ≥ 0.55, kidney AUC 0.63 → ≥ 0.80, blood AUC 0.40 → ≥ 0.80 | No gain: the external domain is too different. Report it. |
| Specialist on HOLDOUT | E2 | lung (0.80 AUC), oral (0.995), brain (0.92) not worse | Worse: robustness augmentation cost accuracy. Tune it down for that organ. |
| Robustness | E2 | lymphoma stain 0.52 → ≥ 0.85; JPEG 30 ≥ 0.90 | Still low: raise HED sigma |
| Internal (group split) | E2 | may be **lower** than old v1 numbers | **This is the honest number.** Old internal numbers were inflated by image-level leakage. |

---

## 8. What this adds to the paper (novelty that survives review)

1. **Within-organ source-adversarial routing.** A router trained to be blind to *dataset identity inside each organ*, measured with a dataset-identification probe before and after.
2. **Set-valued routing with a coverage guarantee.** Conformal route sets send an ambiguous image to every plausible specialist and return the union of their class sets. It is the principled fix for colon/gastric-type ambiguity.
3. **Training across heterogeneous label spaces.** Set-NLL lets coarse external labels train fine-grained specialists. This is partial-label learning, applied to multi-dataset medical training.
4. **A leakage-free protocol:** group splits, cross-source de-duplication, and true holdout hospitals, with old-vs-new numbers on *identical* images.

---

## 9. After E1 + E2: Stage F (I build it once E finishes)

1. Load the V25 gateway and specialists.
2. Re-run calibration, conformal, routing (θ sweep plus route sets), the end-to-end pipeline accuracy (route → specialist → prediction set), external (C1 protocol), OOD (with CIFAR and the chest X-ray set attached this time) and XAI agreement.
3. Then Stage B figures on the final models, and I rewrite the paper's results with one consistent set of numbers.

**Estimated paper rating** with E working as intended: **≈ 7.5–8/10**. That means robust models, an honest audit and two genuinely new methods (source-blind routing and route sets).
