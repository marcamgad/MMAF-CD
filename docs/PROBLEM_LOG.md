# Engineering problem log

Every problem met during the project, from the first implementation to the final system.

Severity: 🔴 **Critical** = wrong or unsafe result without a visible error · 🟠 **High** = major loss of accuracy or validity · 🟡 **Medium** = limited loss, or caught before it affected results · 🟢 **Low** = cosmetic or efficiency.


## Building the system

| # | Problem | Severity | Effect | How it was solved |
|---|---|---|---|---|
| 1 | Padded Level-2 logits (−∞) with label smoothing | 🔴 Critical | Level-2 loss ≈5×10⁶, organ head corrupted | Per-organ subclass loss without padding |
| 2 | Class indices stored during the file walk | 🔴 Critical | 56% subclass accuracy from index–label mismatch | Store names; map to indices after sorting |
| 3 | `RandomErasing` applied to PIL images | 🟢 Low | Augmentation silently skipped | Moved after `ToTensor()` |
| 4 | AMP scaling in the XAI backward pass | 🟠 High | Blank first-row maps | Autocast off for XAI; warm-up forward pass |
| 5 | EigenCAM SVD sign ambiguity | 🟡 Medium | Background highlighted on some calls | Sign fixed by eigen-decomposition / alignment with the mean activation |
| 6 | Hook on non-existent `stages.3` (EfficientViT) | 🟠 High | Zero-valued maps | Safe layer discovery |
| 7 | AMP overflow in EfficientViT LayerNorm | 🟠 High | NaN confidences (cervical) | `nan_to_num`, logit clamp; backbone later replaced |
| 8 | ViT tokens not reshaped | 🟠 High | Zero maps for ViT specialists | Strip CLS token, reshape to 14×14 |
| 9 | `x or y` on NumPy arrays | 🟡 Medium | `ValueError`, panel crash | Explicit `is not None` test |
| 10 | SHAP ≥0.42 returns a 5D array | 🟠 High | 100% blank SHAP panels | Index by predicted class; SHAP later removed |
| 11 | Boolean mask built on CPU, applied to GPU tensor | 🟠 High | AUC-Deletion = 0.000 for every image | Deletion loop on CPU; later replaced (No. 28) |
| 12 | Non-contiguous arrays passed to `pydensecrf` | 🟠 High | CRF failed on all samples while reporting success | `np.ascontiguousarray`; truthful flag |
| 13 | HoVer-Net BCE on probabilities under AMP | 🟡 Medium | Unstable training | `binary_cross_entropy_with_logits` |
| 14 | HoVer-Net labelled most nuclei neoplastic | 🟡 Medium | `seg_net_used` 0% | Confidence gate 0.55 (workaround); model later replaced (No. 46) |
| 15 | Keyword collision in `advanced_cam_ensemble` | 🟠 High | Recursion error for 9 of 16 organs; silent fallback | Documented; ensemble replaced by benchmark selection |

## Testing the submitted system

| # | Problem | Severity | Effect | How it was solved |
|---|---|---|---|---|
| 16 | H&E stain normaliser applied to every test image | 🔴 Critical | −7.7 macro-F1, −12.7 routing points; lymphoma ``collapse'' | Removed; stain augmentation in training |
| 17 | Image-level split of an augmented re-upload | 🔴 Critical | 55–78% test near-duplicates in six organs | D8 pHash groups, group split |
| 18 | Pancreatic labels per patient on whole-body slices | 🔴 Critical | Specificity 0.000 on healthy pancreas | MSD slice labels, pancreas-centred crops |
| 19 | FiLM read the gateway's hint | 🔴 Critical | Bladder/thyroid gains not the specialist's | FiLM removed; validated late fusion |
| 20 | One source per organ | 🔴 Critical | Routing 0–3% on new colon, lung, blood data | Multi-source training; guard |
| 21 | CAM agreement on same-layer maps, outcome-selected fallback | 🟠 High | Reported ρ = 0.56; independent \|ρ\| = 0.17 | Agreement dropped as a metric; faithfulness benchmark |
| 22 | Grad-CAM++ formula error, score ×100 | 🔴 Critical | Map faithful for no organ | Corrected formula; per-organ benchmark selection |
| 23 | Hook failures (lymphoma, pancreatic flat maps; kidney, thyroid empty CAMs) | 🟠 High | No explanation for four organs | One backbone family; hook-free implementation |
| 24 | Trust score never tested as an error detector | 🟠 High | AUROC 0.826 < confidence 0.834; focus term 0.437 | Validated reliability layer |
| 25 | CAM masks presented as lesion outlines | 🔴 Critical | Dice 0.06–0.22 against true lesions | MedSAM only where validated (skin, thyroid) |
| 26 | Stages and treatments derived from image classes | 🔴 Critical | Unsafe clinical text | Curated table: next step and staging system only |
| 27 | Projected values reported as results (≥ values) | 🔴 Critical | Unmeasured gains in tables | Withdrawn; replaced by measured results |
| 28 | AUC-Deletion as an unnormalised logit integral | 🟠 High | 4.8164 compared with 0.681; negative values | Probability curves vs random, Wilcoxon test |
| 29 | Single routing threshold stated, two deployed | 🟡 Medium | Misrouting rate unmeasured | Full-split sweep |

## Rebuilding the system

| # | Problem | Severity | Effect | How it was solved |
|---|---|---|---|---|
| 30 | Source-adversarial head did not remove source identity | 🟠 High | LungHist700 still 0% routed | Kept as negative result; guard uses the residual signal |
| 31 | Conformal route sets degenerate (q̂ = 0.003) | 🟢 Low | Every route set a singleton | Not used; guard carries abstention |
| 32 | Data search 5 levels deep through all inputs | 🟡 Medium | 66 minutes lost | Search small inputs only, depth ≤4 |
| 33 | Robustness test looked for a removed pancreatic test set | 🟡 Medium | Crash after pancreatic training (caught in a mock run) | Fall back to the largest test source |
| 34 | Whole-slice pancreatic CT | 🟠 High | AUC 0.64 | Pancreas-centred crops, fixed orientation (AUC 0.858) |
| 35 | 28 attached datasets; one failed to mount | 🟡 Medium | Notebook did not start | Standalone notebooks with lazy checks and path remapping |
| 36 | KiTS23 folder search | 🟢 Low | 18 minutes | Glob literal `images/`, `masks/` |
| 37 | No validation images for an organ (mock run) | 🟡 Medium | `KeyError` in threshold fit | Pooled fallback, flagged ``uncalibrated'' |
| 38 | Validation review rate ignored the singleton rule | 🟡 Medium | Review volume under-stated | Full accept rule |
| 39 | Heat-map rule labelled negative drops ``weak'' | 🟢 Low | Misleading wording | Positive drop required; signed points |
| 40 | Saturated specialists (p ≈ 1.00) | 🟡 Medium | Degenerate τ_p; inconclusive checks | Open: logit-margin thresholds |
| 41 | Underscores in FHIR ids | 🟢 Low | Validation failure | Replace with hyphens; 32/32 valid |
| 42 | Indentation error when inserting code into the nuclei runner | 🟢 Low | Cell crash | De-indent the template before insertion |
| 43 | MoNuSeg masks read with the wrong background value | 🟠 High | PQ = 0 for every tile | Background = most frequent value; checked 42.2 nuclei per tile |
| 44 | TIAToolbox 2.x API change; ``MPP is None'' | 🟡 Medium | HoVer-Net would not run | Call the network directly on 256×256 windows |
| 45 | `DataParallel` hides `_proc_np_hv` | 🟢 Low | Post-processing crash | Unwrap `.module`; call the class method |
| 46 | Our HoVer-Net finds half of the nuclei | 🟠 High | MoNuSeg PQ 0.281 vs 0.372 | Replaced by the pretrained model |
| 47 | Guard blind to text-like pages | 🟡 Medium | 0% flagged | Open: input-quality check |

**Lessons.** (1) The most damaging problems raised no error. (2) They were found by comparisons: with and without a step, against the conditioning signal alone, against a random map, against ground truth, and on data from another hospital. (3) Silent numerical corruption needs explicit checks that outputs are not empty or constant.
