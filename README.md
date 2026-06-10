# stroke-segmentation-isles22

3D U-Net probabilistic stroke lesion segmentation from DWI + ADC MRI (ISLES 2022)

\*\*Biomedical Engineering — Universitat Pompeu Fabra, 2026\*\*  

Iago Aboal · Arnau Raya · Arnau Sanfeliu



\---



\## What this project does



We train a 3D U-Net on the \[ISLES 2022](https://isles22.grand-challenge.org/) dataset to generate a \*\*voxel-wise probability heatmap\*\* of likely ischemic tissue from two MRI sequences routinely acquired in acute stroke: DWI and ADC.



The intended output is not a binary answer. It is a graded probability map — one value per voxel — that highlights candidate ischemic regions while keeping uncertainty visible. A thresholded binary mask (τ = 0.5) is derived from it as a secondary product.



\*\*This is a technology study, not a clinical trial.\*\* We measure whether the map is accurate and fast enough to be worth testing in a clinical workflow. We do not test whether it actually helps clinicians.



\---



\## Pipeline



```

DWI volume  ┐

&#x20;           ├──► Pre-processing ──► 3D U-Net ──► Sigmoid ──► Probability heatmap

ADC volume  ┘    (resize H×W,              (90.3M params)       │

&#x20;                 pad/crop D→80,                                 └──► Threshold 0.5 ──► Binary mask

&#x20;                 min-max norm)

```



\---



\## Dataset



| Property | Value |

|---|---|

| Source | \[ISLES 2022](https://doi.org/10.1038/s41597-022-01875-5) — publicly released training partition |

| Cases | 250 multi-centre MRI cases |

| Input channels | DWI (b≈1000) + ADC map |

| Target | Expert binary lesion mask |

| Split | 70 / 15 / 15 % — train / val / test (seed 67) |

| Input size | 80 × 128 × 128 (D × H × W) |



The data are not included in this repository. Download from \[Zenodo](https://doi.org/10.5281/zenodo.7153326).



\---



\## Results (test set, 38 cases)



\### Heatmap quality (threshold-free)



| Metric | Value |

|---|---|

| ROC-AUC | 0.936 |

| PR-AUC | 0.827 |



\### Voxel overlap (mask, τ = 0.5)



| Metric | Value |

|---|---|

| Global (pooled) Dice | 0.808 |

| Per-patient mean Dice | 0.670 |

| Voxel precision | 0.862 |

| Voxel recall | 0.761 |



> The gap between pooled Dice (0.808) and per-patient mean Dice (0.670) is itself a result: the pooled metric is dominated by large lesions, while the per-patient mean weights every case equally and exposes small-lesion failures.



\### Lesion-wise detection



| Metric | Value |

|---|---|

| Lesion-wise precision | 1.000 |

| Lesion-wise recall | 0.383 |

| Mean centroid distance | 3.4 voxels |

| Median centroid distance | 0.8 voxels |



>Lesion-wise precision of 1.0 is an artefact of the matching algorithm (50-voxel radius + fragment merging absorbs nearby predicted blobs into matched lesions rather than counting them as false positives). It should be read as optimistic, not as evidence of perfect spatial precision. Visual inspection confirms over-segmentation in several cases.



\## Known limitations



\*\*Depth crop (pre-processing bug).\*\* H and W are resized by interpolation to 128×128, but the depth axis is handled by zero-padding shorter volumes and \*\*hard-cropping\*\* volumes deeper than 80 slices (`\[:, :, :80, :, :]`). Scans acquired with thin slices (\~2 mm) can exceed 80 slices and have their superior/inferior extent silently discarded — including any lesion tissue in those slices. A correct implementation would resize (not crop) the depth axis. This is a fixable engineering error and is the most likely explanation for `sub-strokecase0140` (large lesion, low Dice).



\*\*Small lesion detection.\*\* Lesion-wise recall is 0.38 and decreases with lesion size. Punctiform embolic infarcts (<200 voxels) are frequently missed. This is a known difficulty in the ISLES 2022 dataset and in medical image segmentation generally.



\*\*Lesion-wise precision metric.\*\* As noted above, the matching radius + fragment merging produces an inflated precision value. A stricter IoU-based matching would give a more honest estimate.



\*\*No formal probability calibration.\*\* The sigmoid output is interpreted as a probability, but calibration (reliability diagrams, ECE) was not measured. Miscalibrated confidence values would reduce clinical interpretability.



\---



\## Training configuration



| Hyperparameter | Value |

|---|---|

| Optimizer | AdamW |

| Learning rate | 1 × 10⁻⁴ |

| Weight decay | 5 × 10⁻⁵ |

| LR schedule | CosineAnnealingLR (η\_min = lr × 0.01) |

| Loss | 0.5 × Dice + 0.5 × Focal (α=0.75, γ=2) |

| Epochs | 100 |

| Batch size | 2 |

| GPU | 1 × NVIDIA (SLURM cluster) |

| Training time | \~34 min (15:39 → 16:13, May 31 2026) |

| Seed | 67 |



\---



\## Repository structure



```

stregmentation/

├── Dataset.py               # Dataset classes (Dataset\_v1, Dataset\_combined)

├── UNet\_classes.py          # 3D U-Net architecture

├── Model\_acts.py            # Training loop, evaluation, lesion-wise metrics

├── train\_cluster\_adamw.py   # Main training script (cluster / CLI)

├── train\_adamw\_job.sh       # SLURM job submission script

├── requirements.txt

├── results/

│   ├── args.json            # Exact hyperparameters used

│   ├── test\_results.json    # Test set metrics

│   ├── summary\_stats.json   

│   ├── history.json         # Per-epoch train/val loss curves

│   ├── per\_patient\_results.csv

│   ├── train\_idx.npy        # Reproducibility: exact split indices

│   ├── val\_idx.npy

│   ├── test\_idx.npy

│   └── figures/

└── README.md

```



\---



\## How to run



\*\*1. Install dependencies\*\*

```bash

pip install torch nibabel numpy pandas scikit-learn scipy tqdm

```



\*\*2. Prepare the data\*\*  

Download ISLES 2022 from \[Zenodo](https://doi.org/10.5281/zenodo.7153326) and create a CSV with columns `dwi\_path`, `adc\_path`, `mask\_path` pointing to the NIfTI files.



\*\*3. Train\*\*

```bash

python train\_cluster\_adamw.py \\

&#x20; --base\_path /path/to/ISLES-2022/ \\

&#x20; --csv paths\_isles\_2022.csv \\

&#x20; --modalities dwi\_path adc\_path \\

&#x20; --image\_size 80 128 128 \\

&#x20; --features 64 128 256 512 \\

&#x20; --epochs 100 \\

&#x20; --batch\_size 2 \\

&#x20; --lr 1e-4 \\

&#x20; --weight\_decay 5e-5 \\

&#x20; --output\_dir ./runs/experiment\_1 \\

&#x20; --seed 67

```



\*\*4. Reproduce our exact split\*\*  

The index files `train\_idx.npy`, `val\_idx.npy`, `test\_idx.npy` in `results/` contain the exact indices used. Pass `--seed 67` to reproduce the split from scratch.



\---



\## Reference



Hernandez Petzsche MR, et al. \*ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.\* Scientific Data. 2022;9:762. https://doi.org/10.1038/s41597-022-01875-5



