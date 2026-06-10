# Stregmentation

### Probabilistic ischemic stroke lesion localisation from DWI + ADC MRI

**Biomedical Engineering — Universitat Pompeu Fabra, 2026**
Iago Aboal · Arnau Raya · Arnau Sanfeliu

---

## Project overview

This project trains a **3D U-Net** to identify ischemic stroke lesions in brain MRI using two input sequences:

* **DWI**: highlights restricted diffusion, commonly used in acute ischemic stroke.
* **ADC**: helps confirm diffusion restriction and reduce ambiguity.

The model does not only generate a binary segmentation mask. Its main output is a **voxel-wise probability heatmap**, where each voxel receives a value between 0 and 1 indicating how likely it is to belong to ischemic tissue.

A binary mask is obtained afterwards by applying a threshold of **0.5**, but this mask is considered a secondary output.

The goal of the project is therefore **lesion localisation**, not perfect manual-style segmentation.

---

## Why a probability map?

In acute care, a single automatic binary mask can be misleading because it hides uncertainty. This is especially risky for small, subtle, or ambiguous lesions.

A probability heatmap is more informative because it shows:

* where the model is confident,
* where the model is uncertain,
* and which regions should be reviewed more carefully.

This project should be understood as a **technology study**, not as a clinical validation study. We evaluate whether the model can localise lesions accurately and quickly enough to justify further testing. We do **not** claim that it improves clinical decisions or accelerates the diagnostic workflow.

---

## Pipeline

```text
DWI + ADC MRI
     │
     ▼
Pre-processing
- resize each slice to 128 × 128
- pad or crop the depth to 80 slices
- normalise each modality to [0, 1]
     │
     ▼
3D U-Net
     │
     ▼
Sigmoid output
     │
     ├──► Probability heatmap
     │
     └──► Threshold at 0.5 → Binary mask
```

---

## Dataset

| Property         | Value                                 |
| ---------------- | ------------------------------------- |
| Dataset          | ISLES 2022                            |
| Cases            | 250 multi-centre MRI cases            |
| Inputs           | DWI + ADC                             |
| Target           | Expert binary lesion mask             |
| Split            | 70% train / 15% validation / 15% test |
| Random seed      | 67                                    |
| Final input size | 80 × 128 × 128                        |

The ISLES 2022 data are not included in this repository. They can be downloaded from the official ISLES 2022 release:

* ISLES 2022 challenge: https://isles22.grand-challenge.org/
* Zenodo dataset: https://doi.org/10.5281/zenodo.7153326

---

## Model

The model is a **3D U-Net** with two input channels, one for DWI and one for ADC.

| Component      | Description                |
| -------------- | -------------------------- |
| Architecture   | 3D U-Net                   |
| Input channels | 2                          |
| Output         | 1-channel probability map  |
| Activation     | Sigmoid                    |
| Parameters     | Approximately 90.3 million |
| Main script    | `train_cluster_adamw.py`   |

---

## Training configuration

| Hyperparameter        | Value                                |
| --------------------- | ------------------------------------ |
| Optimizer             | AdamW                                |
| Learning rate         | 1 × 10⁻⁴                             |
| Weight decay          | 5 × 10⁻⁵                             |
| LR schedule           | CosineAnnealingLR                    |
| Loss function         | 0.5 Dice loss + 0.5 Focal loss       |
| Focal loss parameters | α = 0.75, γ = 2                      |
| Epochs                | 100                                  |
| Batch size            | 2                                    |
| Seed                  | 67                                   |
| Hardware              | Single NVIDIA GPU on a SLURM cluster |
| Training time         | Approximately 34 minutes             |

The combined Dice + Focal loss was used because ischemic lesions occupy only a very small fraction of the brain volume. Dice loss helps optimise lesion overlap, while Focal loss reduces the influence of easy background voxels and focuses learning on harder lesion regions.

---

## Results

The model was evaluated on a held-out test set of **38 cases**.

### Threshold-free heatmap quality

| Metric  | Value |
| ------- | ----- |
| ROC-AUC | 0.936 |
| PR-AUC  | 0.827 |

These metrics evaluate the probability heatmap before converting it into a binary mask.

---

### Voxel-level segmentation performance

The binary mask was obtained by thresholding the probability map at **0.5**.

| Metric                | Value |
| --------------------- | ----- |
| Global Dice           | 0.808 |
| Per-patient mean Dice | 0.670 |
| Voxel precision       | 0.862 |
| Voxel recall          | 0.761 |

The difference between global Dice and per-patient mean Dice is important. Global Dice is more influenced by large lesions because they contribute more voxels. Per-patient Dice gives the same weight to every patient and therefore reveals failures in smaller or more difficult cases.

---

### Lesion-wise detection

| Metric                   | Value      |
| ------------------------ | ---------- |
| Lesion-wise precision    | 1.000      |
| Lesion-wise recall       | 0.383      |
| Mean centroid distance   | 3.4 voxels |
| Median centroid distance | 0.8 voxels |

Lesion-wise recall is the most informative metric here. It shows that the model often misses small lesions. This is one of the main limitations of the current version.

Lesion-wise precision should be interpreted with caution. The evaluation uses a large matching radius and merges predicted fragments close to the same true lesion. This can artificially reduce the number of false positives and make precision appear higher than it really is.

---

## What we can claim

This project supports the following claim:

> The model can generate a fast probability heatmap that often localises the main ischemic lesion region.

This is supported by:

* good global voxel overlap,
* high ROC-AUC and PR-AUC,
* low centroid distance in matched lesions,
* and fast inference.

---

## What we cannot claim

This project does **not** prove that the model:

* detects all ischemic lesions,
* performs well on very small lesions,
* produces formally calibrated probabilities,
* improves clinical decision-making,
* or accelerates the diagnostic workflow.

These claims would require additional clinical validation.

---

## Main limitations

### 1. Depth cropping during pre-processing

Each scan was converted to a fixed size of **80 × 128 × 128**. Height and width were resized to 128 × 128, but the depth axis was handled by padding or cropping.

This means that scans with more than 80 slices were cropped. In some cases, this may remove superior or inferior slices that contain lesion tissue.

This is a relevant pre-processing limitation and may explain some low-Dice cases, especially when the lesion is large or located in cropped regions.

A better approach would be to resize the depth axis or choose a larger target depth based on the distribution of scan sizes.

---

### 2. Small lesion detection

Small ischemic lesions are difficult to detect because they occupy very few voxels compared with the full brain volume.

In this project, lesion-wise recall was limited, especially for small lesions. This means that the model can miss punctiform or scattered infarcts.

---

### 3. Lesion-wise precision is optimistic

The lesion-wise precision value of 1.0 should not be interpreted as perfect spatial precision.

Because of the matching strategy, predicted blobs near a true lesion can be absorbed into the same matched lesion instead of being counted as false positives. Visual inspection still shows some over-segmentation and spurious predicted regions.

---

### 4. No formal calibration analysis

The sigmoid output is interpreted as a probability map, but formal calibration was not assessed. Reliability diagrams or Expected Calibration Error would be needed to evaluate whether predicted probabilities are numerically well calibrated.

---

## Repository structure

```text
stregmentation/
├── README.md
├── requirements.txt
├── Dataset.py
├── UNet_classes.py
├── Model_acts.py
├── train_cluster_adamw.py
├── train_adamw_job.sh
├── results/
│   ├── args.json
│   ├── test_results.json
│   ├── summary_stats.json
│   ├── history.json
│   ├── per_patient_results.csv
│   ├── train_idx.npy
│   ├── val_idx.npy
│   ├── test_idx.npy
│   └── figures/
└── .gitignore
```

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install torch nibabel numpy pandas scikit-learn scipy tqdm
```

---

### 2. Download the dataset

Download ISLES 2022 from Zenodo:

```text
https://doi.org/10.5281/zenodo.7153326
```

The dataset is not included in this repository.

---

### 3. Prepare the input paths

Create a CSV file containing the paths to the images and masks.

The CSV should include at least the following columns:

```text
dwi_path
adc_path
mask_path
```

Each row should correspond to one patient.

---

### 4. Train the model

Example command:

```bash
python train_cluster_adamw.py \
  --base_path /path/to/ISLES-2022/ \
  --csv paths_isles_2022.csv \
  --modalities dwi_path adc_path \
  --image_size 80 128 128 \
  --features 64 128 256 512 \
  --epochs 100 \
  --batch_size 2 \
  --lr 1e-4 \
  --weight_decay 5e-5 \
  --output_dir ./runs/experiment_1 \
  --seed 67
```

---

## Reproducibility

The exact train, validation, and test split indices are stored in:

```text
results/train_idx.npy
results/val_idx.npy
results/test_idx.npy
```

The training seed was fixed to **67**.

---

## Reference

Hernandez Petzsche MR, et al.
**ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.**
*Scientific Data.* 2022;9:762.
https://doi.org/10.1038/s41597-022-01875-5
