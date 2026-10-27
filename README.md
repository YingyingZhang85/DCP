# DCP — Disease Continuum Positioning

**Longitudinal Bayesian Learning of Continuous Disease Position across the Alzheimer's Disease Continuum**

Yingying Zhang, Kun Zhao, Guodong Liu, Qi Huang, Pengfei Gu, Dongchul Kim, Erik Enriquez, Alex D. Leow, Paul M. Thompson, Heng Huang, Hongchang Gao, Liang Zhan, Haoteng Tang, for the Alzheimer's Disease Neuroimaging Initiative (ADNI) Project

[[Paper]](https://arxiv.org/abs/2608.19436) [[GitHub]](https://github.com/YingyingZhang85/DCP)

## Highlights

- We propose **Disease Continuum Positioning (DCP)**, a longitudinal Bayesian learning framework that generates an uncertainty-aware **Disease Continuum Score (DCS)** for continuous patient positioning along the Alzheimer's disease continuum.
- DCP explicitly learns disease position beyond discrete diagnostic stages, enabling fine-grained characterization of individual disease severity.
- DCS captures clinically meaningful disease variation and provides prospective information on future CN-to-MCI and MCI-to-AD conversion.

## Method Overview

DCP learns a continuous **Disease Continuum Score (DCS)** from longitudinal diffusion tensor imaging (DTI) data. A 3D ResNet-18 encoder maps each DTI scan into a structured probabilistic latent representation, where the first latent dimension (`s̃`) is a disease-order-aware variable from which the non-negative DCS is derived via a Softplus transformation: `s = Softplus(s̃)`. The remaining dimensions (`z_r`) capture residual imaging information and are decoded to reconstruct the input.

**Key components of the learning objective:**

- **Diagnosis-anchored prior (L_KL)** — diagnosis-conditioned Gaussian priors impose population-level disease ordering on the latent disease-continuum axis, while an isotropic Gaussian prior regularizes the residual latent variables
- **Ordinal ranking loss (L_rank)** — a margin-based pairwise constraint enforcing CN < MCI < AD ordering at the subject level
- **Monotonicity loss (L_mono)** — a soft longitudinal constraint encouraging non-decreasing DCS over time within each subject, weighted by follow-up interval
- **Reconstruction loss (L_rec)** — MSE-based regularization encouraging the latent representation to preserve complementary imaging information

## Project Structure

```
DCP/
├── configs/config.py                # Central configuration (paths, hyperparameters)
├── train_engine.py                  # Main training loop
├── run_optuna.py                    # Optuna hyperparameter optimization
├── models/
│   └── resnet18.py                  # 3D ResNet-18 VAE (encoder + decoder)
├── losses/loss.py                   # Composite loss function
├── datasets/
│   ├── dataset.py                   # DTI dataset and dynamic sampler
│   ├── atlas/                       # FreeSurfer atlas for ROI ablation
│   └── regression/
│       └── nilearn_regression.py    # Normative residualization (nilearn)
├── predict_analysis/
│   ├── predict_model.py             # Inference / prediction
│   ├── evaluate_all_metrics.py      # Multi-dimensional evaluation
│   ├── volume_correlation.py        # Brain region volume correlation
│   ├── roi_ablation_adni.py         # ROI ablation with permutation test
│   └── plot/                        # Visualization scripts (correlation, trajectory, conversion, etc.)
├── submit_resnet.sh                 # SLURM submission script
├── submit_regression.sh             # SLURM script for residualization
├── submit_roi_ablation.sh           # SLURM script for ROI ablation
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/YingyingZhang85/DCP.git
cd DCP
pip install -r requirements.txt
```

## Data Preparation

This study uses diffusion-weighted MRI (dMRI) data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu) database. The dataset consists of 695 participants (380 CN, 222 MCI, 93 AD) with 1–13 longitudinal visits each, totaling 2,036 dMRI scans.

1. **Obtain DTI data** — download processed DTI images and reconstruct four diffusion tensor-derived parametric maps: fractional anisotropy (FA), mean diffusivity (MD), radial diffusivity (RD), and axial diffusivity (AD). Place them under `data/dti_residuals/`.
2. **Prepare metadata CSV** — create a CSV file with columns: `Subject (PTID)`, `Image Data ID`, `Acq Date`, `Group`, `DIAGNOSIS`, `Age`, `Sex`, `PTEDUCAT`. Place it at `datasets/csv/5_folds/dataset_5_folds_DTI.csv`.
3. **Normative residualization** — run `datasets/regression/nilearn_regression.py` to regress out the effects of age, sex, and education from the DTI data using voxel-wise OLS, with regression coefficients estimated exclusively from cognitively normal (CN) subjects.

## Usage

### Training (single fold)

```bash
export FOLD=1
python train_engine.py
```

### Full cross-validation pipeline (SLURM)

Edit `submit_resnet.sh` to set your partition and node, then:

```bash
sbatch submit_resnet.sh
```

This runs training, prediction, and evaluation for each fold across a 5-fold subject-level cross-validation.

### Hyperparameter search

```bash
python run_optuna.py
```

### Evaluation only

```bash
python predict_analysis/evaluate_all_metrics.py \
    --csv_dir outputs/<exp_name>/predictions \
    --all_folds \
    --label "DCP"
```

## Configuration

All paths and hyperparameters are centralized in `configs/config.py`. Override paths via environment variables:

| Variable | Description | Default |
|---|---|---|
| `DCP_ROOT` | Project root directory | Auto-detected |
| `DATA_ROOT` | DTI data directory | `<root>/data/dti_residuals` |
| `CSV_PATH` | Metadata CSV path | `<root>/datasets/csv/5_folds/dataset_5_folds_DTI.csv` |
| `FOLD` | Current cross-validation fold | `1` |
| `EXP_NAME` | Experiment name for outputs | `train_tr_<job_id>` |

## Evaluation Metrics

DCS is evaluated from three complementary perspectives:

| Perspective | Metric | Description |
|---|---|---|
| Clinical relevance | **Spearman ρ** | Rank correlation between DCS and MMSE |
| Stage discrimination | **ANOVA F** | Between-stage variation (CN / MCI / AD) relative to within-stage variability |
| | **Stage C** | Concordance between DCS ordering and ordinal diagnostic stages across subjects |
| Longitudinal & conversion | **Long. C** | Concordance between DCS ordering and temporal ordering within subjects |
| | **AUC (CN→MCI)** | Area under ROC for predicting future CN-to-MCI conversion |
| | **AUC (MCI→AD)** | Area under ROC for predicting future MCI-to-AD conversion |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhang2026longitudinalbayesianlearningcontinuous,
      title={Longitudinal Bayesian Learning of Continuous Disease Position across the Alzheimer's Disease Continuum}, 
      author={Yingying Zhang and Kun Zhao and Guodong Liu and Qi Huang and Pengfei Gu and Dongchul Kim and Erik Enriquez and Alex D. Leow and Paul M. Thompson and Heng Huang and Hongchang Gao and Liang Zhan and Haoteng Tang},
      year={2026},
      eprint={2608.19436},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.19436}, 
}
```

## Acknowledgment

Data used in preparation of this work were obtained from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu) database, funded by NIH grant U19AG024904. This study is partially supported by NIH (R33AG087888, R01AG092661, U01AG068057), AHA award 26AIREA1574568, and NSF (CCF 2523787, IIS 2319450, IIS 2045848). Computational resources were provided by the Bridges-2 system at the Pittsburgh Supercomputing Center (NSF OAC-1928147) and the UTRGV HPC Resource (NSF 2018900, IIS-2334389; DoD W911NF2110169).

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [LICENSE](LICENSE) for details.
