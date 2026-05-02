# 🫀 CardioVision AI

### A Multimodal Vision Transformer Framework for Cardiovascular Event Risk Assessment

> **Publication-grade deep learning system combining coronary angiography with clinical tabular data to classify stenosis severity and estimate 10-year cardiovascular risk.**

**Institution:** Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore  
**Course:** 22AIE499 – Project Phase 2 | Group 11 – Batch B  
**Guide:** Dr. Shakti Prasad Sethi

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Gandarapu Ajay Kumar | CB.EN.U4AIE22120 |
| M. Yashwanth Venkat Chowdary | CB.EN.U4AIE22131 |
| Nagabandi Sampreeth | CB.EN.U4AIE22141 |
| Pusala Sai Santhan | CB.EN.U4AIE22146 |

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Datasets](#-datasets)
- [Repository Structure](#-repository-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Models & Training Strategy](#-models--training-strategy)
- [Explainability Pipeline](#-explainability-pipeline)
- [Results & Ablation Study](#-results--ablation-study)
- [Clinical Modules](#-clinical-modules)
- [Streamlit Web App](#-streamlit-web-app)
- [Publication Experiments](#-publication-experiments)
- [Installation & Usage](#-installation--usage)
- [Dependencies](#-dependencies)
- [Citation](#-citation)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

CardioVision AI addresses a core limitation in automated cardiovascular AI: most systems treat angiography images in isolation, ignoring everything clinically known about the patient. This project fuses two independent data modalities — coronary angiography frames and population-level clinical risk features — through a learned adaptive gate, producing stenosis severity predictions that reflect both visual evidence and patient risk burden simultaneously.

**Key contributions:**

- A cross-dataset multimodal fusion architecture pairing ViT-Base/16 with a 4-layer clinical MLP, combined via a compact learned FusionGate (224 parameters) for dynamic per-sample modality weighting
- A class-conditional clinical prior construction strategy that bridges NHANES and CADICA without requiring any shared patient identifiers
- DINO self-supervised pretraining on unlabelled ARCADE angiograms to domain-adapt the ViT backbone before supervised fine-tuning
- A blended Grad-CAM++ + SmoothGrad explainability pipeline with a 3-level bounding box fallback guaranteeing localisation on every input
- A post-training logit debiasing scheme correcting minority-class amplification from Focal Loss — applied at inference time, no retraining needed
- A standalone validated 2013 ACC/AHA Pooled Cohort Equations ASCVD calculator
- A six-variant ablation study with bootstrap confidence intervals (n=10,000) and McNemar significance testing
- A four-tab Streamlit application for clinical and academic evaluation

---

## 🏗 Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       CardioVision AI                            │
│                                                                  │
│  ┌─────────────────────┐      ┌──────────────────────────────┐  │
│  │    IMAGE BRANCH      │      │      CLINICAL BRANCH          │  │
│  │                     │      │                              │  │
│  │  Coronary Angiogram  │      │  NHANES Clinical Vector      │  │
│  │  224×224 RGB         │      │  (22-dim, z-scored)          │  │
│  │        ↓             │      │         ↓                    │  │
│  │  ViT-Base/16         │      │  4-Layer Clinical MLP        │  │
│  │  (ImageNet-21k or    │      │  256 → 128 → 64 → 3          │  │
│  │   DINO checkpoint)   │      │  BatchNorm + GELU            │  │
│  │  12 blocks, 768-CLS  │      │  Dropout 0.4 / 0.3           │  │
│  │        ↓             │      │         ↓                    │  │
│  │  Image Head          │      │  Clinical Logits [3-dim]     │  │
│  │  LayerNorm→256→GELU  │      │                              │  │
│  │  Dropout(0.3)→Lin(3) │      └───────────────┬──────────────┘  │
│  │        ↓             │                      │                 │
│  │  Image Logits [3]    │                      │                 │
│  └──────────┬───────────┘                      │                 │
│             └──────────────┬───────────────────┘                 │
│                            ▼                                     │
│                      FusionGate                                  │
│              Concat [6-dim] → Linear(32) → GELU                  │
│              Linear(3) → Sigmoid  →  gate α ∈ (0,1)³            │
│                            ↓                                     │
│              fused = α · img + (1−α) · clin                      │
│                            ↓                                     │
│              Log-Prior Debiasing: subtract [0.0, 1.099, 1.609]  │
│                            ↓                                     │
│                    Softmax → Prediction                          │
│                Normal │ Mild │ Severe                            │
└──────────────────────────────────────────────────────────────────┘
```

**Cross-dataset fusion rationale:** No NHANES respondent appears in any CADICA angiogram. The clinical branch is bridged through shared clinical semantics — an XGBoost model trained on NHANES maps cardiovascular risk features onto the same 3-class severity scale as CADICA, generating class-conditional clinical representations used during training. At deployment, the patient's own clinical measurements replace the population sample. This is feature-distribution-level fusion with no patient-level linkage required.

---

## 📦 Datasets

| Dataset | Modality | Role | Size |
|---------|----------|------|------|
| **[CADICA](https://www.uco.es/investigacion/proyectos/cadica/)** | Coronary angiography frames | Primary supervised classification | 22,761 frames total |
| **[NHANES](https://www.cdc.gov/nchs/nhanes/)** | Clinical & lifestyle tabular (XPT) | Population-level CAD risk prior | 11,933 participants |
| **[ARCADE](https://arcade.grand-challenge.org/)** | Coronary angiography images | DINO self-supervised pretraining | Unlabelled |

### CADICA Label Mapping

| Raw Category | Class | Clinical Meaning |
|---|---|---|
| `p0`, `p0_20` | **Normal (0)** | 0–20% narrowing |
| `p20`, `p20_35`, `p35`, `p35_60`, `p50`, `p0_50`, `p20_50` | **Mild (1)** | 20–60% narrowing |
| `p60`, `p60_100`, `p50_70`, `p50_100`, `p70`, `p70_100`, `p100` | **Severe (2)** | ≥60% narrowing (revascularisation threshold) |

**Class distribution (post-filtering):** Normal: 5,984 | Mild: 2,472 | Severe: 2,960  
**Split strategy:** Patient-level split (70 / 15 / 15) — all frames from a single patient stay in one partition to prevent data leakage.

### NHANES Feature Set (22 dimensions)

Laboratory measurements (total cholesterol, HDL, HbA1c, hs-CRP, HOMA-IR, triglycerides, fasting glucose), blood pressure and medication, lifestyle variables (smoking, activity, diet), and derived risk indicators (BMI, non-HDL, triglyceride-to-HDL ratio, diabetes/hypertension flags, age, sex, ethnicity).

---

## 📁 Repository Structure

```
CardioVision-AI/
│
├── preprocessingx.ipynb                         # Full data preprocessing pipeline
├── MAINX.ipynb                                  # ResNet-18 baseline + ResNet Fusion
├── vit_fusion_advanced.ipynb                    # ViT-Base Fusion (CADICA + NHANES)
├── vit_fusion_triple_CADICA_NHANES_ARCADE.ipynb # Triple Fusion (DINO-init + ARCADE)
├── ARCADE_DINO_Pretraining.ipynb                # DINO self-supervised pretraining
├── publication_experiments.ipynb               # Ablation, cross-val, stats, figures
│
├── appxx.py                                     # Streamlit web application (4 tabs)
├── llm_integration.py                           # Guideline-based recommendation engine
├── risk_calculator.py                           # ASCVD 10-year risk calculator
├── risk_calculator_cli.py                       # CLI interface for ASCVD calculator
│
├── processed_data/                              # ← generated by preprocessingx.ipynb
│   ├── CADICA/
│   │   ├── train/    *.npz shards
│   │   ├── val/      *.npz shards
│   │   └── test/     *.npz shards
│   └── NHANES/
│       ├── NHANES_clean.csv
│       ├── nhanes_xgb.pkl
│       └── nhanes_scaler.pkl
│
├── results/                                     # ← generated by training notebooks
│   ├── vit_experiments/
│   ├── ablation/
│   ├── cross_validation/
│   └── pub_figures/
│
└── dino_results/                                # ← generated by ARCADE_DINO_Pretraining
    ├── models/
    │   └── dino_checkpoint_final.pth
    ├── embeddings/
    └── plots/
```

> `processed_data/`, `results/`, and `dino_results/` are generated locally and are **not** committed to the repository.

---

## 🔄 Pipeline Walkthrough

### Step 1 — Preprocessing (`preprocessingx.ipynb`)

**CADICA:** Validates angiography frames (rejects high-saturation, RGB-mismatch, low-contrast non-X-ray inputs), resizes to 224×224, applies ImageNet normalization, maps raw stenosis percentile labels to 3 classes, and saves memory-efficient `.npz` shards with a patient-level split.

**NHANES:** Reads XPT survey files, engineers all 22 cardiovascular risk features, imputes missing values with column medians, enforces clinical value ranges, fits a `StandardScaler` on the training split only, trains an XGBoost classifier to map risk factors to the 3-class severity scale, and exports `nhanes_scaler.pkl`, `nhanes_xgb.pkl`, and `NHANES_clean.csv`.

**ARCADE:** Applies CLAHE contrast enhancement to raw angiogram images for DINO pretraining compatibility.

---

### Step 2 — ARCADE DINO Pretraining (`ARCADE_DINO_Pretraining.ipynb`)

Implements **DINO (Self-DIstillation with NO labels)** to domain-adapt a ViT backbone on unlabelled coronary angiograms before any supervised training — mirroring how clinical trainees learn visual structure before receiving formal diagnoses.

- Student/teacher DINO framework with ViT-Small/16 backbone
- Tuned for consumer GPU (RTX 3050, 4GB VRAM)
- Only image files used — ARCADE mask labels intentionally ignored (pure SSL)
- Outputs `dino_results/models/dino_checkpoint_final.pth`, used to initialise the Triple Fusion model in Step 5
- Also saves embeddings and plots for visual analysis of learned representations

---

### Step 3 — Baseline + ResNet Fusion (`MAINX.ipynb`)

**CNN Baseline (V1):** ResNet-18 fine-tuned on CADICA only. Focal Loss (γ=2), class weights [1.0, 2.5, 4.5], WeightedRandomSampler, OneCycleLR.

**ResNet Fusion (V4):** Adds a 4-layer Clinical MLP + FusionGate. During training, each CADICA frame with label k is paired with a NHANES feature vector sampled from XGBoost-predicted class-k pool — the cross-dataset class-conditional prior. Auxiliary cross-entropy losses on both branches (λ_img=0.3, λ_clin=0.2) prevent either from collapsing into passivity.

---

### Step 4 — ViT Fusion (`vit_fusion_advanced.ipynb`)

Replaces ResNet with **ViT-Base/16** (ImageNet-21k, via `timm`):

- Backbone frozen epochs 1–5; clinical branch and gate stabilise first
- CosineAnnealingWarmRestarts LR schedule (T₀=10, η_min=10⁻⁶)
- Mixed-precision training (FP16 + GradScaler) and gradient clipping
- **Post-training logit debiasing:** Subtracts correction vector [0.0, 1.099, 1.609] from raw logits before softmax at inference — corrects Focal Loss minority-class amplification with zero extra data and no model retraining

---

### Step 5 — Triple Fusion (`vit_fusion_triple_CADICA_NHANES_ARCADE.ipynb`)

Initialises the ViT backbone from the DINO checkpoint (Step 2) instead of ImageNet weights, then incorporates ARCADE as an auxiliary supervised signal:

- ARCADE images receive pseudo-labels from the intermediate ViT model
- Trains simultaneously on CADICA (labeled) + ARCADE (pseudo-labeled) + NHANES (clinical priors)
- Produces the best overall performance across all reported metrics

---

### Step 6 — Publication Experiments (`publication_experiments.ipynb`)

Comprehensive evaluation suite for peer review:

| Phase | Contents |
|-------|----------|
| **B** | Systematic 6-variant ablation study (all variants, identical splits) |
| **D** | Publication-quality GradCAM saliency maps for ResNet and ViT variants |
| **E** | Monte Carlo Dropout uncertainty quantification (T stochastic passes) |
| **F** | Model calibration: reliability diagrams, Expected Calibration Error |
| **G** | Bootstrap 95% CIs (n=10,000) + McNemar significance testing |
| **H** | ASCVD ↔ Stenosis cross-module clinical concordance analysis |
| **I** | Publication-quality figure generation (IEEE JBHI / MICCAI standard) |

---

## 🤖 Models & Training Strategy

| Component | CNN Baseline | ResNet Fusion | ViT Fusion (Final) |
|-----------|-------------|---------------|-------------------|
| Vision backbone | ResNet-18 | ResNet-18 | ViT-Base/16 |
| Pre-training | ImageNet-21k | ImageNet-21k | ImageNet-21k (or DINO) |
| Clinical branch | None | 4-layer MLP | 4-layer MLP |
| Fusion mechanism | None | Learned FusionGate | Learned FusionGate |
| Loss | Focal (γ=2) | Focal + aux CE | Focal + aux CE |
| Class weights | [1.0, 2.5, 4.5] | [1.0, 2.5, 4.5] | [1.0, 3.0, 5.0] |
| Optimizer | AdamW | AdamW | AdamW |
| LR schedule | OneCycleLR | OneCycleLR | CosineAnnealingWarmRestarts |
| Warmup (frozen) | Epochs 1–5 | Epochs 1–5 | Epochs 1–5 |
| Debiasing | None | None | [0.0, 1.099, 1.609] |
| Explainability | None | Limited | Grad-CAM++ + SmoothGrad |

---

## 🔬 Explainability Pipeline

Every prediction is accompanied by a class-specific saliency overlay — not a generic attention map.

**Why not Attention Rollout:** Rollout is class-agnostic. It highlights the full visible vasculature regardless of prediction class. A Severe and a Normal prediction on two different angiograms can yield visually indistinguishable rollout maps — useless for clinical localisation.

```
Angiogram Image
      ↓
Grad-CAM++ (last ViT block, 2nd-order gradients → 14×14 → bicubic 224×224)
      +
SmoothGrad (16 noisy passes, σ=8% of value range, averaged absolute gradients)
      ↓
Blended Heatmap:  0.7 × GradCAM++ + 0.3 × SmoothGrad
      ↓
Bilateral filter (edge-preserving smoothing) + Gamma correction (γ=1.5)
      ↓
3-Level Bounding Box Fallback:
  Level 1: contours exceeding 90th-percentile saliency threshold
  Level 2: relax to 75th percentile if Level 1 finds nothing
  Level 3: fixed box at global saliency maximum (cannot fail)
      ↓
Colour-coded output: green=Normal | amber=Mild | red=Severe
```

Contours occupying >35% of image area are rejected as anatomically implausible. Best contour selected by highest mean saliency score within boundary, not by size.

---

## 📊 Results & Ablation Study

### Six-Variant Ablation

| Variant | Description | Accuracy | Weighted F1 | Macro F1 | Severe F1 |
|---------|-------------|----------|-------------|----------|-----------|
| V1 | CNN only (ResNet-18, image only) | 26.25% | 0.154 | 0.197 | 0.383 |
| V2 | Clinical MLP only, no images | — | 0.906 | 0.913 | — |
| V3 | ResNet + concatenation fusion, no gate | — | 0.900 | 0.907 | — |
| V4 | ResNet-Fusion with learned FusionGate | 83.80% | 0.832 | 0.791 | 0.776 |
| V5 | V4 with clinical input zeroed at test time | 20.55% | 0.177 | 0.177 | — |
| **V6** | **ViT-Fusion (proposed)** | **94.22%** | **0.942** | **0.932** | **0.938** |

**Key insights:**
- The V4→V5 drop (63 percentage points when clinical inputs are zeroed) proves the FusionGate actively routes clinical information — it is not decorative
- The V4→V6 gain (+10.4pp accuracy) reflects ViT's structural advantage for relational spatial reasoning across coronary anatomy
- V2 (clinical only) outperforms V1 (image only) substantially — confirming clinical data carries strong standalone predictive signal
- Statistical validation: Bootstrap 95% CIs (n=10,000) + McNemar test p<0.05 confirms the ViT and ResNet backbones produce qualitatively different error structures, ruling out sampling variability as the source of the performance gap

### Classification Reports

**ResNet Fusion:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Normal | 0.9735 | 0.9727 | 0.9731 | 1172 |
| Mild | 0.7791 | 0.5206 | 0.6242 | 630 |
| Severe | 0.6797 | 0.9041 | 0.7760 | 636 |
| **Macro avg** | 0.8108 | 0.7991 | **0.7911** | 2438 |

**ViT Fusion:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Normal | 0.9728 | 0.9761 | 0.9744 | 1172 |
| Mild | 0.9123 | 0.8587 | 0.8847 | 630 |
| Severe | 0.9148 | 0.9623 | 0.9379 | 636 |
| **Macro avg** | 0.9333 | 0.9324 | **0.9324** | 2438 |

---

## 🏥 Clinical Modules

### `llm_integration.py` — Guideline-Based Recommendation Engine

Converts model predictions into structured clinical guidance with no API key, no internet, and no cost. Fully deterministic, grounded in published ACC/AHA guidelines.

**Input:** Predicted class (0/1/2) + confidence scores  
**Output:**

| Field | Contents | Reference |
|-------|----------|-----------|
| `diagnosis_explanation` | Predicted class, confidence breakdown, clinical meaning | — |
| `lifestyle_modifications` | Diet, exercise, BMI, sodium, smoking cessation | ACC/AHA Sec 3.2 / 7.3 |
| `medication_guidance` | Statin intensity, dosing, LDL targets | ACC/AHA Sec 5.3 / 5.4 |
| `testing_recommendations` | CAC scoring, hs-CRP, ECG, lipid panels | ACC/AHA Sec 6.2 / 6.3 |
| `followup_schedule` | Follow-up intervals, referral triggers | ACC/AHA Sec 8.1–8.3 |

---

### `risk_calculator.py` — ASCVD 10-Year Risk Calculator

Implements the **2013 ACC/AHA Pooled Cohort Equations** with four sex/race-stratified coefficient sets. Fully self-contained, no external dependencies.

**Required inputs:** Age (40–79), Sex, Race, Total Cholesterol, HDL, Systolic BP, BP medication, Diabetes, Smoking  
**Optional inputs:** BMI, HbA1c, hs-CRP (for risk enhancer detection)

**Risk tiers:**

| Category | 10-Year ASCVD Risk |
|----------|--------------------|
| Low | < 5% |
| Borderline | 5–7.5% |
| Intermediate | 7.5–20% |
| High | ≥ 20% |

**Why pair this with stenosis classification:** The two outputs address different clinical time horizons. Stenosis classification describes current arterial state — atherosclerotic burden already present. The ASCVD score projects the patient's likely event trajectory over the coming decade. A patient graded Normal today but with a High ASCVD score is still a strong candidate for early lipid-lowering therapy — something image-only systems cannot surface.

---

### `risk_calculator_cli.py` — Command-Line Calculator

```bash
python risk_calculator_cli.py
```

---

## 🖥 Streamlit Web App (`appxx.py`)

| Tab | Function |
|-----|----------|
| **1 — Angiogram Analysis** | Upload → 5-rule input validation → ViT image-branch inference → confidence bars + blended saliency overlay with class-coloured bounding box + temperature slider (1.0–3.0) + raw logit debug expander |
| **2 — 10-Year ASCVD Risk** | Patient demographics + labs → ACC/AHA Pooled Cohort Equations → risk %, tier, enhancer list, clinical interpretation |
| **3 — Multimodal Fusion** | Image + patient values → full ViT + MLP + FusionGate forward pass → side-by-side fused / image-only / clinical-only probability distributions + ASCVD score |
| **4 — About** | Architecture, datasets, cross-dataset fusion methodology, limitations, research-only disclaimer |

```bash
streamlit run appxx.py
```

**Required files before launch:**
```
nhanes_scaler.pkl          # from preprocessingx.ipynb
nhanes_xgb.pkl             # from preprocessingx.ipynb
<model_checkpoint>.pth     # from any training notebook
```

The app hard-fails if the `.pth` checkpoint is missing — no random-weight fallback by design.

---

## 📐 Publication Experiments (`publication_experiments.ipynb`)

Produces all results required for peer-reviewed submission (IEEE JBHI / MICCAI / CMPB):

- **Ablation** (Phase B): All 6 variants on identical test splits
- **GradCAM** (Phase D): Publication-quality saliency maps, ResNet and ViT
- **MC Dropout** (Phase E): Bayesian uncertainty via T stochastic forward passes; flags unreliable predictions clinically
- **Calibration** (Phase F): Reliability diagrams and ECE — identifies and corrects overconfidence
- **Statistical testing** (Phase G): Bootstrap 95% CIs (n=10,000) + McNemar pairwise model comparison
- **ASCVD ↔ Stenosis concordance** (Phase H): Novel clinical contribution — cross-module analysis quantifying how well the imaging and clinical modules agree across patient subgroups
- **Publication figures** (Phase I): All plots formatted for journal submission

---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/cardiovision-ai.git
cd cardiovision-ai
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Run the full pipeline in order

```
1. preprocessingx.ipynb
2. ARCADE_DINO_Pretraining.ipynb          (optional — only needed for Triple Fusion)
3. MAINX.ipynb
4. vit_fusion_advanced.ipynb
5. vit_fusion_triple_CADICA_NHANES_ARCADE.ipynb   (optional — best model)
6. publication_experiments.ipynb
```

### 4. Launch the web app

```bash
streamlit run appxx.py
```

### 5. Programmatic usage

```python
from risk_calculator import calculate_ascvd_risk

result = calculate_ascvd_risk(
    age=58, sex="Male", race="White",
    total_cholesterol=220, hdl_cholesterol=42,
    systolic_bp=145, on_bp_medication=True,
    diabetes=False, current_smoker=True,
    bmi=29.5, hba1c=5.9, hs_crp=2.8
)
print(result["risk_category"])            # "High"
print(result["risk_enhancers"])           # ["Elevated inflammation (hs-CRP ≥2.0 mg/L)", ...]
```

```python
import numpy as np
from llm_integration import get_recommendations

recs = get_recommendations(
    prediction=2,
    confidence_scores=np.array([0.04, 0.12, 0.84])
)
print(recs["medication_guidance"])
print(recs["followup_schedule"])
```

---

## 📋 Dependencies

```
torch >= 2.0          timm            numpy           pandas
torchvision           scikit-learn    xgboost         scipy
opencv-python         Pillow          imageio         statsmodels
grad-cam              matplotlib      seaborn         tqdm
streamlit
```

> **GPU recommended** for training. Inference runs on CPU. FP16 mixed precision activates automatically when CUDA is available. DINO pretraining is tuned for 4GB VRAM (RTX 3050).

---

## 📎 Citation

```bibtex
@article{cardiovisionai2025,
  title   = {CardioVision AI: A Multimodal Vision Transformer Framework
             for Cardiovascular Event Risk Assessment},
  author  = {Nagabandi, Sampreeth and Sethi, Shakti Prasad and
             Gandarapu, Ajay Kumar and Pusala, Sai Santhan and
             Mulpuri, Yashwanth Venkat Chowdary},
  institution = {Amrita School of Artificial Intelligence,
                 Amrita Vishwa Vidyapeetham},
  year    = {2025}
}
```

---

## ⚠️ Disclaimer

> **This tool is intended for research and educational purposes only.**  
> It is **not** a certified medical device and **must not** be used as a substitute for professional clinical judgment, diagnosis, or treatment.  
> All clinical recommendations are derived from published ACC/AHA guidelines and are provided as reference material only.  
> The system has not undergone prospective clinical validation and should be regarded as a research prototype until such evaluation is completed.  
> Always consult a qualified cardiologist or physician for patient care decisions.
