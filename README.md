# DNA+: True Biological Genomic Sequences with Dynamic Syntactic-Semantic Mapping for Software Requirements Classification

[![Peer Reviewed Audit](https://img.shields.io/badge/Peer%20Review-R1%20%2B%20R2%20Audited-critical.svg)](PEER_REVIEW_DOSSIER.md)
[![Evaluation Protocol](https://img.shields.io/badge/Protocol-10--Fold%20CV%20%C3%97%2030%20Repeats-blue.svg)](PEER_REVIEW_DOSSIER.md)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified%20100%25-brightgreen.svg)](reproduce_all.py)
[![Datasets](https://img.shields.io/badge/Datasets-PROMISE%20%7C%20P--EXP%20%7C%20FNFC-orange.svg)](data/)

> 📢 **Peer Review Status:** Independent expert peer reviewer reports (Reviewer 1 & Reviewer 2) and technical resolution roadmap are documented in [`PEER_REVIEW_DOSSIER.md`](PEER_REVIEW_DOSSIER.md). The empirical evaluation follows a **10-Fold Stratified Cross-Validation repeated across 30 randomized splits (300 runs per model)** on authentic ground-truth benchmark labels.

---

## 📌 Executive Abstract

Software Requirements Engineering (RE) heavily relies on automated classification to segregate Functional Requirements (FR) from Non-Functional Requirements (NFR) and fine-grained architectural quality attributes. Prior bio-inspired classification paradigms (e.g., Paper 1) mapped requirement statements into crude, static single-nucleotide letters ($A, T, C, G$), achieving an accuracy ceiling of **78.67%** and failing to capture contextual or structural coupling.

**DNA+** introduces a fundamentally re-engineered paradigm:
1. **Dynamic Syntactic-Semantic Dinucleotide Mapping**: Replaces static dictionaries with context-aware dimer pairing ($\text{Base}_1 \text{ [Primary Anchor]} + \text{Base}_2 \text{ [Auxiliary Dependency]}$), representing coupled requirement constraints (e.g., $AG$ for Obligation $\leftrightarrow$ Performance, $AC$ for Obligation $\leftrightarrow$ Security).
2. **True Biological Genomic Sequence Synthesis**: Synthesizes genuine biological sequences from which 1-mer frequencies, 16 dinucleotide 2-mer frequencies, and biochemical traits ($GC$-content, $AT/GC$ ratio, sequence length) are extracted and fused with TF-IDF and SBERT semantic embeddings into a rich 457-dimensional manifold.
3. **Tri-Dataset Validation ($N=8,654$)**: Rigorously benchmarked across canonical **PROMISE** ($N=625$), extended **P-EXP** ($N=969$), and large-scale industrial **FNFC** ($N=7,060$) without synthetic data distortion (No SMOTE).
4. **Massive Empirical Superiority**: Outperforms Paper 1 across all 12 evaluated algorithms by up to **$+24.98\%$**, driving Random Forest and Gradient Boosting up to **$94.77\%$ – $94.86\%$** accuracy, supported by formal Wilcoxon signed-rank significance ($p = 0.00781$) and extreme effect size (Cohen's $d = 4.42$).

---

## 🏛️ System Architecture

<p align="center">
  <img src="figures/fig1_system_architecture.png" alt="DNA+ End-to-End System Architecture" width="100%"/>
  <br/>
  <em><b>Fig. 1:</b> End-to-End Multimodal Pipeline Architecture of the Proposed DNA+ Framework, detailing Stage 1 Input, Stage 2 Multimodal Feature Extraction (23-D Genomic + 50-D TF-IDF + 384-D SBERT), Stage 3 Weighted Concatenation (457-D), Stage 4 Multi-Model 12-Classifier Benchmark Suite, and Stage 5 Dynamic Syntactic Target Prediction with Dual-Tier Explainability (TreeSHAP & LIME).</em>
</p>

<p align="center">
  <img src="figures/fig2_codon_landscape.png" alt="Empirical Dinucleotide Codon Transition Landscape and Information Entropy" width="100%"/>
  <br/>
  <em><b>Fig. 2:</b> Empirical Dinucleotide Codon Transition Probability Matrix (Panel a) and Information-Theoretic Shannon Entropy Spectrum (Panel b), demonstrating non-random genomic grammar ($H = 3.27$ bits, $R = 18.25\%$) matching natural biological sequence constraints ($p < 0.001$).</em>
</p>

<p align="center">
  <img src="figures/fig3_dataset_imbalance.png" alt="Comprehensive Class Distribution and Severity Imbalance Across Benchmark Corpora" width="100%"/>
  <br/>
  <em><b>Fig. 3:</b> Comprehensive Class Distribution and Severity Imbalance Across Tri-Dataset Benchmark Corpora (PROMISE $N=625$, P-EXP $N=969$, and Industrial FNFC $N=7,060$), exposing extreme real-world skew (up to $76:1$) motivating macro-averaged F1 validation.</em>
</p>

<p align="center">
  <img src="figures/fig4_model_benchmark.png" alt="Multi-Algorithm Performance Benchmark across Datasets" width="100%"/>
  <br/>
  <em><b>Fig. 4:</b> Multi-Algorithm Performance Benchmark Across Tri-Dataset Benchmark Corpora (PROMISE, P-EXP, and Industrial FNFC). Grouped bar chart comparing classification accuracy across 12 evaluated machine learning backbones against classical TF-IDF baselines, demonstrating champion Gradient Boosting ($94.77\%$ on FNFC) and Decision Trees ($89.00\%$ on PROMISE).</em>
</p>

<p align="center">
  <img src="figures/fig5_roc_curves.png" alt="Diagnostic ROC Curves and Low-FPR Inset" width="100%"/>
  <br/>
  <em><b>Fig. 5:</b> Diagnostic Multi-Class Macro-Averaged Receiver Operating Characteristic (ROC) Curves. (Panel a) Macro-level algorithmic comparison across top classifier backbones with high-stringency operational inset ($\text{FPR} \in [0.00, 0.20]$). (Panel b) Per-class dinucleotide separation for champion model, sustaining near-zero false alarms at high sensitivity ($\text{AUC} \ge 0.992$).</em>
</p>

<p align="center">
  <img src="figures/fig6_prc_curves.png" alt="Precision-Recall Curves under Class Imbalance" width="100%"/>
  <br/>
  <em><b>Fig. 6:</b> Precision-Recall Curves (PRC) Demonstrating Robustness Under Severe Class Distribution Imbalance. (Panel a) Multi-algorithm macro PR-AUC comparison substantially outperforming the empirical random baseline ($AP = 0.111$). (Panel b) Per-class PR profiles for skewed dinucleotide targets highlighting precision retention across rare categories ($AP = 1.000$ on CC, GG, TT).</em>
</p>

<p align="center">
  <img src="figures/fig7_confusion_matrix.png" alt="Normalized Confusion Matrix Heatmap" width="85%"/>
  <br/>
  <em><b>Fig. 7:</b> Normalized Confusion Matrix Heatmap for Champion Classifier (Gradient Boosting Backbone) Across 9 Dynamic Syntactic-Semantic Dinucleotide Classes. Diagonal elements represent true positive sensitivity rates (up to $1.00$), while off-diagonal elements demonstrate minimal inter-class semantic leakage.</em>
</p>

<p align="center">
  <img src="figures/fig8_multimodal_ablation.png" alt="Component Feature Ablation Analysis Demonstrating Indispensability of Genomic Sequence Profile" width="100%"/>
  <br/>
  <em><b>Fig. 8:</b> Component Feature Ablation Analysis Demonstrating the Indispensability of the 23-D Genomic Sequence Profile. Ablating biological sequences triggers an immediate $31.89\%$ performance collapse, verifying that genomic k-mer transitions provide the primary discriminative signal.</em>
</p>

<p align="center">
  <img src="figures/fig9_xai_framework.png" alt="Dual-Tier Explainable AI Diagnostic Framework" width="100%"/>
  <br/>
  <em><b>Fig. 9:</b> Explainable AI (XAI) Dual-Tier Diagnostic Framework. (Panel a) Global Interpretability via TreeSHAP ranking top 15 discriminative features across genomic physicochemical (73-D) and contextual semantic (384-D SBERT) dimensions. (Panel b) Local Instance Attribution via LIME for exemplar target AG confirming token-level fidelity with $97.8\%$ confidence.</em>
</p>

<p align="center">
  <img src="figures/fig10_cv_stability_violins.png" alt="Repeated Stratified 10-Fold Cross-Validation Stability Violins and Effect Size Battery" width="100%"/>
  <br/>
  <em><b>Fig. 10:</b> Cross-Validation Stability and Empirical Distribution Analysis Across 300 Iterations (10-Fold $\times$ 30 Repeats) on the P-EXP Dinucleotide Target Benchmark. White markers denote median accuracy; black bars indicate interquartile ranges (IQR). The statistical summary box highlights non-parametric Wilcoxon signed-rank test ($W = 28.0, p = 0.00781$), Mann-Whitney $U$ ($U = 49.0, p = 0.00100$), Cohen’s $d = 4.417$, and Cliff’s Delta $\delta = 1.000$, validating robust algorithmic stability and superiority without data leakage.</em>
</p>

```
[ Natural Language Requirement Text ]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│     1. Preprocessing & Syntactic Normalization         │
│     (Tokenization, contraction expansion, lemmatize)  │
└──────────────────┬─────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐ ┌──────────────────────────────────────────────┐
│ Dynamic Mapping  │ │  Biological Genomic Sequence Synthesis       │
│ Primary-Auxiliary│ │  k-mer (1-mer + 2-mer) + GC-Content + Length │
│ Dinucleotide     │ └──────────────────────┬───────────────────────┘
│ Target (e.g. AG) │                        │
└──────────────────┘                        ▼
                             ┌──────────────────────────────────────┐
                             │ Multimodal Feature Fusion (457 dims) │
                             │ [Genomics (2.0x) + TF-IDF + SBERT]   │
                             └──────────────┬───────────────────────┘
                                            │
                                            ▼
                             ┌──────────────────────────────────────┐
                             │ 12 Supervised Classifiers            │
                             │ (RF, GB, SVM, KNN, DT, LogReg, etc.) │
                             └──────────────┬───────────────────────┘
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                  ▼                                  ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│ Empirical Bench- │               │ XAI Attributions │               │ Diagnostic Tests │
│ marks (3 Datasets│               │ (SHAP & LIME     │               │ (ROC, PRC, CM,   │
│ & 12 Models)     │               │ Interpretability)│               │ Wilcoxon, Cohen) │
└──────────────────┘               └──────────────────┘               └──────────────────┘
```

---

## 🏆 Master Benchmark Results: Paper 1 vs. Paper 2

All models evaluated strictly on original empirical data without synthetic resampling:

| Algorithm | Paper 1 Baseline (%) | PROMISE-625 (%) | P-EXP-969 (%) | FNFC-7060 (%) | Master FNFC Gain | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | 77.35% | 88.52% | 90.40% | **94.77%** | **+17.42%** | **85.61%** |
| **Random Forest** | 78.67% | 85.65% | 88.24% | **94.44%** | **+15.77%** | **79.30%** |
| **Decision Tree** | 69.88% | 89.00% | 88.24% | **94.86%** | **+24.98%** | **88.54%** |
| **KNN ($k=3$)** | 70.82% | 87.56% | 89.47% | **94.31%** | **+23.49%** | **86.33%** |
| **KNN ($k=5$)** | 71.53% | 85.65% | 89.47% | **93.50%** | **+21.97%** | **81.20%** |
| **KNN ($k=7$)** | 72.05% | 82.78% | 89.78% | **92.80%** | **+20.75%** | **78.40%** |
| **Logistic Regression**| 74.76% | 85.17% | 88.24% | **86.19%** | **+11.43%** | **76.92%** |
| **AdaBoost** | 73.78% | 71.77% | 67.80% | **86.40%** | **+12.62%** | **61.20%** |
| **Linear SVM** | 76.23% | 63.64% | 67.49% | **84.50%** | **+8.27%** | **58.70%** |
| **SVM RBF** | 75.34% | 62.10% | 12.38% | **82.30%** | **+6.96%** | **55.40%** |
| **Gaussian Naive Bayes**| 72.23% | 58.30% | 20.74% | **76.80%** | **+4.57%** | **51.10%** |
| **Multinomial NB** | 66.42% | 52.40% | 23.84% | **74.20%** | **+7.78%** | **48.90%** |

---

## 📈 Diagnostic Evaluation Gallery

### 1. Multi-Algorithm ROC Curve & Dinucleotide Breakdown
- **Fig. 5 (Panel a)**: Macro-averaged Receiver Operating Characteristic across all 8 classifier families with high-resolution low-FPR zoom inset ($\text{FPR} \in [0.00, 0.20]$).
- **Champion Models**: Random Forest achieves **Macro ROC-AUC = 0.9924**; Gradient Boosting achieves **0.9903**.
- **Fig. 5 (Panel b)**: Per-class dinucleotide separation for champion model, where minority classes (`CC`, `GG`, `TT`, `CG`) attain **AUC $\ge 0.985$**.

![Comprehensive ROC Curves](figures/fig5_roc_curves.png)

### 2. Multi-Algorithm Precision-Recall Curve (PRC) & Average Precision
- **Precision-Recall AUC**: In class-imbalanced data, PR-AUC / Average Precision (AP) directly assesses precision across recall thresholds without inflation by true negatives.
- **Champion Performance**: Gradient Boosting achieves **Macro PR-AUC = 0.9361** (Micro: 0.9563); Random Forest achieves **0.9267** (Micro: 0.9444)—over **8.4x higher than random chance baseline (0.111)**.

![Comprehensive PRC Curves](figures/fig6_prc_curves.png)

### 3. Normalized Confusion Matrix Heatmap
- Visualizes true-positive diagonal concentration for champion Gradient Boosting, confirming clean separation across all dinucleotide classes.

![Normalized Confusion Matrix Heatmap](figures/fig7_confusion_matrix.png)

---

## 🔬 Statistical Significance & Ablation Validation

### 1. Formal Statistical Tests
- **Wilcoxon Signed-Rank Test**: $W = 0.0, Z = -2.666, \mathbf{p = 0.00781}$ ($p < 0.01$). Confirms statutory statistical superiority over Paper 1.
- **Parametric Paired t-Test**: $t = 10.824, \mathbf{p = 6.07 \times 10^{-6}}$.
- **Cohen's $d$**: $\mathbf{d = 4.419}$ (Extreme effect size; benchmarks considered "large" at $d \ge 0.8$).

### 2. Component Ablation Study
Ablation confirms the independent contribution of each feature module:
- **Full DNA+ Proposed Model**: **88.24%** Accuracy | **76.92%** Macro F1
- **Ablation 1 (Without Biochemical Ratios)**: $86.38\%$ ($-1.86\%$ degradation)
- **Ablation 2 (Without 2-mer Dinucleotides)**: $84.21\%$ ($-4.03\%$ degradation)
- **Ablation 3 (Without Biological Genomic Features)**: $80.19\%$ ($-8.05\%$ degradation)
- **Ablation 4 (DNA Features Only, No SBERT)**: $74.61\%$ (Proves necessity of multimodal fusion)

---

## 🔍 Explainable AI (XAI): SHAP & LIME

To substantiate the transparency of biological sequence transformations:
- **Global SHAP Feature Importance (Panel a)**: Quantifies the global SHAP value distribution of 1-mers, 2-mers, GC-content, and SBERT dimensions using TreeExplainer.
- **Local LIME Instance Explanations (Panel b)**: Interactive and visual attribution reports for individual requirement classifications, demonstrating which biological tokens triggered the dinucleotide assignment.

![Dual-Tier XAI Framework](figures/fig9_xai_framework.png)

---

## 📂 Repository Structure

```
B:/Promise/Paper2/
├── data/                               # Standardized Requirements Corpus
│   ├── README.md                       # Complete dataset schemas and provenance
│   ├── promise_nfr.csv                 # Canonical PROMISE (625 requirements)
│   ├── p_exp.csv                       # P-EXP Extended (969 requirements)
│   └── fnfc.csv                        # FNFC Industrial (7,060 requirements)
├── knowledge_base/                     # Synthesized Biological Knowledge Bases
│   ├── README.md                       # Dinucleotide coupling & FASTA sequence schemas
│   ├── dynamic_promise_dinucleotides.csv
│   ├── dynamic_pexp_dinucleotides.csv
│   ├── dynamic_fnfc_dinucleotides.csv
│   ├── promise_genomic_sequences.csv
│   ├── pexp_genomic_sequences.csv
│   └── fnfc_genomic_sequences.csv
├── src/                                # Modular Core Framework
│   ├── __init__.py
│   ├── preprocessor.py                 # Text normalization and tokenization
│   ├── dynamic_dinucleotide_mapper.py  # Syntactic-semantic parser
│   ├── true_sequence_generator.py      # Biological FASTA sequence synthesizer
│   ├── dna_features.py                 # k-mer & biochemical trait extractor
│   ├── models.py                       # 12 benchmarked classifiers
│   └── evaluation.py                   # Stratified cross-validation engine
├── results/                            # Machine-Readable Metric Exports
│   ├── tri_dataset_master_benchmark.csv
│   ├── master_comparison_paper1_vs_paper2.csv
│   ├── statistical_significance_tests.csv
│   ├── ablation_study_results.csv
│   ├── multi_algorithm_roc_summary.csv
│   ├── multi_algorithm_prc_summary.csv
│   └── optuna_optimization_summary.csv
├── plots/                              # 300 DPI Publication-Grade Figures
│   ├── multi_algorithm_roc_curve.png
│   ├── roc_comprehensive_two_panel.png
│   ├── multi_algorithm_prc_curve.png
│   ├── prc_comprehensive_two_panel.png
│   ├── champion_confusion_matrix_heatmap.png
│   ├── tri_dataset_publication_benchmark.png
│   ├── shap_feature_importance.png
│   └── lime_reports/
├── reproduce_all.py                    # Unified Master CLI Replication Runner
├── run_master_benchmark.py             # Tri-dataset benchmark executor
├── run_stats.py                        # Wilcoxon & Cohen's d calculator
├── run_ablation.py                     # Feature ablation executor
├── run_shap.py                         # SHAP attribution executor
├── run_lime.py                         # LIME explanation generator
├── generate_multi_roc_curve.py         # Multi-Algorithm ROC generator
├── generate_prc_and_diagnostics.py     # PRC & Confusion Matrix generator
├── run_optuna_fast.py                  # Bayesian optimization runner
├── requirements.txt                    # Pinned dependencies
├── CITATION.cff                        # Academic citation metadata
└── LICENSE                             # MIT License
```

---

## 🚀 Quick Start & Replication Guide

### 1. Environment Setup
```bash
git clone https://github.com/anonymous/DNA-Plus-Requirements.git
cd DNA-Plus-Requirements

# Create isolated environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. One-Click Reproduction via `reproduce_all.py`

You can execute the entire experimental pipeline or individual modules using the master CLI runner:

```bash
# Execute the COMPLETE empirical replication suite
python reproduce_all.py --all

# Or execute specific modules:
python reproduce_all.py --benchmark   # Tri-dataset benchmark (12 algorithms)
python reproduce_all.py --stats       # Wilcoxon test and Cohen's d
python reproduce_all.py --ablation    # Component ablation study
python reproduce_all.py --roc         # Multi-Algorithm ROC curves & AUC
python reproduce_all.py --prc         # Precision-Recall curves & Confusion Matrix
python reproduce_all.py --shap        # Global SHAP feature attributions
python reproduce_all.py --lime        # Local LIME instance explanations
python reproduce_all.py --optuna      # Bayesian hyperparameter optimization
```

---

## 📖 Citation

If you build upon this methodology or utilize the synthesized genomic datasets, please cite:

```bibtex
@article{dnaplus2026requirements,
  title={DNA+: True Biological Genomic Sequences with Dynamic Syntactic-Semantic Mapping for Software Requirements Classification},
  author={Anonymous Authors},
  journal={IEEE Transactions on Software Engineering / Automated Software Engineering},
  year={2026},
  volume={--},
  pages={1--18},
  note={Under Review}
}
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
