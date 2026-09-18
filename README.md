# DNA+ : Advanced Dinucleotide Semantic Coupling & Knowledge-Based Feature Engineering for Software Requirements Classification

This repository contains the complete experimental code, knowledge base generator, and evaluation pipeline for Paper 2 (DNA+).

## Key Innovations over Paper 1:
1. **Linguistic Knowledge Base**: Every requirement is grounded and scored against ISO-25010 domain vocabularies, eliminating heuristic unmapped ("N") classes.
2. **Dinucleotide Semantic Cross-Coupling**: Introduces 6 dimer features ($AT, AG, AC, TG, TC, GC$) plus biochemical ratios ($AT\_ratio, GC\_ratio, ATGC\_ratio$) and Shannon entropy.
3. **Leakage-Free SMOTE**: Class imbalance handled strictly inside cross-validation partitions.
4. **Multi-Dataset Rigor**: Evaluated across both PROMISE (625 requirements) and FNFC (7,060 requirements).
5. **Explainable AI (XAI)**: Full SHAP TreeExplainer attribution quantifying the impact of DNA features on model decisions.
6. **Bayesian Hyperparameter Optimization**: Automated tuning via Optuna.

## Execution Guide

1. **Build Knowledge Base**:
   ```bash
   python src/knowledge_base_builder.py
   ```

2. **Run Comparative Benchmarks (12 Classifiers on both datasets)**:
   ```bash
   python run_pipeline.py --splits 3
   ```

3. **Run Bayesian Optuna Tuning**:
   ```bash
   python run_optuna.py --dataset promise --trials 30
   ```

4. **Run SHAP Explainability**:
   ```bash
   python src/explainability.py
   ```

5. **Generate Publication Visualizations**:
   ```bash
   python generate_plots.py
   ```
