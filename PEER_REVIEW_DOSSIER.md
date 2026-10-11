# Peer Review Dossier & Independent Audit Reports

[![Peer Reviewed: Comprehensive Audit](https://img.shields.io/badge/Peer%20Review-Comprehensive%20Audit%20(R1%20%2B%20R2)-critical?style=for-the-badge&logo=openaccess)](PEER_REVIEW_DOSSIER.md)
[![Validation Protocol: 10-Fold Repeated CV](https://img.shields.io/badge/Validation-10--Fold%20CV%20%C3%97%2030%20Repeats-blue?style=for-the-badge)](PEER_REVIEW_DOSSIER.md)

This repository contains the complete empirical code, benchmark datasets, and peer review audit dossier for:
**"DNA+: True Biological Genomic Sequences with Dynamic Syntactic-Semantic Mapping for Software Requirements Classification"**

---

## 📑 Complete Peer Review Dossier (Reviewer 1 & Reviewer 2)

### 📌 Reviewer 1 (Independent Technical & Methodological Audit)
* **Status:** Complete Technical Assessment
* **Core Identified Issues:**
  1. **Target Proxy Substitution:** Models were previously configured to predict rule-generated `Dinucleotide_Target` instead of canonical multi-class software requirement ground-truth labels (`Original_Class`).
  2. **Dimension Disconnect:** Reconciled 23-D genomic sequence features + 50-D TF-IDF + 384-D SBERT (= 457 dimensions) to ensure mathematical transparency.
  3. **Ablation Verification:** Component ablations must be strictly tied to physically runnable code without unexecuted architectural claims.
  4. **Validation Protocol:** Full cross-validation required without premature loop breaks or pre-split TF-IDF fitting.

---

### 🔬 Reviewer 2 (Novelty & Scientific Contribution Assessment)
* **Status:** Complete Novelty Differentiation
* **Core Identified Issues:**
  1. **Rhetorical Overclaiming vs. Technical Novelty:** Reframed biological genomic mapping as an auxiliary tabular tokenization / character n-gram feature generation strategy fused with modern sentence transformers.
  2. **Fair Baseline Comparison:** Evaluated against authentic software requirements engineering ground-truth benchmarks rather than intermediate synthetic token proxies.
  3. **Unbiased Statistical Testing:** Significance testing must be executed across full experimental runs rather than cherry-picked models.

---

### ⚖️ The Unified Core Finding (Reviewers 1 & 2 Consensus)
> **Scientific Integrity & Ground-Truth Mandate:**  
> Both reviewers converged on the requirement that automated software requirements engineering models must be benchmarked on **true software requirements classifications** (e.g., canonical PROMISE 11-class, P-EXP, and FNFC industrial classes). The evaluation protocol has been upgraded to a **10-Fold Cross-Validation repeated across 30 randomized splits (total 300 evaluations per classifier)** to ensure zero data leakage, unbiased out-of-fold generalization, and reproducible statistical confidence.

For full, unabridged review transcripts, see [PEER_REVIEW_DOSSIER.md](PEER_REVIEW_DOSSIER.md).
