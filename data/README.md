# Software Requirements Datasets: Tri-Dataset Empirical Corpus

This directory contains the three standardized benchmark datasets utilized across all evaluations in Paper 2:

| Dataset | Identifier | Requirements ($N$) | Categories / Classes | Primary Scope | Sourcing & Provenance |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **PROMISE Canonical** | `promise_nfr.csv` | **625** | 11 Classes (`F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `PO`) | Benchmark Baseline | Canonical PROMISE Software Engineering Repository (Cleland-Huang et al.) |
| **P-EXP Extended** | `p_exp.csv` | **969** | Multi-class NFRs & FRs | Intermediate Generalizability | Extended PROMISE requirements with balanced functional and quality sub-types |
| **FNFC Industrial** | `fnfc.csv` | **7,060** | Large-scale functional and non-functional requirements | Industrial Generalizability | Large-scale public industrial requirements corpus (Functional vs. Non-Functional Criteria) |

---

## 1. Dataset Schemas

### `promise_nfr.csv`
- `Project_ID`: Identification code of the software project.
- `Requirement_Text`: Natural language requirement specification string.
- `Class`: Original PROMISE categorization tag:
  - `F`: Functional Requirement
  - `A`: Availability
  - `L`: Legal / Regulatory
  - `LF`: Look and Feel
  - `MN`: Maintainability
  - `O`: Operability
  - `PE`: Performance
  - `SC`: Scalability
  - `SE`: Security
  - `US`: Usability
  - `PO`: Portability

### `p_exp.csv`
- `Requirement_Text`: Extended natural language requirements.
- `Class`: Fine-grained non-functional and architectural quality attribute labels.

### `fnfc.csv`
- `Requirement_Text`: Sentence-level and paragraph-level requirement statements.
- `Class`: Multi-tier quality tags across large industrial project portfolios.

---

## 2. Integrity and Preprocessing Protocol
1. **No Synthetic Oversampling (No SMOTE)**: To maintain complete empirical integrity and prevent hallucinated feature representations, all benchmarks are conducted directly on real requirements data.
2. **Text Normalization**: Requirements undergo standardized lowercasing, punctuation stripping, tokenization, and contraction expansion via `src/preprocessor.py`.
