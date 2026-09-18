# Knowledge Base: Dynamic Syntactic-Semantic Dinucleotide & True Sequence Repository

This directory contains the dynamic mappings and biological genomic sequence representations synthesized from natural language requirements.

---

## 1. Dynamic Syntactic-Semantic Dinucleotide Mapping

In contrast to static 4-monomer assignments (which force requirements into crude single bases like `A`, `T`, `C`, or `G`), the **DNA+ framework** parses requirements using a two-tier syntactic-semantic architecture:

$$\text{Dinucleotide Pair} = \text{Base}_1 (\text{Primary Semantic Anchor}) + \text{Base}_2 (\text{Auxiliary Syntactic Dependency})$$

### Core Biological Monomers:
- **A (Adenine)**: Obligation, Imperative Logic, and Core Provisions (`shall`, `must`, `will`, `provide`, `execute`).
- **T (Thymine)**: Usability, Interface Aesthetics, and Human Interaction (`user`, `interface`, `screen`, `menu`, `readable`).
- **C (Cytosine)**: Security, Compliance, Encryption, and Access Control (`secure`, `authenticate`, `gdpr`, `audit`, `privacy`).
- **G (Guanine)**: Performance, Speed, Capacity, Reliability, and Infrastructure (`fast`, `latency`, `throughput`, `concurrent`, `failover`).

### Coupled Dinucleotide Classes:
- **`AG`**: Obligation coupled with Performance/System Capacity (e.g., *"The system shall process 5,000 transactions per second"*).
- **`AT`**: Obligation coupled with User Interface & Usability (e.g., *"The dashboard shall display intuitive graphical navigation"*).
- **`AA`**: Pure Functional Imperative (e.g., *"The platform must store and retrieve customer invoice records"*).
- **`AC`**: Obligation coupled with Security & Regulatory Policy (e.g., *"The database must encrypt all stored credentials using AES-256"*).
- **`GG`**: Pure Intensive Performance, Scalability & Fault-Tolerance.
- **`GT`**: High-Speed Rendering, Interactive Latency & Real-Time Visualization.
- **`CC`**: High-Assurance Cryptography, Security Auditing & Governance.
- **`TT`**: Pure Ergonomic, Accessibility & Aesthetic Usability.
- **`CG`**: Resilient Security Architecture, Zero-Downtime High Availability.

---

## 2. True Biological Genomic Sequence Synthesis

For every requirement text, a true biological sequence is assembled in FASTA-like nucleotide format ($A, T, C, G$). From these synthesized sequences, our pipeline extracts:
1. **1-mer Frequencies (4 dimensions)**: Relative frequencies of $A, T, C, G$.
2. **2-mer Dinucleotide Frequencies (16 dimensions)**: Full dimer permutation frequencies ($AA, AT, AC, AG, TA, TT, TC, TG, CA, CT, CC, CG, GA, GT, GC, GG$).
3. **Biochemical Traits (3 dimensions)**:
   - **GC-Content**: Ratio of strong triple-hydrogen-bonded pairs: $\frac{G + C}{L}$.
   - **AT/GC Ratio**: Charge balance indicator: $\frac{A + T}{G + C + \epsilon}$.
   - **Normalized Sequence Length**: Log-scaled length: $\ln(1 + L)$.
4. **Multimodal Fusion**: Combined with TF-IDF and SBERT semantic embeddings to form the unified 457-dimensional feature space.
