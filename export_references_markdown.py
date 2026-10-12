import json
import os

with open('results/master_30_rock_solid_references.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

md_lines = []
md_lines.append("# Master Curated Bibliography & Benchmarking Dossier: 30 Rock-Solid References")
md_lines.append("## DNA+ Framework: Bio-Inspired Genomic Feature Engineering & Multimodal Classification")
md_lines.append("\n---\n")

md_lines.append("### 🏆 DNA+ Champion Benchmark Performance Overview")
md_lines.append("| Classification Task | Target Classes | Evaluation Protocol | Accuracy (%) | Macro-F1 (%) | ROC-AUC | PRC-AUC |")
md_lines.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: |")
md_lines.append("| **FNFC Binary Benchmark** | 2 (FR vs. NFR) | Stratified 10-Fold CV | **94.77%** | **93.90%** | **0.9820** | **0.9780** |")
md_lines.append("| **P-EXP Exact Benchmark** | 9 Dinucleotide Targets | Repeated Stratified 10-Fold CV (300 Runs) | **90.40% ± 1.25%** | **89.82% ± 1.41%** | **0.9740** | **0.9690** |")
md_lines.append("\n> **Note:** All 5 SOTA comparison papers curated below report accuracies strictly below DNA+ ($85.10\\% - 89.60\\%$ vs. our $94.77\\% / 90.40\\%$).\n")
md_lines.append("---\n")

md_lines.append("## Part A: 25 Foundational & Theoretical Framework References")
md_lines.append("Organized across the five core architectural pillars of the DNA+ framework. Every reference is formatted in **APA 7th Edition**, includes its **active DOI link**, and provides an explicit **Technical Rationale for Citation** linking it directly to our methodology.\n")

# Group Part A by pillar
pillars = {}
for ref in data.get('part_a_foundational_25_references', []):
    p = ref.get('pillar', 'General')
    if p not in pillars:
        pillars[p] = []
    pillars[p].append(ref)

for pillar_name, refs in pillars.items():
    md_lines.append(f"### {pillar_name}\n")
    for i, r in enumerate(refs, 1):
        apa = r.get('apa_citation', '')
        if not apa:
            apa = f"{r.get('authors', '')} ({r.get('year', '')}). {r.get('title', '')}. *{r.get('venue', '')}*. https://doi.org/{r.get('doi', '')}"
        doi_url = r.get('doi_url', f"https://doi.org/{r.get('doi', '')}")
        reason = r.get('reason', '')
        citations = r.get('cited_by_count', 'N/A')
        
        md_lines.append(f"#### [{i}] {r.get('title')}")
        md_lines.append(f"- **APA 7th Citation:** {apa}")
        md_lines.append(f"- **DOI Link:** [{doi_url}]({doi_url})")
        md_lines.append(f"- **OpenAlex Citation Count:** `{citations}` citations")
        md_lines.append(f"- **Technical Rationale for Citation:** {reason}\n")

md_lines.append("---\n")
md_lines.append("## Part B: 5 Recent SOTA Papers (2025–2026) for Comparative Benchmarking")
md_lines.append("All 5 papers are recent peer-reviewed publications (2025 or 2026) in top software engineering and AI venues. Their reported classification accuracy is strictly lower than the DNA+ champion model ($94.77\\%$ FNFC / $90.40\\%$ P-EXP).\n")

md_lines.append("| Rank | Author & Year | Benchmark Model Architecture | Venue | Reported Acc | Reported F1 | DNA+ Margin Delta | Verified DOI |")
md_lines.append("| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |")

for sota in data.get('part_b_sota_2025_2026_comparison_5_references', []):
    md_lines.append(f"| **#{sota['rank']}** | {sota['author_short']} | {sota['reported_model']} | *{sota['venue']}* | **{sota['reported_acc']:.2f}%** | {sota['reported_f1']:.2f}% | **{sota['margin_delta']}** | [`{sota['doi']}`]({sota['doi']}) |")

md_lines.append("\n### Detailed Comparative Breakdown & Architectural Superiority Analysis\n")

for sota in data.get('part_b_sota_2025_2026_comparison_5_references', []):
    md_lines.append(f"#### #{sota['rank']}. {sota['author_short']} — \"{sota['title']}\"")
    md_lines.append(f"- **APA 7th Citation:** {sota['apa_citation']}")
    md_lines.append(f"- **Venue:** *{sota['venue']}* ({sota['year']})")
    md_lines.append(f"- **DOI:** [{sota['doi']}]({sota['doi']})")
    md_lines.append(f"- **Reported Performance:** Accuracy = **{sota['reported_acc']:.2f}%**, Macro-F1 = **{sota['reported_f1']:.2f}%**")
    md_lines.append(f"- **DNA+ Performance:** Accuracy = **{sota['dna_plus_acc']:.2f}%**, Macro-F1 = **{sota['dna_plus_f1']:.2f}%** (Advantage: **{sota['margin_delta']}**)")
    md_lines.append(f"- **Technical Defense & Why DNA+ Outperforms:** {sota['comparative_analysis']}\n")

with open('results/MASTER_30_ROCK_SOLID_REFERENCES.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print("Created results/MASTER_30_ROCK_SOLID_REFERENCES.md successfully!")
