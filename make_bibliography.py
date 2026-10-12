import json

with open('results/master_30_rock_solid_references.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []
lines.append('% =========================================================================')
lines.append('% MASTER BIBLIOGRAPHY: 30 VERIFIED ROCK-SOLID REFERENCES (MDPI / CATEGORY W)')
lines.append('% Strictly ascending citation keys: ref1 to ref30')
lines.append('% =========================================================================')
lines.append('\\begin{thebibliography}{99}')
lines.append('')

# Part A: Foundational 25 references
for i, r in enumerate(data.get('part_a_foundational_25_references', [])):
    key = f'ref{i+1}'
    authors = r.get('authors', '')
    year = r.get('year', '')
    title = r.get('title', '')
    venue = r.get('venue', '')
    doi = r.get('doi', '')
    if not doi.startswith('http'):
        doi_url = f"https://doi.org/{doi}"
    else:
        doi_url = doi
    
    lines.append(f'\\bibitem{{{key}}}')
    lines.append(f'{authors} {title}. \\emph{{{venue}}} \\textbf{{{year}}}. \\url{{{doi_url}}}')
    lines.append('')

# Part B: 5 SOTA references
for i, r in enumerate(data.get('part_b_sota_2025_2026_comparison_5_references', [])):
    key = f'ref{i+26}'
    authors = r.get('author_short', '')
    year = r.get('year', '')
    title = r.get('title', '')
    venue = r.get('venue', '')
    doi_url = r.get('doi', '')
    
    lines.append(f'\\bibitem{{{key}}}')
    lines.append(f'{authors} {title}. \\emph{{{venue}}} \\textbf{{{year}}}. \\url{{{doi_url}}}')
    lines.append('')

lines.append('\\end{thebibliography}')

with open('manuscript_bibliography.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Generated manuscript_bibliography.tex with 30 items!')
