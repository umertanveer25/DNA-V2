import os
import re

files_to_update = [
    'manuscript_abstract.tex',
    'manuscript_section1_introduction.tex',
    'build_section1.py',
    'distribute_references.py',
    'export_references_markdown.py',
    'generate_master_references.py',
    'results/MASTER_30_ROCK_SOLID_REFERENCES.md',
    'results/master_30_rock_solid_references.json',
    'README.md'
]

for fpath in files_to_update:
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace variations of P-EXP with PROMISE-EXP / Promise-EXP
        # If it's P-EXP, replace with PROMISE-EXP
        new_content = re.sub(r'\bP-EXP\b', 'PROMISE-EXP', content)
        new_content = re.sub(r'\bP-EXP-969\b', 'PROMISE-EXP-969', new_content)
        new_content = re.sub(r'\bp-exp\b', 'promise-exp', new_content)
        
        if new_content != content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {fpath}")
        else:
            print(f"No changes needed in {fpath}")

print("Dataset naming standardized to PROMISE-EXP across all target files!")
