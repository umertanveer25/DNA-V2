import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.dynamic_dinucleotide_mapper import process_dynamic_mapping
from src.true_sequence_generator import process_true_dna_sequences

print("[*] Processing P-EXP Dataset (969 Requirements)...")

# 1. Dynamic Syntactic-Semantic Mapping
p_exp_in = "B:/Promise/Paper2/data/p_exp.csv"
p_exp_dinucl_out = "B:/Promise/Paper2/knowledge_base/dynamic_pexp_dinucleotides.csv"
process_dynamic_mapping(p_exp_in, 'Requirement', 'Type', 'P-EXP', p_exp_dinucl_out)

# 2. True Biological Genomic Sequences
p_exp_genomic_out = "B:/Promise/Paper2/knowledge_base/pexp_genomic_sequences.csv"
process_true_dna_sequences(p_exp_in, 'Requirement', 'Type', 'P-EXP', p_exp_genomic_out)

print("\n[+] P-EXP Genomic Processing Complete!")
