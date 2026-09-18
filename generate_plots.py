import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('plots', exist_ok=True)

def plot_benchmark_comparison(csv_path, dataset_name):
    if not os.path.exists(csv_path):
        print(f"[-] Missing {csv_path}, skipping benchmark plot.")
        return
        
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='Accuracy (%) (DNA+ & SMOTE)', ascending=True)
    
    y = np.arange(len(df))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.barh(y - height/2, df['Accuracy (%) (Original)'], height, label='Paper 1 Baseline (No Resampling)', color='#6c757d', alpha=0.85)
    rects2 = ax.barh(y + height/2, df['Accuracy (%) (DNA+ & SMOTE)'], height, label='Paper 2 Enhanced (DNA+ Pairs & SMOTE)', color='#1d3557', alpha=0.95)
    
    ax.set_xlabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Comparison: Paper 1 vs. Paper 2 on {dataset_name.upper()}', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(df['Algorithm'], fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0, 100)
    
    # Add labels
    for rect in rects2:
        width = rect.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold', color='#1d3557')
                    
    plt.tight_layout()
    out_img = f"plots/{dataset_name.lower()}_benchmark_comparison.png"
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"[+] Saved comparison plot to {out_img}")

def plot_dna_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PROMISE
    if os.path.exists('knowledge_base/promise_knowledge_base.csv'):
        df_p = pd.read_csv('knowledge_base/promise_knowledge_base.csv')
        counts_p = df_p['Assigned_DNA'].value_counts()
        sns.barplot(x=counts_p.index, y=counts_p.values, ax=axes[0], palette='Blues_d')
        axes[0].set_title('PROMISE Dataset: DNA Target Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('DNA Monomer / Dinucleotide Code')
        axes[0].set_ylabel('Number of Requirements')
        for i, v in enumerate(counts_p.values):
            axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')
            
    # FNFC
    if os.path.exists('knowledge_base/fnfc_knowledge_base.csv'):
        df_f = pd.read_csv('knowledge_base/fnfc_knowledge_base.csv')
        counts_f = df_f['Assigned_DNA'].value_counts()
        sns.barplot(x=counts_f.index, y=counts_f.values, ax=axes[1], palette='Purples_d')
        axes[1].set_title('FNFC Dataset: DNA Target Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('DNA Monomer / Dinucleotide Code')
        axes[1].set_ylabel('Number of Requirements')
        for i, v in enumerate(counts_f.values):
            axes[1].text(i, v + 25, str(v), ha='center', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig('plots/dna_distribution_both_datasets.png', dpi=300)
    plt.close()
    print("[+] Saved DNA target distribution plot to plots/dna_distribution_both_datasets.png")

if __name__ == '__main__':
    plot_benchmark_comparison('results/promise_benchmark_comparison.csv', 'PROMISE')
    plot_benchmark_comparison('results/fnfc_benchmark_comparison.csv', 'FNFC')
    plot_dna_distribution()
