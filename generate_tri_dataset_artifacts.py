import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Master Data across 3 Datasets
master_data = [
    {
        "Algorithm": "Gradient Boosting",
        "Paper 1 DNA (%)": 77.35,
        "PROMISE-625 (%)": 88.52,
        "P-EXP-969 (%)": 90.40,
        "P-EXP Gain (%)": "+13.05%",
        "FNFC-7060 (%)": 94.77,
        "FNFC Macro F1 (%)": 85.61
    },
    {
        "Algorithm": "KNN (k=3)",
        "Paper 1 DNA (%)": 70.82,
        "PROMISE-625 (%)": 87.56,
        "P-EXP-969 (%)": 89.47,
        "P-EXP Gain (%)": "+18.65%",
        "FNFC-7060 (%)": 94.31,
        "FNFC Macro F1 (%)": 86.33
    },
    {
        "Algorithm": "KNN (k=5)",
        "Paper 1 DNA (%)": 71.53,
        "PROMISE-625 (%)": 85.65,
        "P-EXP-969 (%)": 89.47,
        "P-EXP Gain (%)": "+17.94%",
        "FNFC-7060 (%)": 93.50,
        "FNFC Macro F1 (%)": 81.20
    },
    {
        "Algorithm": "KNN (k=7)",
        "Paper 1 DNA (%)": 72.05,
        "PROMISE-625 (%)": 82.78,
        "P-EXP-969 (%)": 89.78,
        "P-EXP Gain (%)": "+17.73%",
        "FNFC-7060 (%)": 92.80,
        "FNFC Macro F1 (%)": 78.40
    },
    {
        "Algorithm": "Decision Tree",
        "Paper 1 DNA (%)": 69.88,
        "PROMISE-625 (%)": 89.00,
        "P-EXP-969 (%)": 88.24,
        "P-EXP Gain (%)": "+18.36%",
        "FNFC-7060 (%)": 94.86,
        "FNFC Macro F1 (%)": 88.54
    },
    {
        "Algorithm": "Random Forest",
        "Paper 1 DNA (%)": 78.67,
        "PROMISE-625 (%)": 85.65,
        "P-EXP-969 (%)": 88.24,
        "P-EXP Gain (%)": "+9.57%",
        "FNFC-7060 (%)": 94.44,
        "FNFC Macro F1 (%)": 79.30
    },
    {
        "Algorithm": "Logistic Regression",
        "Paper 1 DNA (%)": 74.76,
        "PROMISE-625 (%)": 85.17,
        "P-EXP-969 (%)": 88.24,
        "P-EXP Gain (%)": "+13.48%",
        "FNFC-7060 (%)": 86.19,
        "FNFC Macro F1 (%)": 76.92
    },
    {
        "Algorithm": "AdaBoost",
        "Paper 1 DNA (%)": 73.78,
        "PROMISE-625 (%)": 71.77,
        "P-EXP-969 (%)": 67.80,
        "P-EXP Gain (%)": "-5.98%",
        "FNFC-7060 (%)": 86.40,
        "FNFC Macro F1 (%)": 61.20
    },
    {
        "Algorithm": "SVM Linear",
        "Paper 1 DNA (%)": 76.23,
        "PROMISE-625 (%)": 63.64,
        "P-EXP-969 (%)": 67.49,
        "P-EXP Gain (%)": "-8.74%",
        "FNFC-7060 (%)": 84.50,
        "FNFC Macro F1 (%)": 58.70
    },
    {
        "Algorithm": "Multinomial NB",
        "Paper 1 DNA (%)": 66.42,
        "PROMISE-625 (%)": 52.40,
        "P-EXP-969 (%)": 23.84,
        "P-EXP Gain (%)": "-42.58%",
        "FNFC-7060 (%)": 74.20,
        "FNFC Macro F1 (%)": 48.90
    },
    {
        "Algorithm": "Gaussian NB",
        "Paper 1 DNA (%)": 72.23,
        "PROMISE-625 (%)": 58.30,
        "P-EXP-969 (%)": 20.74,
        "P-EXP Gain (%)": "-51.49%",
        "FNFC-7060 (%)": 76.80,
        "FNFC Macro F1 (%)": 51.10
    },
    {
        "Algorithm": "SVM RBF",
        "Paper 1 DNA (%)": 75.34,
        "PROMISE-625 (%)": 62.10,
        "P-EXP-969 (%)": 12.38,
        "P-EXP Gain (%)": "-62.96%",
        "FNFC-7060 (%)": 82.30,
        "FNFC Macro F1 (%)": 55.40
    }
]

df = pd.DataFrame(master_data)
out_csv = 'results/tri_dataset_master_benchmark.csv'
df.to_csv(out_csv, index=False)
print(f"[+] Tri-Dataset Master CSV saved to: {out_csv}")
print(df.to_string(index=False))

# --- PLOT: 3-Dataset Publication Comparison ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 8.5))

y_pos = np.arange(len(df))
h = 0.20

rects1 = ax.barh(y_pos - 1.5*h, df['Paper 1 DNA (%)'], h, label='Paper 1 DNA Baseline', color='#95a5a6', alpha=0.85)
rects2 = ax.barh(y_pos - 0.5*h, df['PROMISE-625 (%)'], h, label='Paper 2 (PROMISE - 625)', color='#2980b9', alpha=0.95)
rects3 = ax.barh(y_pos + 0.5*h, df['P-EXP-969 (%)'], h, label='Paper 2 (P-EXP - 969)', color='#e67e22', alpha=0.95)
rects4 = ax.barh(y_pos + 1.5*h, df['FNFC-7060 (%)'], h, label='Paper 2 (FNFC - 7,060)', color='#27ae60', alpha=0.95)

ax.set_yticks(y_pos)
ax.set_yticklabels(df['Algorithm'], fontsize=11, fontweight='bold')
ax.set_xlabel('Classification Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Tri-Dataset Comparative Evaluation: Paper 1 vs. Paper 2 across PROMISE, P-EXP, & FNFC', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 105)
ax.legend(loc='lower right', fontsize=11, frameon=True)

# Annotate P-EXP and FNFC bars
for r in rects3:
    w = r.get_width()
    if w > 20:
        ax.annotate(f"{w:.1f}%", xy=(w, r.get_y() + r.get_height()/2), xytext=(4, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=8, fontweight='bold', color='#d35400')

plt.tight_layout()
out_plot = 'plots/tri_dataset_publication_benchmark.png'
plt.savefig(out_plot, dpi=300)
plt.close()
print(f"[+] Saved Tri-Dataset Plot to: {out_plot}")
