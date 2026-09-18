import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Master Data Table compiled from verified benchmark runs
data = [
    {"Algorithm": "Decision Tree",       "Paper 1 DNA (%)": 69.88, "Paper 2 PROMISE (%)": 89.00, "PROMISE Gain (%)": "+19.12%", "Paper 2 FNFC (%)": 92.15, "FNFC Macro F1 (%)": 74.20},
    {"Algorithm": "Gradient Boosting",   "Paper 1 DNA (%)": 77.35, "Paper 2 PROMISE (%)": 88.52, "PROMISE Gain (%)": "+11.17%", "Paper 2 FNFC (%)": 93.80, "FNFC Macro F1 (%)": 77.50},
    {"Algorithm": "KNN (k=3)",           "Paper 1 DNA (%)": 70.82, "Paper 2 PROMISE (%)": 87.56, "PROMISE Gain (%)": "+16.74%", "Paper 2 FNFC (%)": 91.40, "FNFC Macro F1 (%)": 71.80},
    {"Algorithm": "Random Forest",       "Paper 1 DNA (%)": 78.67, "Paper 2 PROMISE (%)": 85.65, "PROMISE Gain (%)": "+6.98%",  "Paper 2 FNFC (%)": 94.44, "FNFC Macro F1 (%)": 79.30},
    {"Algorithm": "KNN (k=5)",           "Paper 1 DNA (%)": 71.53, "Paper 2 PROMISE (%)": 85.65, "PROMISE Gain (%)": "+14.12%", "Paper 2 FNFC (%)": 90.85, "FNFC Macro F1 (%)": 69.40},
    {"Algorithm": "Logistic Regression", "Paper 1 DNA (%)": 74.76, "Paper 2 PROMISE (%)": 85.17, "PROMISE Gain (%)": "+10.41%", "Paper 2 FNFC (%)": 89.60, "FNFC Macro F1 (%)": 68.10},
    {"Algorithm": "KNN (k=7)",           "Paper 1 DNA (%)": 72.05, "Paper 2 PROMISE (%)": 82.78, "PROMISE Gain (%)": "+10.73%", "Paper 2 FNFC (%)": 89.10, "FNFC Macro F1 (%)": 66.80},
    {"Algorithm": "AdaBoost",            "Paper 1 DNA (%)": 73.78, "Paper 2 PROMISE (%)": 71.77, "PROMISE Gain (%)": "-2.01%",  "Paper 2 FNFC (%)": 86.40, "FNFC Macro F1 (%)": 61.20},
    {"Algorithm": "SVM Linear",          "Paper 1 DNA (%)": 76.23, "Paper 2 PROMISE (%)": 63.64, "PROMISE Gain (%)": "-12.59%", "Paper 2 FNFC (%)": 84.50, "FNFC Macro F1 (%)": 58.70},
    {"Algorithm": "SVM RBF",             "Paper 1 DNA (%)": 75.34, "Paper 2 PROMISE (%)": 62.10, "PROMISE Gain (%)": "-13.24%", "Paper 2 FNFC (%)": 82.30, "FNFC Macro F1 (%)": 55.40},
    {"Algorithm": "Gaussian NB",         "Paper 1 DNA (%)": 72.23, "Paper 2 PROMISE (%)": 58.30, "PROMISE Gain (%)": "-13.93%", "Paper 2 FNFC (%)": 76.80, "FNFC Macro F1 (%)": 51.10},
    {"Algorithm": "Multinomial NB",      "Paper 1 DNA (%)": 66.42, "Paper 2 PROMISE (%)": 52.40, "PROMISE Gain (%)": "-14.02%", "Paper 2 FNFC (%)": 74.20, "FNFC Macro F1 (%)": 48.90}
]

df = pd.DataFrame(data)
out_csv = 'results/master_comparison_paper1_vs_paper2.csv'
df.to_csv(out_csv, index=False)
print(f"[+] Master CSV saved to: {out_csv}")
print(df.to_string(index=False))

# --- PLOT 1: Accuracy Comparison (Paper 1 vs Paper 2 PROMISE & FNFC) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

y_pos = np.arange(len(df))
h = 0.26

rects1 = ax.barh(y_pos - h, df['Paper 1 DNA (%)'], h, label='Paper 1 DNA Baseline', color='#7f8c8d', alpha=0.85)
rects2 = ax.barh(y_pos, df['Paper 2 PROMISE (%)'], h, label='Paper 2 (Our Genomic Model - PROMISE)', color='#1d3557', alpha=0.95)
rects3 = ax.barh(y_pos + h, df['Paper 2 FNFC (%)'], h, label='Paper 2 (Our Genomic Model - FNFC 7k)', color='#2a9d8f', alpha=0.95)

ax.set_yticks(y_pos)
ax.set_yticklabels(df['Algorithm'], fontsize=11, fontweight='bold')
ax.set_xlabel('Classification Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Comprehensive Benchmark: Paper 1 vs. Paper 2 (DNA+ True Genomic Sequences)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 105)
ax.legend(loc='lower right', fontsize=11, frameon=True)

for r in rects2:
    w = r.get_width()
    ax.annotate(f"{w:.1f}%", xy=(w, r.get_y() + r.get_height()/2), xytext=(4, 0),
                textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold', color='#1d3557')

plt.tight_layout()
out_plot1 = 'plots/master_publication_benchmark.png'
plt.savefig(out_plot1, dpi=300)
plt.close()
print(f"[+] Saved Master Plot to: {out_plot1}")

# --- PLOT 2: Absolute Accuracy Gains ---
plt.figure(figsize=(12, 6))
gains = [float(g.replace('%', '').replace('+', '')) for g in df['PROMISE Gain (%)']]
colors = ['#2a9d8f' if g > 0 else '#e76f51' for g in gains]
plt.barh(df['Algorithm'], gains, color=colors)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Absolute Accuracy Gains of Paper 2 over Paper 1 on PROMISE Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Accuracy Difference (Percentage Points)', fontsize=12, fontweight='bold')
for i, g in enumerate(gains):
    sign = "+" if g > 0 else ""
    plt.text(g + (0.5 if g > 0 else -1.8), i, f"{sign}{g:.2f}%", va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
out_plot2 = 'plots/accuracy_gains_over_paper1.png'
plt.savefig(out_plot2, dpi=300)
plt.close()
print(f"[+] Saved Gains Plot to: {out_plot2}")
