#!/usr/bin/env python3
"""
MOM2 Samples Ablation Study Visualization
Plots the impact of mom2_n_samples on BERT Score and ROUGE-L for MEMIT_ARE.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from ablation experiments (extracted from summary files)
mom2_values = [10, 100, 1000, 10000]
mom2_labels = ['10', '100', '1K', '10K']

# Llama3-8B-Instruct results (converted to percentage scale 0-100)
llama_original_rougeL = [65.2, 100.0, 84.5, 87.2]
llama_original_bert = [72.6, 100.0, 91.9, 93.1]
llama_para_rougeL = [1.0, 92.6, 77.2, 76.4]
llama_para_bert = [-1.5, 93.9, 92.1, 88.8]

# Qwen2.5-7B-Instruct results (converted to percentage scale 0-100)
qwen_original_rougeL = [66.3, 99.8, 99.6, 99.6]
qwen_original_bert = [65.0, 99.1, 99.2, 99.2]
qwen_para_rougeL = [0.2, 70.4, 71.0, 72.8]
qwen_para_bert = [-0.04, 87.5, 88.2, 90.4]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle('MOM2 Samples Ablation Study: BERT Score & ROUGE-L\n(MEMIT_ARE on UnKEBench)', 
             fontsize=14, fontweight='bold')

x_pos = np.arange(len(mom2_values))
bar_width = 0.35

# Color scheme
llama_color = '#2563eb'  # Blue
qwen_color = '#16a34a'   # Green

# ============ Plot 1: Original ROUGE-L ============
ax1 = axes[0, 0]
bars1 = ax1.bar(x_pos - bar_width/2, llama_original_rougeL, bar_width, 
                label='Llama3-8B', color=llama_color, alpha=0.85, edgecolor='white')
bars2 = ax1.bar(x_pos + bar_width/2, qwen_original_rougeL, bar_width, 
                label='Qwen2.5-7B', color=qwen_color, alpha=0.85, edgecolor='white')
ax1.set_ylabel('ROUGE-L (%)', fontsize=11, fontweight='bold')
ax1.set_title('Original Prediction', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(mom2_labels)
ax1.set_ylim(0, 115)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============ Plot 2: Original BERT Score ============
ax2 = axes[0, 1]
bars3 = ax2.bar(x_pos - bar_width/2, llama_original_bert, bar_width, 
                label='Llama3-8B', color=llama_color, alpha=0.85, edgecolor='white')
bars4 = ax2.bar(x_pos + bar_width/2, qwen_original_bert, bar_width, 
                label='Qwen2.5-7B', color=qwen_color, alpha=0.85, edgecolor='white')
ax2.set_ylabel('BERT Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('Original Prediction', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(mom2_labels)
ax2.set_ylim(0, 115)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============ Plot 3: Para ROUGE-L ============
ax3 = axes[1, 0]
bars5 = ax3.bar(x_pos - bar_width/2, llama_para_rougeL, bar_width, 
                label='Llama3-8B', color=llama_color, alpha=0.85, edgecolor='white')
bars6 = ax3.bar(x_pos + bar_width/2, qwen_para_rougeL, bar_width, 
                label='Qwen2.5-7B', color=qwen_color, alpha=0.85, edgecolor='white')
ax3.set_xlabel('mom2_n_samples', fontsize=11, fontweight='bold')
ax3.set_ylabel('ROUGE-L (%)', fontsize=11, fontweight='bold')
ax3.set_title('Paraphrase Prediction', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(mom2_labels)
ax3.set_ylim(0, 115)
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

for bar in bars5:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars6:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============ Plot 4: Para BERT Score ============
ax4 = axes[1, 1]
llama_para_bert_display = [max(0, v) for v in llama_para_bert]
qwen_para_bert_display = [max(0, v) for v in qwen_para_bert]

bars7 = ax4.bar(x_pos - bar_width/2, llama_para_bert_display, bar_width, 
                label='Llama3-8B', color=llama_color, alpha=0.85, edgecolor='white')
bars8 = ax4.bar(x_pos + bar_width/2, qwen_para_bert_display, bar_width, 
                label='Qwen2.5-7B', color=qwen_color, alpha=0.85, edgecolor='white')
ax4.set_xlabel('mom2_n_samples', fontsize=11, fontweight='bold')
ax4.set_ylabel('BERT Score (%)', fontsize=11, fontweight='bold')
ax4.set_title('Paraphrase Prediction', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(mom2_labels)
ax4.set_ylim(0, 115)
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_axisbelow(True)

for i, bar in enumerate(bars7):
    height = bar.get_height()
    val = llama_para_bert[i]
    label = f'{val:.1f}' if val >= 0 else f'{val:.1f}*'
    ax4.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
for i, bar in enumerate(bars8):
    height = bar.get_height()
    val = qwen_para_bert[i]
    label = f'{val:.1f}' if val >= 0 else f'{val:.1f}*'
    ax4.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

fig.text(0.5, 0.01, '*Negative scores indicate model destruction (gibberish output)', 
         ha='center', fontsize=9, style='italic', color='#666666')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/pg2860/anyedit/AnyEdit/output/mom2_ablation_results/mom2_ablation_bert_rougeL.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/home/pg2860/anyedit/AnyEdit/output/mom2_ablation_results/mom2_ablation_bert_rougeL.pdf', 
            bbox_inches='tight', facecolor='white')
print("Bar plots saved!")

# ============ Create trend line plot ============
fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('MOM2 Samples Ablation: Trend Analysis', fontsize=14, fontweight='bold')

# Left: ROUGE-L
ax_left.plot(mom2_values, llama_original_rougeL, 'o-', color=llama_color, linewidth=2.5, 
             markersize=10, label='Llama3-8B Original')
ax_left.plot(mom2_values, qwen_original_rougeL, 's-', color=qwen_color, linewidth=2.5, 
             markersize=10, label='Qwen2.5-7B Original')
ax_left.plot(mom2_values, llama_para_rougeL, 'o--', color=llama_color, linewidth=2, 
             markersize=8, alpha=0.6, label='Llama3-8B Para')
ax_left.plot(mom2_values, qwen_para_rougeL, 's--', color=qwen_color, linewidth=2, 
             markersize=8, alpha=0.6, label='Qwen2.5-7B Para')
ax_left.axvline(x=100, color='red', linestyle=':', linewidth=2.5, alpha=0.8, label='Optimal (100)')
ax_left.set_xscale('log')
ax_left.set_xlabel('mom2_n_samples (log scale)', fontsize=11, fontweight='bold')
ax_left.set_ylabel('ROUGE-L (%)', fontsize=11, fontweight='bold')
ax_left.set_title('ROUGE-L Trends', fontsize=12, fontweight='bold')
ax_left.set_ylim(-5, 110)
ax_left.legend(loc='lower right', fontsize=8)
ax_left.grid(alpha=0.3, linestyle='--')

# Right: BERT Score
ax_right.plot(mom2_values, llama_original_bert, 'o-', color=llama_color, linewidth=2.5, 
              markersize=10, label='Llama3-8B Original')
ax_right.plot(mom2_values, qwen_original_bert, 's-', color=qwen_color, linewidth=2.5, 
              markersize=10, label='Qwen2.5-7B Original')
ax_right.plot(mom2_values, llama_para_bert, 'o--', color=llama_color, linewidth=2, 
              markersize=8, alpha=0.6, label='Llama3-8B Para')
ax_right.plot(mom2_values, qwen_para_bert, 's--', color=qwen_color, linewidth=2, 
              markersize=8, alpha=0.6, label='Qwen2.5-7B Para')
ax_right.axvline(x=100, color='red', linestyle=':', linewidth=2.5, alpha=0.8, label='Optimal (100)')
ax_right.set_xscale('log')
ax_right.set_xlabel('mom2_n_samples (log scale)', fontsize=11, fontweight='bold')
ax_right.set_ylabel('BERT Score (%)', fontsize=11, fontweight='bold')
ax_right.set_title('BERT Score Trends', fontsize=12, fontweight='bold')
ax_right.set_ylim(-10, 110)
ax_right.legend(loc='lower right', fontsize=8)
ax_right.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/home/pg2860/anyedit/AnyEdit/output/mom2_ablation_results/mom2_ablation_trends.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/home/pg2860/anyedit/AnyEdit/output/mom2_ablation_results/mom2_ablation_trends.pdf', 
            bbox_inches='tight', facecolor='white')
print("Trend plots saved!")

# Print summary table
print("\n" + "="*90)
print("MOM2 SAMPLES ABLATION STUDY - SUMMARY (BERT Score & ROUGE-L)")
print("="*90)
print(f"\n{'Model':<15} {'mom2':<8} {'Ori ROUGE-L':<14} {'Ori BERT':<12} {'Para ROUGE-L':<14} {'Para BERT':<12}")
print("-"*90)
for i, m2 in enumerate(mom2_values):
    print(f"{'Llama3-8B':<15} {m2:<8} {llama_original_rougeL[i]:<14.1f} {llama_original_bert[i]:<12.1f} {llama_para_rougeL[i]:<14.1f} {llama_para_bert[i]:<12.1f}")
print()    
for i, m2 in enumerate(mom2_values):
    print(f"{'Qwen2.5-7B':<15} {m2:<8} {qwen_original_rougeL[i]:<14.1f} {qwen_original_bert[i]:<12.1f} {qwen_para_rougeL[i]:<14.1f} {qwen_para_bert[i]:<12.1f}")
print("-"*90)
print("\n🔑 KEY FINDINGS:")
print("   • mom2_n_samples=10:  MODEL DESTROYED - outputs gibberish (negative BERT scores)")
print("   • mom2_n_samples=100: OPTIMAL - near-perfect scores on original, good on paraphrase")
print("   • mom2_n_samples>100: Stable but slightly lower than 100 for original predictions")
