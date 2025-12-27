import matplotlib.pyplot as plt
import numpy as np
import os

# Create results folder if it doesn't exist
os.makedirs('results', exist_ok=True)

# ============================================================
# DATA EXTRACTION FROM YOUR LOGS
# ============================================================

# Model 1: all-mpnet-base-v2 + flan-t5-base (WITHOUT LoRA)
model1_no_lora = {
    'name': 'MPNET-Base + T5-Base (No LoRA)',
    'company1': [3.1835, 2.2237, 1.8108, 1.4467, 1.0361, 0.8147, 0.7511, 0.7204, 
                 0.5502, 0.4457, 0.3225, 0.2654, 0.1868, 0.2294, 0.1251, 0.0999, 
                 0.0586, 0.0780, 0.0567, 0.0356, 0.2146, 0.0457, 0.0219, 0.0396, 0.0112],
    'company2': [3.5012, 2.8641, 2.4000, 2.0886, 1.8563, 1.4902, 1.1859, 0.9825, 
                 0.8333, 0.6365, 0.5616, 0.3844, 0.2937, 0.3138, 0.1927, 0.2868, 
                 0.1605, 0.1390, 0.0886, 0.0499, 0.0608, 0.0951, 0.0772, 0.0993, 0.1681],
    'final_loss': 0.0896,
    'communication_mb': 3777.74
}

# Model 1: all-mpnet-base-v2 + flan-t5-base (WITH LoRA) 
model1_with_lora = {
    'name': 'MPNET-Base + T5-Base (LoRA)',
    'company1': [2.9327, 2.8867, 3.1712, 2.7487, 2.8065, 2.9134, 2.9989, 2.9518, 
                 2.7973, 2.7007, 2.5638, 2.8368, 2.5771, 2.6449, 2.6346, 2.4187, 
                 2.4755, 2.2344, 2.3740, 2.4058, 2.4480, 2.3523, 2.3129, 2.1300, 2.2045],
    'company2': [3.4946, 3.3509, 3.3831, 3.4966, 3.1318, 3.1512, 3.2911, 3.4366, 
                 3.4962, 3.0866, 3.1891, 3.1129, 3.3305, 2.9540, 2.9510, 3.1334, 
                 3.1363, 2.9866, 3.0788, 2.9761, 2.7522, 2.9191, 2.8863, 2.8452, 2.8113],
    'final_loss': 2.5079,
    'communication_mb': 27.00
}

# Model 2: all-MiniLM-L6-v2 + flan-t5-small (WITHOUT LoRA)
model2_no_lora = {
    'name': 'MiniLM + T5-Small (No LoRA)',
    'company1': [3.5219, 3.0807, 2.7240, 2.4110, 2.0594, 1.9052, 1.7298, 1.3625, 
                 1.2873, 1.2594, 1.1430, 0.9960, 1.0347, 0.8264, 0.7569, 0.8051, 
                 0.6164, 0.6294, 0.5275, 0.5441, 0.3989, 0.5339, 0.4056, 0.3382, 0.2945],
    'company2': [3.7606, 3.3806, 3.1389, 2.8313, 2.7232, 2.5008, 2.3757, 2.1554, 
                 1.9635, 2.1351, 1.8135, 1.7163, 1.5464, 1.4815, 1.3824, 1.3046, 
                 1.1779, 1.1557, 1.1291, 1.0701, 1.0277, 0.9050, 0.8630, 0.6890, 0.6895],
    'final_loss': 0.4920,
    'communication_mb': 1174.33
}

# Model 2: all-MiniLM-L6-v2 + flan-t5-small (WITH LoRA)
model2_with_lora = {
    'name': 'MiniLM + T5-Small (LoRA)',
    'company1': [3.7967, 3.6995, 3.6653, 3.6265, 3.5616, 3.7108, 3.6835, 3.6475, 
                 3.5912, 3.2659, 3.2588, 3.1080, 3.3441, 3.4722, 3.4219, 3.3211, 
                 3.6499, 3.4030, 3.0768, 2.9628, 3.1036, 3.1284, 2.9550, 3.1656, 2.9479],
    'company2': [3.8175, 3.5457, 3.5157, 3.6815, 3.4237, 3.5823, 3.8099, 3.5991, 
                 3.6715, 3.5217, 3.5812, 3.5283, 3.1969, 3.3654, 3.6113, 3.5678, 
                 3.5011, 3.6394, 3.4483, 3.2837, 3.3087, 3.4680, 3.2500, 3.3565, 3.4010],
    'final_loss': 3.1744,
    'communication_mb': 10.50
}

# Model 3: paraphrase-multilingual-mpnet-base-v2 + flan-t5-large (WITHOUT LoRA)
model3_no_lora = {
    'name': 'Multilingual-MPNET + T5-Large (No LoRA)',
    'company1': [2.2428, 1.3970, 1.0944, 0.6142, 0.4826, 0.3442, 0.2125, 0.1185, 
                 0.1305, 0.0255, 0.0215, 0.0766, 0.0594, 0.0460, 0.0132, 0.0080, 
                 0.0013, 0.0229, 0.0201, 0.0095, 0.0514, 0.0372, 0.0012, 0.0011, 0.0042],
    'company2': [2.9846, 1.9657, 1.5080, 1.1314, 0.7077, 0.4783, 0.2910, 0.2310, 
                 0.1011, 0.1913, 0.0490, 0.0364, 0.0621, 0.0338, 0.0132, 0.0909, 
                 0.0523, 0.0617, 0.0259, 0.0085, 0.0148, 0.0249, 0.0127, 0.0033, 0.0063],
    'final_loss': 0.0053,
    'communication_mb': 11949.92
}

# Model 3: paraphrase-multilingual-mpnet-base-v2 + flan-t5-large (WITH LoRA)
model3_with_lora = {
    'name': 'Multilingual-MPNET + T5-Large (LoRA)',
    'company1': [2.1284, 2.1534, 2.2053, 2.2947, 2.2028, 2.2666, 2.0940, 2.1431, 
                 2.0915, 2.0220, 2.0794, 1.9179, 1.9975, 1.9498, 1.7341, 1.9134, 
                 1.7825, 1.5914, 1.6896, 1.5983, 1.5296, 1.5973, 1.6178, 1.4355, 1.5340],
    'company2': [2.6782, 3.1129, 2.7569, 2.7946, 2.5686, 2.8710, 2.9199, 2.5934, 
                 3.0323, 2.7955, 2.6651, 2.4341, 2.7618, 2.6676, 2.6142, 2.4450, 
                 2.5790, 2.4083, 2.4694, 2.4299, 2.2326, 2.2164, 2.0883, 2.1321, 2.0844],
    'final_loss': 1.8092,
    'communication_mb': 72.00
}

# ============================================================
# PLOT 1: Training Loss Curves for All Models
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Training Loss Curves: Model Comparison', fontsize=16, fontweight='bold')

models = [
    (model1_no_lora, model1_with_lora, 'Model 1: MPNET-Base + T5-Base'),
    (model2_no_lora, model2_with_lora, 'Model 2: MiniLM + T5-Small'),
    (model3_no_lora, model3_with_lora, 'Model 3: Multilingual-MPNET + T5-Large')
]

epochs = list(range(1, 26))

for col, (no_lora, with_lora, title) in enumerate(models):
    # Company 1
    ax1 = axes[0, col]
    ax1.plot(epochs, no_lora['company1'], 'b-', linewidth=2, label='Without LoRA', marker='o', markersize=4)
    ax1.plot(epochs, with_lora['company1'], 'r--', linewidth=2, label='With LoRA', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title(f'{title}\nCompany 1', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Company 2
    ax2 = axes[1, col]
    ax2.plot(epochs, no_lora['company2'], 'b-', linewidth=2, label='Without LoRA', marker='o', markersize=4)
    ax2.plot(epochs, with_lora['company2'], 'r--', linewidth=2, label='With LoRA', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_title(f'Company 2', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_loss_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/training_loss_comparison.png")
plt.close()

# ============================================================
# PLOT 2: Final Loss Comparison (Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

models_names = ['MPNET-Base\n+ T5-Base', 'MiniLM\n+ T5-Small', 'Multilingual-MPNET\n+ T5-Large']
no_lora_losses = [model1_no_lora['final_loss'], model2_no_lora['final_loss'], model3_no_lora['final_loss']]
with_lora_losses = [model1_with_lora['final_loss'], model2_with_lora['final_loss'], model3_with_lora['final_loss']]

x = np.arange(len(models_names))
width = 0.35

bars1 = ax.bar(x - width/2, no_lora_losses, width, label='Without LoRA', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, with_lora_losses, width, label='With LoRA', color='#e74c3c', edgecolor='black')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
ax.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
ax.set_title('Final Loss Comparison: LoRA vs Non-LoRA', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_names, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/final_loss_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/final_loss_comparison.png")
plt.close()

# ============================================================
# PLOT 3: Communication Cost vs Performance (Enhanced)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

models_full = [
    ('MPNET-Base + T5-Base', model1_no_lora['communication_mb'], model1_no_lora['final_loss'], 'No LoRA', '#3498db'),
    ('MPNET-Base + T5-Base', model1_with_lora['communication_mb'], model1_with_lora['final_loss'], 'LoRA', '#e74c3c'),
    ('MiniLM + T5-Small', model2_no_lora['communication_mb'], model2_no_lora['final_loss'], 'No LoRA', '#2ecc71'),
    ('MiniLM + T5-Small', model2_with_lora['communication_mb'], model2_with_lora['final_loss'], 'LoRA', '#f39c12'),
    ('Multilingual-MPNET + T5-Large', model3_no_lora['communication_mb'], model3_no_lora['final_loss'], 'No LoRA', '#9b59b6'),
    ('Multilingual-MPNET + T5-Large', model3_with_lora['communication_mb'], model3_with_lora['final_loss'], 'LoRA', '#e67e22'),
]

# Plot points with connecting lines between LoRA/No-LoRA pairs
for i in range(0, len(models_full), 2):
    name1, comm1, loss1, type1, color1 = models_full[i]
    name2, comm2, loss2, type2, color2 = models_full[i+1]
    
    # Draw connecting line
    ax.plot([comm1, comm2], [loss1, loss2], 'k--', alpha=0.3, linewidth=1.5, zorder=1)
    
    # Plot No LoRA point (circle)
    ax.scatter(comm1, loss1, s=300, marker='o', color=color1, 
              edgecolor='black', linewidth=2.5, zorder=3, alpha=0.9)
    
    # Plot LoRA point (star)
    ax.scatter(comm2, loss2, s=400, marker='*', color=color2, 
              edgecolor='black', linewidth=2, zorder=3, alpha=0.9)
    
    # Add labels with arrows
    ax.annotate(f'{name1}\n(No LoRA)\nLoss: {loss1:.4f}\nComm: {comm1:.0f} MB', 
                xy=(comm1, loss1), xytext=(20, 20), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color1, alpha=0.7, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))
    
    ax.annotate(f'{name2}\n(LoRA)\nLoss: {loss2:.4f}\nComm: {comm2:.0f} MB', 
                xy=(comm2, loss2), xytext=(-20, -30), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color2, alpha=0.7, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', lw=2))

# Custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Without LoRA',
           markerfacecolor='gray', markersize=12, markeredgecolor='black', markeredgewidth=2),
    Line2D([0], [0], marker='*', color='w', label='With LoRA',
           markerfacecolor='gray', markersize=15, markeredgecolor='black', markeredgewidth=2),
    Line2D([0], [0], color='black', linestyle='--', alpha=0.3, label='LoRA Impact', linewidth=2)
]

ax.legend(handles=legend_elements, fontsize=12, loc='upper left', 
         framealpha=0.95, edgecolor='black', shadow=True)

ax.set_xlabel('Communication Cost (MB) - Log Scale', fontsize=13, fontweight='bold')
ax.set_ylabel('Final Loss', fontsize=13, fontweight='bold')
ax.set_title('Communication Efficiency vs Model Performance\n(Lower is Better for Both Axes)', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_xscale('log')
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

# Add "optimal region" annotation
ax.axhspan(0, 0.5, alpha=0.05, color='green', zorder=0)
ax.text(0.02, 0.02, 'Optimal Performance Region', 
       transform=ax.transAxes, fontsize=11, style='italic',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

# Add vertical line showing communication efficiency threshold
ax.axvline(x=100, color='red', linestyle=':', linewidth=2, alpha=0.5, zorder=0)
ax.text(120, ax.get_ylim()[1]*0.95, 'Efficient\nCommunication\n< 100 MB', 
       fontsize=10, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('results/communication_vs_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/communication_vs_performance.png")
plt.close()

# ============================================================
# PLOT 4: Loss Reduction Rate (First 10 Epochs)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Loss Reduction Rate (First 10 Epochs)', fontsize=16, fontweight='bold')

for col, (no_lora, with_lora, title) in enumerate(models):
    ax = axes[col]
    
    # Average loss across both companies for first 10 epochs
    no_lora_avg = [(no_lora['company1'][i] + no_lora['company2'][i]) / 2 for i in range(10)]
    with_lora_avg = [(with_lora['company1'][i] + with_lora['company2'][i]) / 2 for i in range(10)]
    
    ax.plot(range(1, 11), no_lora_avg, 'b-', linewidth=3, label='Without LoRA', marker='o', markersize=8)
    ax.plot(range(1, 11), with_lora_avg, 'r--', linewidth=3, label='With LoRA', marker='s', markersize=8)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Loss', fontsize=11, fontweight='bold')
    ax.set_title(title.replace('Model ', '').replace(':', '\n'), fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/loss_reduction_rate.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/loss_reduction_rate.png")
plt.close()

# ============================================================
# PLOT 5: Summary Table (as image)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Create summary data
summary_data = [
    ['Model', 'LoRA', 'Final Loss', 'Comm. (MB)', 'Loss Reduction', 'Params (%)'],
    ['MPNET-Base + T5-Base', 'No', f"{model1_no_lora['final_loss']:.4f}", f"{model1_no_lora['communication_mb']:.2f}", '96.9%', '100%'],
    ['MPNET-Base + T5-Base', 'Yes', f"{model1_with_lora['final_loss']:.4f}", f"{model1_with_lora['communication_mb']:.2f}", '28.2%', '0.71%'],
    ['MiniLM + T5-Small', 'No', f"{model2_no_lora['final_loss']:.4f}", f"{model2_no_lora['communication_mb']:.2f}", '86.8%', '100%'],
    ['MiniLM + T5-Small', 'Yes', f"{model2_with_lora['final_loss']:.4f}", f"{model2_with_lora['communication_mb']:.2f}", '16.4%', '0.89%'],
    ['Multilingual-MPNET + T5-Large', 'No', f"{model3_no_lora['final_loss']:.4f}", f"{model3_no_lora['communication_mb']:.2f}", '99.8%', '100%'],
    ['Multilingual-MPNET + T5-Large', 'Yes', f"{model3_with_lora['final_loss']:.4f}", f"{model3_with_lora['communication_mb']:.2f}", '32.3%', '0.60%'],
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.1, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Style data rows
colors = ['#ecf0f1', '#d5dbdb']
for i in range(1, 7):
    for j in range(6):
        cell = table[(i, j)]
        cell.set_facecolor(colors[i % 2])
        cell.set_edgecolor('black')
        if j == 1:  # LoRA column
            if summary_data[i][j] == 'Yes':
                cell.set_facecolor('#e74c3c')
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#3498db')
                cell.set_text_props(color='white', weight='bold')

ax.set_title('Federated Learning Models: Performance Summary', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/performance_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/performance_summary_table.png")
plt.close()

# ============================================================
# PLOT 6-8: Individual Model Loss Curves (Separate Images)
# ============================================================

for idx, (no_lora, with_lora, title) in enumerate(models, 1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    model_title = title.replace('Model ', '').replace(':', '\n').replace(str(idx), '').strip()
    fig.suptitle(f'{model_title}\nTraining Loss Comparison', fontsize=16, fontweight='bold')
    
    # Company 1
    ax1 = axes[0]
    ax1.plot(epochs, no_lora['company1'], 'b-', linewidth=2.5, label='Without LoRA', 
             marker='o', markersize=6, markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=1.5)
    ax1.plot(epochs, with_lora['company1'], 'r--', linewidth=2.5, label='With LoRA', 
             marker='s', markersize=6, markerfacecolor='lightcoral', markeredgecolor='red', markeredgewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Company 1', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='best', framealpha=0.9, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xlim(0, 26)
    
    # Add text box with final losses
    textstr = f'Final Loss:\nNo LoRA: {no_lora["company1"][-1]:.4f}\nLoRA: {with_lora["company1"][-1]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Company 2
    ax2 = axes[1]
    ax2.plot(epochs, no_lora['company2'], 'b-', linewidth=2.5, label='Without LoRA', 
             marker='o', markersize=6, markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=1.5)
    ax2.plot(epochs, with_lora['company2'], 'r--', linewidth=2.5, label='With LoRA', 
             marker='s', markersize=6, markerfacecolor='lightcoral', markeredgecolor='red', markeredgewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Company 2', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='best', framealpha=0.9, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_xlim(0, 26)
    
    # Add text box with final losses
    textstr = f'Final Loss:\nNo LoRA: {no_lora["company2"][-1]:.4f}\nLoRA: {with_lora["company2"][-1]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save with model-specific filename
    filename = f'results/model{idx}_loss_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# ============================================================
# PLOT 9-11: Individual Model - Combined Average Loss
# ============================================================

for idx, (no_lora, with_lora, title) in enumerate(models, 1):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_title = title.replace('Model ', '').replace(':', '\n').replace(str(idx), '').strip()
    
    # Calculate average loss across both companies
    no_lora_avg = [(no_lora['company1'][i] + no_lora['company2'][i]) / 2 for i in range(25)]
    with_lora_avg = [(with_lora['company1'][i] + with_lora['company2'][i]) / 2 for i in range(25)]
    
    # Plot lines
    ax.plot(epochs, no_lora_avg, 'b-', linewidth=3, label='Without LoRA', 
            marker='o', markersize=7, markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2)
    ax.plot(epochs, with_lora_avg, 'r--', linewidth=3, label='With LoRA', 
            marker='s', markersize=7, markerfacecolor='lightcoral', markeredgecolor='red', markeredgewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_title}\nAverage Training Loss (Company 1 & 2)', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best', framealpha=0.95, edgecolor='black', shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_xlim(0, 26)
    
    # Add statistics box
    no_lora_improvement = (1 - no_lora_avg[-1] / no_lora_avg[0]) * 100
    with_lora_improvement = (1 - with_lora_avg[-1] / with_lora_avg[0]) * 100
    
    textstr = f'Statistics:\n'
    textstr += f'Without LoRA:\n  Initial: {no_lora_avg[0]:.4f}\n  Final: {no_lora_avg[-1]:.4f}\n  Improvement: {no_lora_improvement:.1f}%\n\n'
    textstr += f'With LoRA:\n  Initial: {with_lora_avg[0]:.4f}\n  Final: {with_lora_avg[-1]:.4f}\n  Improvement: {with_lora_improvement:.1f}%\n\n'
    textstr += f'Communication Cost:\n  No LoRA: {no_lora["communication_mb"]:.2f} MB\n  LoRA: {with_lora["communication_mb"]:.2f} MB'
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save with model-specific filename
    filename = f'results/model{idx}_average_loss.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

print("\n" + "="*60)
print("All plots successfully generated and saved in 'results/' folder!")
print("="*60)
print("\nGenerated files:")
print("  1. training_loss_comparison.png - All models overview")
print("  2. final_loss_comparison.png - Bar chart comparison")
print("  3. communication_vs_performance.png - Efficiency analysis")
print("  4. loss_reduction_rate.png - Early training behavior")
print("  5. performance_summary_table.png - Summary table")
print("  6-8. model1/2/3_loss_comparison.png - Individual model plots")
print("  9-11. model1/2/3_average_loss.png - Individual average plots")
print("="*60)