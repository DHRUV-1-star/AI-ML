import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. Load data
df = pd.read_csv('bdt_mba_supplychain_dataset_2024.csv')

# Prepare text outputs
output = {}
output['head'] = df.head().to_string()

output['shape'] = df.shape
output['columns'] = list(df.columns)

buffer = io.StringIO()
df.info(buf=buffer)
output['info'] = buffer.getvalue()

output['describe'] = df.describe().to_string()
output['missing'] = df.isnull().sum().to_dict()
output['duplicates'] = int(df.duplicated().sum())

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
output['num_cols'] = num_cols
output['cat_cols'] = cat_cols

# Print text outputs to be read by LLM
print("--- HEAD ---")
print(output['head'])
print("\n--- SHAPE ---")
print(output['shape'])
print("\n--- INFO ---")
print(output['info'])
print("\n--- DESCRIBE ---")
print(output['describe'])
print("\n--- MISSING ---")
print(output['missing'])
print("\n--- DUPLICATES ---")
print(output['duplicates'])
print("\n--- NUM_COLS ---")
print(output['num_cols'])
print("\n--- CAT_COLS ---")
print(output['cat_cols'])

# Visualizations
# 7. Histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Temperature'], kde=True, color='skyblue')
plt.title('Temperature Distribution')
plt.subplot(1, 2, 2)
sns.histplot(df['Logistics_Cost'], kde=True, color='salmon')
plt.title('Logistics Cost Distribution')
plt.tight_layout()
plt.savefig('7_histograms.png')
plt.close()

# 8. Countplots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Inventory_Level', order=['Low', 'Medium', 'High'], palette='viridis')
plt.title('Inventory Level Count')
plt.subplot(1, 2, 2)
sns.countplot(data=df, x='SupplyChain_Efficiency_Label', palette='Set2')
plt.title('Efficiency Label Count (0=Inefficient, 1=Efficient)')
plt.tight_layout()
plt.savefig('8_countplots.png')
plt.close()

# 9. Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Inventory_Level', y='Logistics_Cost', order=['Low', 'Medium', 'High'], palette='pastel')
plt.title('Logistics Cost by Inventory Level')
plt.tight_layout()
plt.savefig('9_boxplot.png')
plt.close()

# 10. Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Resource_Utilization', y='Delivery_Efficiency', hue='SupplyChain_Efficiency_Label', palette='coolwarm', alpha=0.7)
plt.title('Resource Utilization vs Delivery Efficiency')
plt.tight_layout()
plt.savefig('10_scatterplot.png')
plt.close()

# 11. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('11_heatmap.png')
plt.close()

# 12. Additional Visualization (Violin plot)
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='SupplyChain_Efficiency_Label', y='Condition_Score', palette='muted')
plt.title('Condition Score Distribution by Efficiency Label')
plt.tight_layout()
plt.savefig('12_violinplot.png')
plt.close()