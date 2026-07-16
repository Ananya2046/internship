import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for premium visualizations
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

# Custom premium palette
colors = ["#2A9D8F", "#E76F51", "#F4A261", "#E9C46A", "#264653"]

def run_eda():
    print("Running Exploratory Data Analysis (EDA)...")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Load dataset
    df = pd.read_csv('data/clean_merged_dataset.csv')
    
    # 1. Crop vs Production
    plt.figure(figsize=(12, 7), dpi=300)
    crop_prod = df.groupby('crop')['production'].mean().reset_index().sort_values(by='production', ascending=False)
    sns.barplot(data=crop_prod, x='production', y='crop', palette="viridis", hue='crop', legend=False)
    plt.title('Average Production by Crop type', pad=20, weight='bold')
    plt.xlabel('Average Production (in Tonnes/Units)', labelpad=10)
    plt.ylabel('Crop', labelpad=10)
    plt.tight_layout()
    plt.savefig('visualizations/crop_vs_production.png')
    plt.close()
    print("Saved visualizations/crop_vs_production.png")
    
    # 2. State-wise Production
    plt.figure(figsize=(12, 7), dpi=300)
    state_prod = df.groupby('state')['production'].sum().reset_index().sort_values(by='production', ascending=False)
    sns.barplot(data=state_prod, x='production', y='state', palette="rocket", hue='state', legend=False)
    plt.title('Total Production by State', pad=20, weight='bold')
    plt.xlabel('Total Production (in Tonnes/Units)', labelpad=10)
    plt.ylabel('State', labelpad=10)
    plt.tight_layout()
    plt.savefig('visualizations/state_wise_production.png')
    plt.close()
    print("Saved visualizations/state_wise_production.png")
    
    # 3. Cost vs Production
    plt.figure(figsize=(10, 6), dpi=300)
    sns.regplot(
        data=df, 
        x='cost_of_cultivation_c2', 
        y='production',
        scatter_kws={'alpha':0.6, 'color': '#2A9D8F', 's': 50},
        line_kws={'color': '#E76F51', 'linewidth': 3}
    )
    plt.title('Cost of Cultivation (C2) vs Production', pad=20, weight='bold')
    plt.xlabel('Cost of Cultivation (C2 in Rs/Hectare)', labelpad=10)
    plt.ylabel('Production (in Tonnes/Units)', labelpad=10)
    plt.tight_layout()
    plt.savefig('visualizations/cost_vs_production.png')
    plt.close()
    print("Saved visualizations/cost_vs_production.png")
    
    # 4. Season-wise Trends
    plt.figure(figsize=(10, 6), dpi=300)
    # Group by season and plot average production
    season_prod = df.groupby('season')['production'].mean().reset_index().sort_values(by='production', ascending=False)
    sns.barplot(data=season_prod, x='season', y='production', palette="crest", hue='season', legend=False)
    plt.title('Average Production by Season', pad=20, weight='bold')
    plt.xlabel('Season / Duration in Days', labelpad=10)
    plt.ylabel('Average Production (in Tonnes/Units)', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/season_wise_trends.png')
    plt.close()
    print("Saved visualizations/season_wise_trends.png")
    
    print("EDA completed successfully.")

if __name__ == '__main__':
    run_eda()
