import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from custom_transformers import CustomTargetEncoder


def generate_eda_plots(df):
    """Generate and save 4 publication-quality plots to the plots/ folder."""
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Set matplotlib parameters for high quality
    plt.rcParams.update({
        'figure.titlesize': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'sans-serif'
    })

    # 1. crop_vs_production.png
    plt.figure(figsize=(12, 6), dpi=300)
    top10_crops = df.groupby('crop')['production'].mean().nlargest(10).reset_index()
    sns.barplot(
        data=top10_crops,
        x='production',
        y='crop',
        palette="viridis",
        hue='crop',
        legend=False
    )
    plt.title('Top 10 Highest-Producing Crops (Average Production)', weight='bold', pad=15)
    plt.xlabel('Average Production (Tons)')
    plt.ylabel('Crop')
    plt.tight_layout()
    plt.savefig('plots/crop_vs_production.png')
    plt.close()
    print("Saved plots/crop_vs_production.png")

    # 2. state_wise_production.png
    plt.figure(figsize=(12, 6), dpi=300)
    state_prod = df.groupby('state')['production'].sum().sort_values(ascending=False).reset_index()
    sns.barplot(
        data=state_prod,
        x='production',
        y='state',
        palette="rocket",
        hue='state',
        legend=False
    )
    plt.title('Total Crop Production by State', weight='bold', pad=15)
    plt.xlabel('Total Production (Tons)')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig('plots/state_wise_production.png')
    plt.close()
    print("Saved plots/state_wise_production.png")

    # 3. cost_vs_production.png
    plt.figure(figsize=(9, 6), dpi=300)
    sns.regplot(
        data=df,
        x='cost',
        y='production',
        scatter_kws={'alpha': 0.6, 'color': '#2A9D8F', 's': 45},
        line_kws={'color': '#E76F51', 'linewidth': 2.5}
    )
    plt.title('Cost of Cultivation vs. Crop Production', weight='bold', pad=15)
    plt.xlabel('Cost of Cultivation (Rs/Hectare)')
    plt.ylabel('Production (Tons)')
    plt.tight_layout()
    plt.savefig('plots/cost_vs_production.png')
    plt.close()
    print("Saved plots/cost_vs_production.png")

    # 4. season_wise_trends.png
    plt.figure(figsize=(9, 6), dpi=300)
    sns.boxplot(
        data=df,
        x='season',
        y='production',
        palette="Set2",
        hue='season',
        legend=False
    )
    plt.title('Crop Production Distribution by Season', weight='bold', pad=15)
    plt.xlabel('Season')
    plt.ylabel('Production (Tons)')
    plt.tight_layout()
    plt.savefig('plots/season_wise_trends.png')
    plt.close()
    print("Saved plots/season_wise_trends.png")


def main():
    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load cleaned dataset
    data_path = 'data/processed/cleaned_crop_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Please run data_pipeline.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape {df.shape}")

    # 2. Run EDA and save plots
    print("\nGenerating Exploratory Data Analysis (EDA) visualizations...")
    generate_eda_plots(df)

    # 3. Split features and target
    X = df[['crop', 'variety', 'state', 'season', 'cost', 'recommended_zone']]
    y = df['production']

    # 4. Train-test split (80% train, 20% test) before any transformations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

    # 5. Define ColumnTransformer for feature engineering
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['cost']),
            ('cat_low', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['crop', 'state', 'season']),
            ('cat_high', CustomTargetEncoder(smoothing=10.0), ['variety', 'recommended_zone'])
        ]
    )

    # 6. Train three distinct models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
    }

    results = []

    print("\n" + "=" * 80)
    print(f"{'Model Name':<30} | {'R2 Score':<12} | {'MAE':<12} | {'RMSE':<12}")
    print("-" * 80)

    for name, model in models.items():
        # Setup the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Train split fit
        pipeline.fit(X_train, y_train)

        # Test split predict
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            'model_name': name,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'pipeline': pipeline
        })

        print(f"{name:<30} | {r2:<12.4f} | {mae:<12.4f} | {rmse:<12.4f}")

    print("=" * 80)

    # 7. Model Selection: automatically pick based on highest R2
    best_model_info = max(results, key=lambda x: x['r2'])
    best_pipeline = best_model_info['pipeline']
    best_model_name = best_model_info['model_name']

    print(f"\nSelected Best Performing Model: {best_model_name}")
    print(f"Best Test R2 Score: {best_model_info['r2']:.4f}")

    # 8. Save the best model alongside its preprocessor
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/best_crop_model.pkl'
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best model pipeline to {model_save_path}")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULY")
    print("=" * 60)


if __name__ == '__main__':
    main()
