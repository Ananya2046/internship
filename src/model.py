"""
model.py — Feature Engineering, Model Training & Evaluation

Trains Linear Regression, Decision Tree, and Random Forest regressors
to predict crop production using only:
  - Categorical features: crop, state, season
  - Numerical features: cost_of_cultivation_a2_fl, cost_of_cultivation_c2,
                         cost_of_production_c2
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_and_evaluate():
    """Load dataset, engineer features, train 3 models, and select the best."""
    print("Starting feature engineering and model training...")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # ── Load the clean merged dataset ──
    df = pd.read_csv('data/clean_merged_dataset.csv')

    # ── Define features and target ──
    # Strictly: Crop, State, Season, and Cost columns only
    categorical_features = ['crop', 'state', 'season']
    numerical_features = [
        'cost_of_cultivation_a2_fl',
        'cost_of_cultivation_c2',
        'cost_of_production_c2',
    ]
    feature_cols = categorical_features + numerical_features
    target_col = 'production'

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features used: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Dataset shape: {X.shape}")

    # ── Preprocessing pipeline ──
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_features),
        ]
    )

    # ── Train / test split (80-20) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── Models to evaluate ──
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(
            random_state=42, n_estimators=100, max_depth=12
        ),
    }

    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None

    # ── Train and evaluate each model ──
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'pipeline': pipeline}

        print(f"\n{name} Evaluation:")
        print(f"  R2 Score : {r2:.4f}")
        print(f"  MAE      : {mae:.4f}")
        print(f"  RMSE     : {rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    # ── Report best model ──
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}  (R2 = {best_r2:.4f})")
    print(f"{'='*50}")

    # ── Save the best pipeline ──
    with open('models/best_model_pipeline.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)
    print("Saved best model pipeline → models/best_model_pipeline.pkl")

    # ── Save evaluation table ──
    summary_rows = [
        {'Model': n, 'R2 Score': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE']}
        for n, m in results.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('data/model_evaluation.csv', index=False)
    print("Saved evaluation results → data/model_evaluation.csv")

    return best_model_name, results


if __name__ == '__main__':
    train_and_evaluate()
