"""
main.py — End-to-End ML Pipeline Orchestrator

Runs all stages of the Agriculture Crop Production prediction project:
  1. Data preprocessing & merging
  2. Exploratory Data Analysis (EDA)
  3. Model training & evaluation
  4. Sample prediction
"""

import sys
from src.data_preprocessing import load_and_preprocess_raw, build_clean_merged_dataset
from src.eda import run_eda
from src.model import train_and_evaluate
from src.predict import predict_crop_production


def main():
    print("=" * 60)
    print("Prediction of Agriculture Crop Production in India")
    print("=" * 60)

    # ── Step 1: Preprocessing & Merging ──
    print("\n--- STEP 1: Preprocessing & Merging ---")
    df1, df2, df3, df4, df5 = load_and_preprocess_raw()
    build_clean_merged_dataset(df1, df2, df3, df4, df5)

    # ── Step 2: Exploratory Data Analysis ──
    print("\n--- STEP 2: Exploratory Data Analysis (EDA) ---")
    run_eda()

    # ── Step 3: Model Training & Evaluation ──
    print("\n--- STEP 3: Model Training & Evaluation ---")
    best_model_name, results = train_and_evaluate()

    # ── Step 4: Sample Prediction ──
    print("\n--- STEP 4: Sample Prediction ---")
    try:
        pred = predict_crop_production(
            crop='Rice',
            state='Uttar Pradesh',
            season='152',
            cost_a2_fl=9794.05,
            cost_c2=23076.74,
            cost_prod=1941.55,
        )
        print("Prediction successful:")
        print(f"  Crop: Rice | State: Uttar Pradesh | Season: 152")
        print(f"  Cost A2+FL: 9794.05 | Cost C2: 23076.74 | Cost Prod: 1941.55")
        print(f"  → Predicted Production: {pred:.4f} Tonnes")
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
