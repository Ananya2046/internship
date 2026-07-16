"""
predict.py — Inference / Prediction Function

Loads the saved best model pipeline and predicts crop production
from only: crop, state, season, and cost inputs.
"""

import pickle
import pandas as pd


# ── Cached model pipeline ──
_pipeline = None


def load_model():
    """Load and cache the trained model pipeline from disk."""
    global _pipeline
    if _pipeline is None:
        try:
            with open('models/best_model_pipeline.pkl', 'rb') as f:
                _pipeline = pickle.load(f)
        except Exception as e:
            raise FileNotFoundError(
                "Could not load best model pipeline. "
                f"Make sure model training was run first. Error: {e}"
            )
    return _pipeline


def predict_crop_production(
    crop: str,
    state: str,
    season: str,
    cost_a2_fl: float,
    cost_c2: float,
    cost_prod: float,
) -> float:
    """
    Predict crop production using the trained model pipeline.

    Parameters
    ----------
    crop : str
        Name of the crop (e.g., 'Rice', 'Wheat').
    state : str
        Indian state (e.g., 'Uttar Pradesh', 'Karnataka').
    season : str
        Season or duration (e.g., 'Kharif', 'Medium', '152').
    cost_a2_fl : float
        Cost of Cultivation A2+FL (Rs / Hectare).
    cost_c2 : float
        Cost of Cultivation C2 (Rs / Hectare).
    cost_prod : float
        Cost of Production C2 (Rs / Quintal).

    Returns
    -------
    float
        Predicted crop production value.
    """
    pipeline = load_model()

    # Build a single-row DataFrame matching the training feature names
    input_df = pd.DataFrame({
        'crop': [str(crop).strip().title()],
        'state': [str(state).strip().title()],
        'season': [str(season).strip().title()],
        'cost_of_cultivation_a2_fl': [float(cost_a2_fl)],
        'cost_of_cultivation_c2': [float(cost_c2)],
        'cost_of_production_c2': [float(cost_prod)],
    })

    prediction = pipeline.predict(input_df)[0]
    return float(prediction)


if __name__ == '__main__':
    # ── Sample prediction test ──
    try:
        result = predict_crop_production(
            crop='Wheat',
            state='Punjab',
            season='Rabi',
            cost_a2_fl=8000,
            cost_c2=20000,
            cost_prod=1500,
        )

        print("\n=== Sample Prediction Test ===")
        print("Inputs:")
        print("  Crop   : Wheat")
        print("  State  : Punjab")
        print("  Season : Rabi")
        print("  Cost A2+FL : 8000")
        print("  Cost C2    : 20000")
        print("  Cost Prod  : 1500")
        print(f"\nPredicted Production: {result:.4f}")
        print("==============================\n")

    except Exception as e:
        print(f"Prediction failed: {e}")