import os
import sys
import joblib
import pandas as pd

# Import CustomTargetEncoder so pickle can locate it during deserialization
from custom_transformers import CustomTargetEncoder


def predict_crop_production(input_dict):
    """
    Predict crop production in Tons based on input agricultural features.

    Parameters
    ----------
    input_dict : dict
        A dictionary containing the following keys:
        - 'crop': Crop name (e.g. 'Rice')
        - 'variety': Crop variety (e.g. 'Hybrid-A')
        - 'state': State name (e.g. 'Punjab')
        - 'season': Growing season (e.g. 'Rabi')
        - 'cost': Cost of cultivation (Rs/Hectare)
        - 'recommended_zone': Recommended cultivation zone (e.g. 'Northern Plains')

    Returns
    -------
    float
        Predicted production in Tons.
    """
    model_path = 'models/best_crop_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'train_model.py' to train and save the model first."
        )

    # Load the best model pipeline (includes preprocessing)
    pipeline = joblib.load(model_path)

    # Standardize input values to Title Case to align with cleaning process in pipeline
    standardized_input = {
        'crop': str(input_dict.get('crop', 'Unknown')).strip().title(),
        'variety': str(input_dict.get('variety', 'Unknown')).strip().title(),
        'state': str(input_dict.get('state', 'Unknown')).strip().title(),
        'season': str(input_dict.get('season', 'Unknown')).strip().title(),
        'cost': float(input_dict.get('cost', 0.0)),
        'recommended_zone': str(input_dict.get('recommended_zone', 'Unknown')).strip().title()
    }

    # Convert single input dict to a DataFrame to match features expected by pipeline
    input_df = pd.DataFrame([standardized_input])

    # Predict using the loaded pipeline
    prediction = pipeline.predict(input_df)[0]

    # Crop production cannot be negative, so bound at 0.0
    return max(0.0, float(prediction))


if __name__ == '__main__':
    print("=" * 50)
    print("RUNNING PREDICTION TEST BLOCK")
    print("=" * 50)

    # Sample input matching the schema
    sample_input = {
        'crop': 'Wheat',
        'variety': 'Sharbati',
        'state': 'Punjab',
        'season': 'Rabi',
        'cost': 22000.0,
        'recommended_zone': 'Northern Plains'
    }

    try:
        predicted_tons = predict_crop_production(sample_input)
        print("\nInput Features:")
        for k, v in sample_input.items():
            print(f"  {k:<18}: {v}")
        print(f"\n→ Predicted Crop Production: {predicted_tons:.4f} Tons")
    except Exception as e:
        print(f"\nPrediction failed with error: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("PREDICTION TEST SUCCESSFUL")
    print("=" * 50)
