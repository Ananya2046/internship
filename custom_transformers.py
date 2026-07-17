import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Target Encoder with Laplace smoothing to prevent target leakage.
    Implements standard fit/transform API compatible with ColumnTransformer.
    """
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.mappings_ = {}
        self.global_mean_ = 0.0

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        y_ser = pd.Series(y)
        self.global_mean_ = y_ser.mean()
        self.mappings_ = {}

        for col in X_df.columns:
            # Group y values by categories in X_df[col]
            stats = y_ser.groupby(X_df[col]).agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            # Apply smoothed target value formula
            smoothed = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            self.mappings_[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_out = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            mapping = self.mappings_.get(col, {})
            # Map values; fallback to global training mean for unseen categories
            X_out[col] = X_df[col].map(mapping).fillna(self.global_mean_)
        return X_out.values
