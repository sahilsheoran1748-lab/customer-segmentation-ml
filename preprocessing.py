import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the customer dataset from a CSV file."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Remove duplicate records from the dataset."""
    return df.drop_duplicates().copy()


def select_features(df):
    """Select the features used for customer segmentation."""
    return df[[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]].copy()


def scale_features(features):
    """Standardize selected features for clustering."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler
