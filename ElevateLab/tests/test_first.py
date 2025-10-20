import pytest
import pandas as pd
import numpy as np
from First import normalize_features, process_data

def test_normalization():
    # Create sample test data
    test_data = {
        'Age': [22, 38, 26, 35],
        'Fare': [7.25, 71.28, 7.92, 53.1],
        'SibSp': [1, 1, 0, 1],
        'Parch': [0, 0, 0, 0],
        'Survived': [0, 1, 1, 1]  # Non-numerical column
    }
    df = pd.DataFrame(test_data)
    
    # Apply normalization
    normalized_df = normalize_features(df)
    
    # Test 1: Check if means are close to 0
    for col in ['Age', 'Fare', 'SibSp', 'Parch']:
        assert abs(normalized_df[col].mean()) < 0.0001
        
    # Test 2: Check if standard deviations are close to 1
    for col in ['Age', 'Fare', 'SibSp', 'Parch']:
        assert abs(normalized_df[col].std() - 1) < 0.0001
        
    # Test 3: Check if non-numerical columns are unchanged
    assert 'Survived' in normalized_df.columns
    assert (normalized_df['Survived'] == df['Survived']).all()

def test_complete_pipeline():
    # Test the entire data processing pipeline
    processed_df = process_data()
    
    # Verify numerical features are normalized
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    for col in numerical_cols:
        assert abs(processed_df[col].mean()) < 0.1  # Allow for small deviation
        assert abs(processed_df[col].std() - 1) < 0.1