"""
Pytest configuration and fixtures for the fivedreg test suite.
"""

import os
import sys
import tempfile
import pickle
from pathlib import Path

import numpy as np
import pytest

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fivedreg import FiveDDataset, FiveDRegressor, create_sample_dataset


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return create_sample_dataset(n_samples=500, random_state=42)


@pytest.fixture
def small_dataset():
    """Create a small dataset for quick tests."""
    return create_sample_dataset(n_samples=100, random_state=42)


@pytest.fixture
def trained_model(small_dataset):
    """Create a trained model for testing."""
    model = FiveDRegressor(
        hidden_layers=[32, 16],
        max_epochs=10,
        patience=5,
        verbose=False,
        random_state=42
    )
    model.fit(small_dataset.X, small_dataset.y)
    return model


@pytest.fixture
def temp_pkl_file(sample_dataset):
    """Create a temporary .pkl file with sample data."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump({'X': sample_dataset.X, 'y': sample_dataset.y}, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_X():
    """Create sample feature array."""
    np.random.seed(42)
    return np.random.randn(100, 5).astype(np.float32)


@pytest.fixture
def sample_y():
    """Create sample target array."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float32)