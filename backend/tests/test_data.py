"""
Tests for the fivedreg.data module.
"""

import pickle
import tempfile
import os

import numpy as np
import pytest

from fivedreg import (
    FiveDDataset,
    DataValidationError,
    load_dataset,
    create_sample_dataset,
)


class TestFiveDDataset:
    """Tests for the FiveDDataset class."""

    def test_create_valid_dataset(self, sample_X, sample_y):
        """Test creating a dataset with valid data."""
        dataset = FiveDDataset(sample_X, sample_y)
        assert len(dataset) == 100
        assert dataset.X.shape == (100, 5)
        assert dataset.y.shape == (100,)

    def test_invalid_x_dimensions(self, sample_y):
        """Test that invalid X dimensions raise an error."""
        X_invalid = np.random.randn(100, 3)  # Wrong number of features
        with pytest.raises(DataValidationError, match="5 features"):
            FiveDDataset(X_invalid, sample_y)

    def test_invalid_x_ndim(self, sample_y):
        """Test that 1D X raises an error."""
        X_invalid = np.random.randn(100)  # 1D array
        with pytest.raises(DataValidationError, match="2-dimensional"):
            FiveDDataset(X_invalid, sample_y)

    def test_mismatched_samples(self, sample_X):
        """Test that mismatched X and y sizes raise an error."""
        y_invalid = np.random.randn(50)  # Different number of samples
        with pytest.raises(DataValidationError, match="same number of samples"):
            FiveDDataset(sample_X, y_invalid)

    def test_empty_dataset(self):
        """Test that empty dataset raises an error."""
        X_empty = np.zeros((0, 5))
        y_empty = np.zeros(0)
        with pytest.raises(DataValidationError, match="cannot be empty"):
            FiveDDataset(X_empty, y_empty)

    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy."""
        X = np.array([[1, 2, 3, 4, 5], [np.nan, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.float32)
        
        dataset = FiveDDataset(X, y)
        dataset.handle_missing_values(strategy="mean")
        
        assert not np.isnan(dataset.X).any()
        assert dataset.X[1, 0] == 1.0  # Mean of [1, 1]

    def test_handle_missing_values_drop(self):
        """Test handling missing values with drop strategy."""
        X = np.array([[1, 2, 3, 4, 5], [np.nan, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.float32)
        
        dataset = FiveDDataset(X, y)
        dataset.handle_missing_values(strategy="drop")
        
        assert len(dataset) == 2
        assert not np.isnan(dataset.X).any()

    def test_standardize(self, sample_dataset):
        """Test feature standardization."""
        sample_dataset.standardize()
        
        assert sample_dataset.is_standardized
        assert sample_dataset.scaler is not None
        # Check mean is approximately 0 and std is approximately 1
        assert np.allclose(sample_dataset.X.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(sample_dataset.X.std(axis=0), 1, atol=1e-6)

    def test_split_default(self, sample_dataset):
        """Test default data splitting."""
        train, val, test = sample_dataset.split()
        
        total = len(train) + len(val) + len(test)
        assert total == len(sample_dataset)
        
        # Check approximate proportions (70/10/20)
        assert len(test) == int(0.2 * len(sample_dataset))

    def test_split_custom_sizes(self, sample_dataset):
        """Test data splitting with custom sizes."""
        train, val, test = sample_dataset.split(test_size=0.3, val_size=0.2)
        
        total = len(train) + len(val) + len(test)
        assert total == len(sample_dataset)

    def test_split_invalid_sizes(self, sample_dataset):
        """Test that invalid split sizes raise an error."""
        with pytest.raises(ValueError, match="must be less than 1.0"):
            sample_dataset.split(test_size=0.6, val_size=0.5)

    def test_repr(self, sample_dataset):
        """Test string representation."""
        repr_str = repr(sample_dataset)
        assert "FiveDDataset" in repr_str
        assert "n_samples=500" in repr_str


class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_load_dict_format(self, temp_pkl_file):
        """Test loading dataset from dict format."""
        dataset = load_dataset(temp_pkl_file)
        assert isinstance(dataset, FiveDDataset)
        assert len(dataset) == 500

    def test_load_tuple_format(self):
        """Test loading dataset from tuple format."""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump((X, y), f)
            temp_path = f.name
        
        try:
            dataset = load_dataset(temp_path)
            assert isinstance(dataset, FiveDDataset)
            assert len(dataset) == 100
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/data.pkl")

    def test_load_invalid_format(self):
        """Test loading invalid file format raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid data")
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Invalid file format"):
                load_dataset(temp_path)
        finally:
            os.unlink(temp_path)


class TestCreateSampleDataset:
    """Tests for the create_sample_dataset function."""

    def test_create_default(self):
        """Test creating sample dataset with defaults."""
        dataset = create_sample_dataset()
        assert len(dataset) == 1000
        assert dataset.X.shape == (1000, 5)

    def test_create_custom_size(self):
        """Test creating sample dataset with custom size."""
        dataset = create_sample_dataset(n_samples=500)
        assert len(dataset) == 500

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        dataset1 = create_sample_dataset(n_samples=100, random_state=42)
        dataset2 = create_sample_dataset(n_samples=100, random_state=42)
        
        assert np.allclose(dataset1.X, dataset2.X)
        assert np.allclose(dataset1.y, dataset2.y)