"""
Tests for the fivedreg.model module.
"""

import tempfile
import os

import numpy as np
import pytest
import torch

from fivedreg import FiveDRegressor, MLP, create_sample_dataset


class TestMLP:
    """Tests for the MLP class."""

    def test_create_default(self):
        """Test creating MLP with default parameters."""
        model = MLP()
        assert model.in_dim == 5
        assert model.hidden == 64
        assert model.depth == 3
        assert model.out_dim == 1

    def test_create_custom(self):
        """Test creating MLP with custom parameters."""
        model = MLP(in_dim=5, hidden=32, depth=2, out_dim=1)
        assert model.hidden == 32
        assert model.depth == 2

    def test_forward_pass(self):
        """Test forward pass through the network."""
        model = MLP()
        x = torch.randn(32, 5)
        y = model(x)
        assert y.shape == (32, 1)

    def test_count_parameters(self):
        """Test counting model parameters."""
        model = MLP(hidden=64, depth=3)
        n_params = model.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)


class TestFiveDRegressor:
    """Tests for the FiveDRegressor class."""

    def test_create_default(self):
        """Test creating regressor with default parameters."""
        model = FiveDRegressor()
        assert model.hidden_layers == [64, 32, 16]
        assert model.learning_rate == 5e-3
        assert model.max_epochs == 200

    def test_create_custom(self):
        """Test creating regressor with custom parameters."""
        model = FiveDRegressor(
            hidden_layers=[32, 16],
            learning_rate=0.01,
            max_epochs=50
        )
        assert model.hidden_layers == [32, 16]
        assert model.learning_rate == 0.01
        assert model.max_epochs == 50

    def test_fit(self, small_dataset):
        """Test model training."""
        model = FiveDRegressor(
            hidden_layers=[32, 16],
            max_epochs=10,
            verbose=False,
            random_state=42
        )
        
        result = model.fit(small_dataset.X, small_dataset.y)
        
        assert result is model  # Returns self
        assert model.model is not None
        assert len(model.train_losses) > 0
        assert len(model.val_losses) > 0

    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        model = FiveDRegressor()
        X = np.random.randn(10, 5).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(X)

    def test_predict(self, trained_model, small_dataset):
        """Test prediction on trained model."""
        predictions = trained_model.predict(small_dataset.X)
        
        assert predictions.shape == (len(small_dataset),)
        assert predictions.dtype == np.float32 or predictions.dtype == np.float64

    def test_predict_single(self, trained_model):
        """Test prediction on single sample."""
        X = np.random.randn(1, 5).astype(np.float32)
        predictions = trained_model.predict(X)
        
        assert predictions.shape == (1,)

    def test_score(self, trained_model, small_dataset):
        """Test scoring method."""
        scores = trained_model.score(small_dataset.X, small_dataset.y)
        
        assert 'mse' in scores
        assert 'rmse' in scores
        assert 'mae' in scores
        assert 'r2' in scores
        
        assert scores['mse'] >= 0
        assert scores['rmse'] >= 0
        assert scores['mae'] >= 0
        assert scores['r2'] <= 1

    def test_get_params(self):
        """Test getting model parameters."""
        model = FiveDRegressor(
            hidden_layers=[32, 16],
            learning_rate=0.01
        )
        params = model.get_params()
        
        assert params['hidden_layers'] == [32, 16]
        assert params['learning_rate'] == 0.01

    def test_save_and_load(self, trained_model, small_dataset):
        """Test saving and loading model."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            trained_model.save(temp_path)
            assert os.path.exists(temp_path)
            
            # Load
            new_model = FiveDRegressor()
            new_model.load(temp_path)
            
            # Compare predictions
            pred1 = trained_model.predict(small_dataset.X)
            pred2 = new_model.predict(small_dataset.X)
            
            assert np.allclose(pred1, pred2)
        finally:
            os.unlink(temp_path)

    def test_reproducibility(self, small_dataset):
        """Test training reproducibility with same random state."""
        model1 = FiveDRegressor(
            hidden_layers=[32, 16],
            max_epochs=10,
            verbose=False,
            random_state=42
        )
        model1.fit(small_dataset.X, small_dataset.y)
        
        model2 = FiveDRegressor(
            hidden_layers=[32, 16],
            max_epochs=10,
            verbose=False,
            random_state=42
        )
        model2.fit(small_dataset.X, small_dataset.y)
        
        pred1 = model1.predict(small_dataset.X)
        pred2 = model2.predict(small_dataset.X)
        
        assert np.allclose(pred1, pred2, rtol=1e-5)

    def test_early_stopping(self, small_dataset):
        """Test that early stopping works."""
        model = FiveDRegressor(
            hidden_layers=[32, 16],
            max_epochs=1000,  # High max epochs
            patience=5,  # Low patience
            verbose=False,
            random_state=42
        )
        model.fit(small_dataset.X, small_dataset.y)
        
        # Should stop before max_epochs due to early stopping
        assert len(model.train_losses) < 1000


class TestModelPerformance:
    """Tests for model performance requirements."""

    def test_training_time_small(self):
        """Test that training completes quickly for small datasets."""
        import time
        
        dataset = create_sample_dataset(n_samples=1000, random_state=42)
        model = FiveDRegressor(
            hidden_layers=[64, 32, 16],
            max_epochs=50,
            verbose=False
        )
        
        start = time.time()
        model.fit(dataset.X, dataset.y)
        elapsed = time.time() - start
        
        # Should complete in under 30 seconds for 1K samples
        assert elapsed < 30, f"Training took {elapsed:.1f}s, expected < 30s"

    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy."""
        dataset = create_sample_dataset(n_samples=1000, random_state=42)
        train, val, test = dataset.split(random_state=42)
        
        model = FiveDRegressor(
            hidden_layers=[64, 32, 16],
            max_epochs=100,
            verbose=False,
            random_state=42
        )
        model.fit(train.X, train.y)
        
        scores = model.score(test.X, test.y)
        
        # Should achieve at least R² > 0.5 on the sample dataset
        assert scores['r2'] > 0.5, f"R² = {scores['r2']:.3f}, expected > 0.5"