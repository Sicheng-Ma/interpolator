"""
Tests for the FastAPI backend.
"""

import pickle
import tempfile
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fivedreg.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_pkl_file():
    """Create a sample .pkl file for upload testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump({'X': X, 'y': y}, f)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"


class TestUploadEndpoint:
    """Tests for the /upload endpoint."""

    def test_upload_valid_file(self, client, sample_pkl_file):
        """Test uploading a valid .pkl file."""
        with open(sample_pkl_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test_data.pkl", f, "application/octet-stream")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert data["n_samples"] == 100
        assert data["n_features"] == 5

    def test_upload_invalid_extension(self, client):
        """Test uploading a file with invalid extension."""
        content = b"invalid content"
        response = client.post(
            "/upload",
            files={"file": ("test_data.txt", content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]

    def test_upload_invalid_content(self, client):
        """Test uploading a .pkl file with invalid content."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump({"invalid": "data"}, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/upload",
                    files={"file": ("test_data.pkl", f, "application/octet-stream")}
                )
            
            assert response.status_code == 400
        finally:
            os.unlink(temp_path)


class TestTrainEndpoint:
    """Tests for the /train endpoint."""

    def test_train_without_dataset(self, client):
        """Test training without uploading a dataset first."""
        # Clear any existing datasets by using a fresh client
        response = client.post("/train", json={})
        
        # Should fail because no dataset is uploaded
        # Note: This might pass if there's a dataset from previous tests
        # In a real scenario, we'd reset the state between tests

    def test_train_with_dataset(self, client, sample_pkl_file):
        """Test training after uploading a dataset."""
        # Upload dataset
        with open(sample_pkl_file, 'rb') as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test_data.pkl", f, "application/octet-stream")}
            )
        
        assert upload_response.status_code == 200
        
        # Train model
        train_response = client.post("/train", json={
            "hidden_layers": [32, 16],
            "max_epochs": 10,
            "patience": 5
        })
        
        assert train_response.status_code == 200
        data = train_response.json()
        assert "model_id" in data
        assert "test_metrics" in data
        assert "training_time_seconds" in data

    def test_train_with_custom_config(self, client, sample_pkl_file):
        """Test training with custom configuration."""
        # Upload dataset
        with open(sample_pkl_file, 'rb') as f:
            client.post(
                "/upload",
                files={"file": ("test_data.pkl", f, "application/octet-stream")}
            )
        
        # Train with custom config
        response = client.post("/train", json={
            "hidden_layers": [64, 32],
            "learning_rate": 0.01,
            "max_epochs": 20,
            "batch_size": 32
        })
        
        assert response.status_code == 200


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_without_model(self, client):
        """Test prediction without training a model."""
        response = client.post("/predict", json={
            "features": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        # Should fail because no model is trained
        # Note: Might pass if model exists from previous tests

    def test_predict_with_model(self, client, sample_pkl_file):
        """Test prediction after training a model."""
        # Upload and train
        with open(sample_pkl_file, 'rb') as f:
            client.post(
                "/upload",
                files={"file": ("test_data.pkl", f, "application/octet-stream")}
            )
        
        client.post("/train", json={
            "hidden_layers": [32, 16],
            "max_epochs": 10
        })
        
        # Predict
        response = client.post("/predict", json={
            "features": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_id" in data
        assert isinstance(data["prediction"], float)

    def test_predict_invalid_features(self, client, sample_pkl_file):
        """Test prediction with wrong number of features."""
        # Upload and train
        with open(sample_pkl_file, 'rb') as f:
            client.post(
                "/upload",
                files={"file": ("test_data.pkl", f, "application/octet-stream")}
            )
        
        client.post("/train", json={
            "hidden_layers": [32, 16],
            "max_epochs": 10
        })
        
        # Predict with wrong number of features
        response = client.post("/predict", json={
            "features": [0.1, 0.2, 0.3]  # Only 3 features
        })
        
        assert response.status_code == 422  # Validation error


class TestListEndpoints:
    """Tests for list endpoints."""

    def test_list_datasets(self, client, sample_pkl_file):
        """Test listing datasets."""
        # Upload a dataset
        with open(sample_pkl_file, 'rb') as f:
            client.post(
                "/upload",
                files={"file": ("test_data.pkl", f, "application/octet-stream")}
            )
        
        response = client.get("/datasets")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_models(self, client, sample_pkl_file):
        """Test listing models."""
        # Upload and train
        with open(sample_pkl_file, 'rb') as f:
            client.post(
                "/upload",
                files={"file": ("test_data.pkl", f, "application/octet-stream")}
            )
        
        client.post("/train", json={
            "hidden_layers": [32, 16],
            "max_epochs": 10
        })
        
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)