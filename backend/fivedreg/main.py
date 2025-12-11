import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fivedreg import FiveDRegressor, load_dataset, FiveDDataset

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="5D Regressor API",
    description="A REST API for training and serving neural network models "
                "that interpolate 5-dimensional datasets.",
    version="0.1.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# In-Memory Storage
# =============================================================================

# Storage for uploaded datasets
datasets: Dict[str, Dict[str, Any]] = {}

# Storage for trained models
models: Dict[str, Dict[str, Any]] = {}

# Current active dataset and model IDs
current_dataset_id: Optional[str] = None
current_model_id: Optional[str] = None

# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: str
    version: str = "0.1.0"


class UploadResponse(BaseModel):
    """Dataset upload response."""
    message: str
    dataset_id: str
    n_samples: int
    n_features: int


class TrainRequest(BaseModel):
    """Training configuration request."""
    dataset_id: Optional[str] = None
    hidden_layers: List[int] = Field(default=[64, 32, 16])
    learning_rate: float = Field(default=5e-3, gt=0)
    max_epochs: int = Field(default=100, gt=0, le=1000)
    batch_size: int = Field(default=256, gt=0)
    patience: int = Field(default=20, gt=0)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    val_size: float = Field(default=0.1, gt=0, lt=1)


class TrainResponse(BaseModel):
    """Training result response."""
    message: str
    model_id: str
    train_samples: int
    val_samples: int
    test_samples: int
    test_metrics: Dict[str, float]
    training_time_seconds: float


class PredictRequest(BaseModel):
    """Prediction request with 5 input values."""
    features: List[float] = Field(..., min_length=5, max_length=5)
    model_id: Optional[str] = None


class PredictBatchRequest(BaseModel):
    """Batch prediction request."""
    features: List[List[float]]
    model_id: Optional[str] = None


class PredictResponse(BaseModel):
    """Prediction response."""
    prediction: float
    model_id: str


class PredictBatchResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[float]
    model_id: str
    n_samples: int


class DatasetInfo(BaseModel):
    """Dataset information."""
    dataset_id: str
    n_samples: int
    n_features: int
    uploaded_at: str


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    hidden_layers: List[int]
    test_r2: float
    test_rmse: float
    trained_at: str


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health status.
    
    Returns the current status of the API service.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0"
    )


@app.post("/upload", response_model=UploadResponse, tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a 5D dataset in .pkl format.
    
    The file should contain a dictionary with 'X' (n_samples, 5) and 'y' (n_samples,) arrays.
    """
    global current_dataset_id
    
    # Validate file extension
    if not file.filename.endswith((".pkl", ".pickle")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a .pkl file."
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load and validate dataset
        dataset = load_dataset(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Generate dataset ID and store
        dataset_id = str(uuid.uuid4())[:8]
        datasets[dataset_id] = {
            "dataset": dataset,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "n_samples": len(dataset),
            "n_features": 5
        }
        
        # Set as current dataset
        current_dataset_id = dataset_id
        
        return UploadResponse(
            message=f"Dataset '{file.filename}' uploaded successfully",
            dataset_id=dataset_id,
            n_samples=len(dataset),
            n_features=5
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load dataset: {str(e)}"
        )


@app.post("/train", response_model=TrainResponse, tags=["Model"])
async def train_model(config: TrainRequest = TrainRequest()):
    """
    Train a model on the uploaded dataset.
    
    Uses the specified or most recently uploaded dataset.
    """
    global current_model_id
    
    import time
    
    # Get dataset
    dataset_id = config.dataset_id or current_dataset_id
    
    if dataset_id is None or dataset_id not in datasets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No dataset available. Please upload a dataset first."
        )
    
    dataset: FiveDDataset = datasets[dataset_id]["dataset"]
    
    try:
        # Handle missing values and split data
        dataset.handle_missing_values(strategy="mean")
        train_data, val_data, test_data = dataset.split(
            test_size=config.test_size,
            val_size=config.val_size
        )
        
        # Create and train model
        model = FiveDRegressor(
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            verbose=False  # Disable verbose for API
        )
        
        start_time = time.time()
        model.fit(train_data.X, train_data.y)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_metrics = model.score(test_data.X, test_data.y)
        
        # Store model
        model_id = str(uuid.uuid4())[:8]
        models[model_id] = {
            "model": model,
            "config": config.model_dump(),
            "test_metrics": test_metrics,
            "trained_at": datetime.now().isoformat(),
            "dataset_id": dataset_id
        }
        
        # Set as current model
        current_model_id = model_id
        
        return TrainResponse(
            message="Model trained successfully",
            model_id=model_id,
            train_samples=len(train_data),
            val_samples=len(val_data),
            test_samples=len(test_data),
            test_metrics=test_metrics,
            training_time_seconds=round(training_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse, tags=["Model"])
async def predict(request: PredictRequest):
    """
    Make a prediction for a single 5D input vector.
    
    Provide exactly 5 feature values.
    """
    # Get model
    model_id = request.model_id or current_model_id
    
    if model_id is None or model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No trained model available. Please train a model first."
        )
    
    model: FiveDRegressor = models[model_id]["model"]
    
    try:
        # Validate input
        if len(request.features) != 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input must have exactly 5 features."
            )
        
        # Make prediction
        X = np.array([request.features], dtype=np.float32)
        prediction = model.predict(X)[0]
        
        return PredictResponse(
            prediction=float(prediction),
            model_id=model_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=PredictBatchResponse, tags=["Model"])
async def predict_batch(request: PredictBatchRequest):
    """
    Make predictions for multiple 5D input vectors.
    """
    # Get model
    model_id = request.model_id or current_model_id
    
    if model_id is None or model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No trained model available. Please train a model first."
        )
    
    model: FiveDRegressor = models[model_id]["model"]
    
    try:
        # Validate inputs
        for i, features in enumerate(request.features):
            if len(features) != 5:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Input at index {i} must have exactly 5 features."
                )
        
        # Make predictions
        X = np.array(request.features, dtype=np.float32)
        predictions = model.predict(X).tolist()
        
        return PredictBatchResponse(
            predictions=predictions,
            model_id=model_id,
            n_samples=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/datasets", response_model=List[DatasetInfo], tags=["Dataset"])
async def list_datasets():
    """List all uploaded datasets."""
    return [
        DatasetInfo(
            dataset_id=did,
            n_samples=info["n_samples"],
            n_features=info["n_features"],
            uploaded_at=info["uploaded_at"]
        )
        for did, info in datasets.items()
    ]


@app.get("/models", response_model=List[ModelInfo], tags=["Model"])
async def list_models():
    """List all trained models."""
    return [
        ModelInfo(
            model_id=mid,
            hidden_layers=info["config"]["hidden_layers"],
            test_r2=info["test_metrics"]["r2"],
            test_rmse=info["test_metrics"]["rmse"],
            trained_at=info["trained_at"]
        )
        for mid, info in models.items()
    ]


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)