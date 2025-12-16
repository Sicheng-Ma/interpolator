# 5D Neural Network Interpolator

A full-stack machine learning system for training and serving neural network models that interpolate 5-dimensional datasets.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Documentation](#documentation)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)

## Overview

This project provides a research-grade system capable of:
- Loading and preprocessing 5D numerical datasets
- Training configurable neural network models
- Serving predictions via REST API
- Interactive web interface for model training and inference

## Features

- **Data Handling**: Load `.pkl` datasets, handle missing values, standardize features, split data
- **Neural Network**: Configurable MLP with customizable layers, learning rate, early stopping
- **REST API**: FastAPI backend with endpoints for upload, train, and predict
- **Web Interface**: Modern Next.js frontend with responsive design
- **Containerization**: Docker support for easy deployment
- **Testing**: Comprehensive test suite with pytest

## Project Structure

```
interpolator/
├── backend/
│   ├── fivedreg/
│   │   ├── __init__.py      # Package initialization
│   │   ├── data.py          # Data loading and preprocessing
│   │   ├── model.py         # Neural network implementation
│   │   └── main.py          # FastAPI application
│   ├── tests/
│   │   ├── conftest.py      # Pytest fixtures
│   │   ├── test_data.py     # Data module tests
│   │   ├── test_model.py    # Model tests
│   │   └── test_api.py      # API tests
│   ├── pyproject.toml       # Python package configuration
│   └── Dockerfile           # Backend container
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js pages
│   │   ├── components/      # React components
│   │   └── lib/             # API utilities
│   ├── package.json         # Node dependencies
│   └── Dockerfile           # Frontend container
├── docker-compose.yml       # Container orchestration
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker (optional, for containerized deployment)

### Backend Setup

```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install dependencies directly
pip install numpy torch scikit-learn fastapi uvicorn python-multipart pydantic
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Or using yarn
yarn install
```

## Usage

### Quick Start with Launch Script

The easiest way to run the application is using the provided launch script:

```bash
# Make the script executable (first time only)
chmod +x scripts/launch.sh

# Launch both backend and frontend
./scripts/launch.sh

# Or launch specific components
./scripts/launch.sh backend   # Backend only
./scripts/launch.sh frontend  # Frontend only
./scripts/launch.sh docker    # Using Docker Compose
```

The script will automatically:
- Check for required dependencies (Python, Node.js)
- Install packages if needed
- Check and free ports if occupied
- Start services in the background
- Open your browser to http://localhost:3000

Press `Ctrl+C` to stop all services.

### Running Locally (Manual)

#### 1. Start the Backend

```bash
cd backend
uvicorn fivedreg.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at http://localhost:8000

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

#### 2. Start the Frontend

```bash
cd frontend
npm run dev
```

The web interface will be available at http://localhost:3000

### Using the Web Interface

1. **Upload**: Navigate to `/upload` and drag-drop your `.pkl` dataset
2. **Train**: Go to `/train`, configure hyperparameters, and click "Start Training"
3. **Predict**: Visit `/predict`, enter 5 feature values, and get predictions

### Using the API Directly

```bash
# Health check
curl http://localhost:8000/health

# Upload dataset
curl -X POST -F "file=@dataset.pkl" http://localhost:8000/upload

# Train model
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"hidden_layers": [64, 32, 16], "max_epochs": 100}'

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5]}'
```

### Using the Python Package

```python
from fivedreg import load_dataset, FiveDRegressor

# Load data
dataset = load_dataset("data.pkl")
dataset.handle_missing_values(strategy="mean")
dataset.standardize()

# Split data
train, val, test = dataset.split(test_size=0.2, val_size=0.1)

# Train model
model = FiveDRegressor(
    hidden_layers=[64, 32, 16],
    learning_rate=0.005,
    max_epochs=100
)
model.fit(train.X, train.y)

# Evaluate
scores = model.score(test.X, test.y)
print(f"R² Score: {scores['r2']:.4f}")
print(f"RMSE: {scores['rmse']:.4f}")

# Predict
predictions = model.predict(test.X)
```

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload .pkl dataset |
| POST | `/train` | Train a model |
| POST | `/predict` | Make single prediction |
| POST | `/predict/batch` | Make batch predictions |
| GET | `/datasets` | List uploaded datasets |
| GET | `/models` | List trained models |

### Train Configuration

```json
{
  "hidden_layers": [64, 32, 16],
  "learning_rate": 0.005,
  "max_epochs": 100,
  "batch_size": 256,
  "patience": 20,
  "test_size": 0.2,
  "val_size": 0.1
}
```

### Predict Request

```json
{
  "features": [0.1, 0.2, 0.3, 0.4, 0.5],
  "model_id": "optional-model-id"
}
```

## Documentation

Full API documentation is available in both interactive and static formats.

### Building Documentation

The project uses Sphinx to generate comprehensive documentation from code docstrings.

```bash
# Build HTML documentation
./scripts/build_docs.sh
```

This script will:
- Install Sphinx and documentation tools (sphinx, sphinx-rtd-theme, myst-parser)
- Mock heavy dependencies (torch, numpy, sklearn) for faster builds
- Generate HTML documentation in `docs/build/html/`
- Automatically open the documentation in your browser

**Note**: The build process uses mock imports for heavy dependencies

### Viewing Documentation

After building, the documentation is available at:
- Local: `docs/build/html/index.html`
- Online: [ReadTheDocs](https://your-project.readthedocs.io) (if deployed)

The documentation includes:
- API reference with detailed docstrings
- Module descriptions
- Class and function signatures
- Usage examples

## Testing

### Running Tests

```bash
cd backend

# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=fivedreg --cov-report=term-missing

# Run specific test file
pytest tests/test_data.py

# Run with verbose output
pytest -v
```

### Test Coverage

The test suite covers:
- **Data Module**: Dataset loading, validation, preprocessing, splitting
- **Model Module**: MLP architecture, training, prediction, save/load
- **API Module**: All REST endpoints, error handling

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

Services will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Building Individual Containers

```bash
# Build backend
cd backend
docker build -t interpolator-backend .

# Run backend
docker run -p 8000:8000 interpolator-backend

# Build frontend
cd frontend
docker build -t interpolator-frontend .

# Run frontend
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://localhost:8000 interpolator-frontend
```

## Environment Variables

### Backend

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |
| `PORT` | Frontend port | `3000` |

## Model Architecture

The neural network is a Multi-Layer Perceptron (MLP) with:
- Configurable hidden layers (default: [64, 32, 16])
- SiLU activation functions
- AdamW optimizer with weight decay
- Cosine annealing learning rate scheduler
- Early stopping with patience

### Performance

On the provided 5000-sample dataset:
- Training time: ~5 seconds (CPU)
- Test R² Score: ~0.99
- Test RMSE: ~0.05

## License

This project is part of the MPhil in Data Intensive Science coursework at the University of Cambridge.

## Author

Sicheng Ma (sm3035@cam.ac.uk)