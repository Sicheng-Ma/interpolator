User Guide
==========

This guide provides detailed information on using the 5D Neural Network Regressor.

Data Handling
-------------

Loading Datasets
^^^^^^^^^^^^^^^^

The system supports loading datasets from ``.pkl`` files:

.. code-block:: python

   from fivedreg import load_dataset
   
   dataset = load_dataset("path/to/data.pkl")
   print(f"Loaded {len(dataset)} samples")
   print(f"Features shape: {dataset.X.shape}")
   print(f"Target shape: {dataset.y.shape}")

The pickle file should contain either:

- A dictionary with ``'X'`` and ``'y'`` keys
- A tuple of ``(X, y)``

Handling Missing Values
^^^^^^^^^^^^^^^^^^^^^^^

The system provides several strategies for handling missing values:

.. code-block:: python

   # Replace with column mean
   dataset.handle_missing_values(strategy="mean")
   
   # Replace with column median
   dataset.handle_missing_values(strategy="median")
   
   # Drop rows with missing values
   dataset.handle_missing_values(strategy="drop")
   
   # Fill with a constant value
   dataset.handle_missing_values(strategy="fill", fill_value=0.0)

Feature Standardization
^^^^^^^^^^^^^^^^^^^^^^^

Standardize features to zero mean and unit variance:

.. code-block:: python

   dataset.standardize()
   
   # Check standardization
   print(f"Mean: {dataset.X.mean(axis=0)}")  # Should be ~0
   print(f"Std: {dataset.X.std(axis=0)}")    # Should be ~1

Data Splitting
^^^^^^^^^^^^^^

Split data into training, validation, and test sets:

.. code-block:: python

   train, val, test = dataset.split(
       test_size=0.2,   # 20% for testing
       val_size=0.1,    # 10% for validation
       random_state=42  # For reproducibility
   )
   
   print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

Model Training
--------------

Basic Training
^^^^^^^^^^^^^^

.. code-block:: python

   from fivedreg import FiveDRegressor
   
   model = FiveDRegressor()
   model.fit(train.X, train.y)

Custom Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = FiveDRegressor(
       hidden_layers=[64, 32, 16],  # Network architecture
       learning_rate=0.005,          # Learning rate
       max_epochs=200,               # Maximum epochs
       batch_size=256,               # Batch size
       patience=20,                  # Early stopping patience
       weight_decay=1e-6,            # L2 regularization
       verbose=True                  # Print progress
   )
   model.fit(train.X, train.y)

Hyperparameter Guide
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - hidden_layers
     - [64, 32, 16]
     - List of neurons per hidden layer
   * - learning_rate
     - 0.005
     - Adam optimizer learning rate
   * - max_epochs
     - 200
     - Maximum training epochs
   * - batch_size
     - 256
     - Mini-batch size
   * - patience
     - 20
     - Early stopping patience
   * - weight_decay
     - 1e-6
     - L2 regularization strength

Model Evaluation
----------------

Scoring
^^^^^^^

.. code-block:: python

   scores = model.score(test.X, test.y)
   
   print(f"MSE:  {scores['mse']:.4f}")
   print(f"RMSE: {scores['rmse']:.4f}")
   print(f"MAE:  {scores['mae']:.4f}")
   print(f"R²:   {scores['r2']:.4f}")

Making Predictions
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Single prediction
   x_new = [[0.1, 0.2, 0.3, 0.4, 0.5]]
   prediction = model.predict(x_new)
   print(f"Prediction: {prediction[0]:.4f}")
   
   # Batch prediction
   predictions = model.predict(test.X)

Saving and Loading Models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Save model
   model.save("my_model.pt")
   
   # Load model
   loaded_model = FiveDRegressor()
   loaded_model.load("my_model.pt")
   
   # Use loaded model
   predictions = loaded_model.predict(test.X)

REST API Usage
--------------

The FastAPI backend provides REST endpoints for all operations.

Starting the Server
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd backend
   uvicorn fivedreg.main:app --host 0.0.0.0 --port 8000

API Endpoints
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Method
     - Endpoint
     - Description
   * - GET
     - /health
     - Health check
   * - POST
     - /upload
     - Upload .pkl dataset
   * - POST
     - /train
     - Train model with configuration
   * - POST
     - /predict
     - Make single prediction
   * - POST
     - /predict/batch
     - Make batch predictions
   * - GET
     - /datasets
     - List uploaded datasets
   * - GET
     - /models
     - List trained models

Example API Workflow
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import requests
   
   BASE_URL = "http://localhost:8000"
   
   # Upload dataset
   with open("data.pkl", "rb") as f:
       response = requests.post(
           f"{BASE_URL}/upload",
           files={"file": ("data.pkl", f)}
       )
   dataset_id = response.json()["dataset_id"]
   
   # Train model
   response = requests.post(
       f"{BASE_URL}/train",
       json={
           "hidden_layers": [64, 32, 16],
           "max_epochs": 100
       }
   )
   model_id = response.json()["model_id"]
   
   # Make prediction
   response = requests.post(
       f"{BASE_URL}/predict",
       json={"features": [0.1, 0.2, 0.3, 0.4, 0.5]}
   )
   prediction = response.json()["prediction"]

Web Interface
-------------

The Next.js frontend provides an intuitive interface:

Upload Page (/upload)
^^^^^^^^^^^^^^^^^^^^^

- Drag and drop ``.pkl`` files
- File validation and preview
- Upload progress indicator

Train Page (/train)
^^^^^^^^^^^^^^^^^^^

- Configure all hyperparameters
- Start training with one click
- View training metrics (R², RMSE, training time)

Predict Page (/predict)
^^^^^^^^^^^^^^^^^^^^^^^

- Enter 5 feature values
- Random value generator for testing
- Instant prediction display