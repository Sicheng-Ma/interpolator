=============
API Reference
=============

This section provides detailed documentation for all modules and classes.

Data Module
-----------

.. module:: fivedreg.data
   :synopsis: Data loading and preprocessing

FiveDDataset
^^^^^^^^^^^^

.. class:: FiveDDataset(X, y)

   A class to handle 5D datasets for regression tasks.
   
   :param X: Feature array of shape (n_samples, 5)
   :type X: numpy.ndarray
   :param y: Target array of shape (n_samples,)
   :type y: numpy.ndarray
   :raises DataValidationError: If data validation fails

   .. method:: handle_missing_values(strategy="mean", fill_value=None)
   
      Handle missing values (NaN) in the dataset.
      
      :param strategy: Strategy for handling missing values. Options: "mean", "median", "drop", "fill"
      :type strategy: str
      :param fill_value: Value to use when strategy is "fill"
      :type fill_value: float, optional
      :returns: Self for method chaining
      :rtype: FiveDDataset

   .. method:: standardize(scaler=None)
   
      Standardize features to zero mean and unit variance.
      
      :param scaler: Optional pre-fitted scaler
      :type scaler: sklearn.preprocessing.StandardScaler, optional
      :returns: Self for method chaining
      :rtype: FiveDDataset

   .. method:: split(test_size=0.2, val_size=0.1, random_state=42)
   
      Split the dataset into train, validation, and test sets.
      
      :param test_size: Proportion of data for test set
      :type test_size: float
      :param val_size: Proportion of data for validation set
      :type val_size: float
      :param random_state: Random seed for reproducibility
      :type random_state: int, optional
      :returns: Tuple of (train, val, test) datasets
      :rtype: tuple[FiveDDataset, FiveDDataset, FiveDDataset]

Functions
^^^^^^^^^

.. function:: load_dataset(filepath)

   Load a 5D dataset from a pickle file.
   
   :param filepath: Path to the .pkl file
   :type filepath: str or Path
   :returns: Dataset object
   :rtype: FiveDDataset
   :raises FileNotFoundError: If file does not exist
   :raises DataValidationError: If data format is invalid

.. function:: create_sample_dataset(n_samples=1000, noise_level=0.1, random_state=42)

   Create a sample 5D dataset for testing.
   
   :param n_samples: Number of samples to generate
   :type n_samples: int
   :param noise_level: Standard deviation of noise
   :type noise_level: float
   :param random_state: Random seed
   :type random_state: int, optional
   :returns: Generated dataset
   :rtype: FiveDDataset

Model Module
------------

.. module:: fivedreg.model
   :synopsis: Neural network model implementation

MLP
^^^

.. class:: MLP(in_dim=5, hidden=64, depth=3, out_dim=1, activation=nn.SiLU)

   Multi-Layer Perceptron for 5D regression.
   
   :param in_dim: Number of input features
   :type in_dim: int
   :param hidden: Number of neurons per hidden layer
   :type hidden: int
   :param depth: Number of hidden layers
   :type depth: int
   :param out_dim: Number of output dimensions
   :type out_dim: int
   :param activation: Activation function class
   :type activation: torch.nn.Module

   .. method:: forward(x)
   
      Forward pass through the network.
      
      :param x: Input tensor of shape (batch_size, 5)
      :type x: torch.Tensor
      :returns: Output tensor of shape (batch_size, 1)
      :rtype: torch.Tensor

   .. method:: count_parameters()
   
      Count total trainable parameters.
      
      :returns: Number of parameters
      :rtype: int

FiveDRegressor
^^^^^^^^^^^^^^

.. class:: FiveDRegressor(hidden_layers=None, learning_rate=0.005, max_epochs=200, batch_size=256, patience=20, weight_decay=1e-6, val_fraction=0.15, random_state=42, verbose=True)

   Scikit-learn style wrapper for the MLP model.
   
   :param hidden_layers: List of neurons per layer, e.g., [64, 32, 16]
   :type hidden_layers: list[int], optional
   :param learning_rate: Learning rate for optimizer
   :type learning_rate: float
   :param max_epochs: Maximum training epochs
   :type max_epochs: int
   :param batch_size: Training batch size
   :type batch_size: int
   :param patience: Early stopping patience
   :type patience: int
   :param weight_decay: L2 regularization strength
   :type weight_decay: float
   :param val_fraction: Fraction of data for validation
   :type val_fraction: float
   :param random_state: Random seed
   :type random_state: int
   :param verbose: Print training progress
   :type verbose: bool

   .. method:: fit(X, y)
   
      Train the model.
      
      :param X: Feature array of shape (n_samples, 5)
      :type X: numpy.ndarray
      :param y: Target array of shape (n_samples,)
      :type y: numpy.ndarray
      :returns: Self for method chaining
      :rtype: FiveDRegressor

   .. method:: predict(X)
   
      Make predictions.
      
      :param X: Feature array of shape (n_samples, 5)
      :type X: numpy.ndarray
      :returns: Predictions of shape (n_samples,)
      :rtype: numpy.ndarray

   .. method:: score(X, y)
   
      Compute evaluation metrics.
      
      :param X: Feature array
      :type X: numpy.ndarray
      :param y: True target values
      :type y: numpy.ndarray
      :returns: Dictionary with 'mse', 'rmse', 'mae', 'r2'
      :rtype: dict[str, float]

   .. method:: save(filepath)
   
      Save the trained model.
      
      :param filepath: Path to save the model
      :type filepath: str

   .. method:: load(filepath)
   
      Load a trained model.
      
      :param filepath: Path to the saved model
      :type filepath: str
      :returns: Self for method chaining
      :rtype: FiveDRegressor

REST API
--------

.. module:: fivedreg.main
   :synopsis: FastAPI backend

Endpoints
^^^^^^^^^

GET /health
"""""""""""

Health check endpoint.

**Response:**

.. code-block:: json

   {
     "status": "healthy",
     "timestamp": "2025-12-15T10:00:00",
     "version": "0.1.0"
   }

POST /upload
""""""""""""

Upload a dataset.

**Request:** Multipart form with ``file`` field containing .pkl file

**Response:**

.. code-block:: json

   {
     "message": "Dataset uploaded successfully",
     "dataset_id": "abc12345",
     "n_samples": 5000,
     "n_features": 5
   }

POST /train
"""""""""""

Train a model.

**Request Body:**

.. code-block:: json

   {
     "hidden_layers": [64, 32, 16],
     "learning_rate": 0.005,
     "max_epochs": 100,
     "batch_size": 256,
     "patience": 20
   }

**Response:**

.. code-block:: json

   {
     "message": "Model trained successfully",
     "model_id": "xyz78901",
     "train_samples": 3500,
     "val_samples": 500,
     "test_samples": 1000,
     "test_metrics": {
       "mse": 0.0112,
       "rmse": 0.1057,
       "mae": 0.0846,
       "r2": 0.9988
     },
     "training_time_seconds": 7.6
   }

POST /predict
"""""""""""""

Make a prediction.

**Request Body:**

.. code-block:: json

   {
     "features": [0.1, 0.2, 0.3, 0.4, 0.5],
     "model_id": "xyz78901"
   }

**Response:**

.. code-block:: json

   {
     "prediction": 1.2345,
     "model_id": "xyz78901"
   }

Exceptions
----------

.. exception:: DataValidationError

   Raised when data validation fails (wrong dimensions, invalid format, etc.)