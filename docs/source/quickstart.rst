Quick Start
===========

This guide will help you get up and running with the 5D Regressor in just a few minutes.

Using the Python Package
------------------------

The simplest way to use the system is through the Python package:

.. code-block:: python

   from fivedreg import load_dataset, FiveDRegressor

   # Load your dataset
   dataset = load_dataset("your_data.pkl")
   
   # Handle missing values
   dataset.handle_missing_values(strategy="mean")
   
   # Split into train/val/test
   train, val, test = dataset.split(test_size=0.2, val_size=0.1)
   
   # Create and train model
   model = FiveDRegressor(
       hidden_layers=[64, 32, 16],
       max_epochs=100
   )
   model.fit(train.X, train.y)
   
   # Evaluate
   scores = model.score(test.X, test.y)
   print(f"R² Score: {scores['r2']:.4f}")
   
   # Make predictions
   predictions = model.predict(test.X)

Using the Web Interface
-----------------------

1. Start the backend server:

.. code-block:: bash

   cd backend
   uvicorn fivedreg.main:app --reload --port 8000

2. Start the frontend (in a new terminal):

.. code-block:: bash

   cd frontend
   npm run dev

3. Open http://localhost:3000 in your browser

4. Follow the workflow:
   
   - **Upload**: Upload your ``.pkl`` dataset
   - **Train**: Configure hyperparameters and train
   - **Predict**: Enter 5 feature values to get predictions

Using the REST API
------------------

You can also interact with the system via REST API:

.. code-block:: bash

   # Health check
   curl http://localhost:8000/health
   
   # Upload dataset
   curl -X POST -F "file=@data.pkl" http://localhost:8000/upload
   
   # Train model
   curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d '{"hidden_layers": [64, 32, 16], "max_epochs": 100}'
   
   # Make prediction
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5]}'

Using Docker
------------

For a one-command deployment:

.. code-block:: bash

   docker-compose up --build

Then access:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

Dataset Format
--------------

Your ``.pkl`` file should contain a dictionary with:

- ``X``: numpy array of shape ``(n_samples, 5)`` - features
- ``y``: numpy array of shape ``(n_samples,)`` - targets

Example of creating a compatible dataset:

.. code-block:: python

   import pickle
   import numpy as np
   
   # Your data
   X = np.random.randn(1000, 5)  # 1000 samples, 5 features
   y = np.random.randn(1000)      # 1000 targets
   
   # Save as pickle
   with open("my_data.pkl", "wb") as f:
       pickle.dump({"X": X, "y": y}, f)