=======
Testing
=======

This section describes the test suite and how to run tests.

Test Suite Overview
-------------------

The project includes comprehensive tests covering all components:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Test File
     - Coverage
     - Description
   * - test_data.py
     - Data module
     - Dataset loading, validation, preprocessing, splitting
   * - test_model.py
     - Model module
     - MLP architecture, training, prediction, save/load
   * - test_api.py
     - REST API
     - All endpoints, error handling

Running Tests
-------------

Basic Test Run
^^^^^^^^^^^^^^

.. code-block:: bash

   cd backend
   pytest

Verbose Output
^^^^^^^^^^^^^^

.. code-block:: bash

   pytest -v

With Coverage Report
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest --cov=fivedreg --cov-report=term-missing

Run Specific Test File
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest tests/test_data.py
   pytest tests/test_model.py
   pytest tests/test_api.py

Run Specific Test
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest tests/test_data.py::TestFiveDDataset::test_standardize

Test Categories
---------------

Data Module Tests
^^^^^^^^^^^^^^^^^

**TestFiveDDataset**

- ``test_create_valid_dataset`` - Create dataset with valid data
- ``test_invalid_x_dimensions`` - Reject wrong feature dimensions
- ``test_invalid_x_ndim`` - Reject 1D input arrays
- ``test_mismatched_samples`` - Reject mismatched X and y sizes
- ``test_empty_dataset`` - Reject empty datasets
- ``test_handle_missing_values_mean`` - Mean imputation
- ``test_handle_missing_values_drop`` - Drop rows with NaN
- ``test_standardize`` - Feature standardization
- ``test_split_default`` - Default train/val/test split
- ``test_split_custom_sizes`` - Custom split proportions
- ``test_split_invalid_sizes`` - Reject invalid split sizes

**TestLoadDataset**

- ``test_load_dict_format`` - Load from dict format
- ``test_load_tuple_format`` - Load from tuple format
- ``test_load_nonexistent_file`` - Handle missing files
- ``test_load_invalid_format`` - Reject invalid file formats

**TestCreateSampleDataset**

- ``test_create_default`` - Create with default parameters
- ``test_create_custom_size`` - Create with custom size
- ``test_reproducibility`` - Verify reproducibility with random_state

Model Module Tests
^^^^^^^^^^^^^^^^^^

**TestMLP**

- ``test_create_default`` - Create with default parameters
- ``test_create_custom`` - Create with custom architecture
- ``test_forward_pass`` - Verify forward pass output shape
- ``test_count_parameters`` - Count model parameters

**TestFiveDRegressor**

- ``test_create_default`` - Create with default parameters
- ``test_create_custom`` - Create with custom configuration
- ``test_fit`` - Training functionality
- ``test_predict_before_fit`` - Error handling for untrained model
- ``test_predict`` - Prediction on trained model
- ``test_predict_single`` - Single sample prediction
- ``test_score`` - Evaluation metrics
- ``test_get_params`` - Get hyperparameters
- ``test_save_and_load`` - Model persistence
- ``test_reproducibility`` - Training reproducibility
- ``test_early_stopping`` - Early stopping behavior

**TestModelPerformance**

- ``test_training_time_small`` - Training completes in reasonable time
- ``test_model_accuracy`` - Model achieves acceptable accuracy

API Tests
^^^^^^^^^

**TestHealthEndpoint**

- ``test_health_check`` - Health endpoint returns correct response

**TestUploadEndpoint**

- ``test_upload_valid_file`` - Upload valid .pkl file
- ``test_upload_invalid_extension`` - Reject non-.pkl files
- ``test_upload_invalid_content`` - Reject invalid pickle content

**TestTrainEndpoint**

- ``test_train_without_dataset`` - Error when no dataset uploaded
- ``test_train_with_dataset`` - Training with uploaded dataset
- ``test_train_with_custom_config`` - Training with custom config

**TestPredictEndpoint**

- ``test_predict_without_model`` - Error when no model trained
- ``test_predict_with_model`` - Prediction with trained model
- ``test_predict_invalid_features`` - Reject wrong number of features

**TestListEndpoints**

- ``test_list_datasets`` - List uploaded datasets
- ``test_list_models`` - List trained models

Test Fixtures
-------------

The ``conftest.py`` file provides reusable fixtures:

.. code-block:: python

   @pytest.fixture
   def sample_dataset():
       """Create a sample dataset (500 samples)."""
       return create_sample_dataset(n_samples=500, random_state=42)
   
   @pytest.fixture
   def small_dataset():
       """Create a small dataset (100 samples) for quick tests."""
       return create_sample_dataset(n_samples=100, random_state=42)
   
   @pytest.fixture
   def trained_model(small_dataset):
       """Create a trained model."""
       model = FiveDRegressor(
           hidden_layers=[32, 16],
           max_epochs=10,
           verbose=False
       )
       model.fit(small_dataset.X, small_dataset.y)
       return model
   
   @pytest.fixture
   def temp_pkl_file(sample_dataset):
       """Create a temporary .pkl file."""
       # Creates and cleans up temp file

Coverage Report
---------------

Current test coverage:

.. code-block:: text

   Name                   Stmts   Miss  Cover
   ------------------------------------------
   fivedreg/__init__.py       5      0   100%
   fivedreg/data.py         120     33    72%
   fivedreg/main.py         150     26    83%
   fivedreg/model.py        164     12    93%
   ------------------------------------------
   TOTAL                    439     71    84%

The overall coverage is **84%**, with critical paths well-tested.

Adding New Tests
----------------

To add new tests:

1. Create a test function in the appropriate test file
2. Use fixtures for common setup
3. Follow the naming convention ``test_<description>``

Example:

.. code-block:: python

   def test_my_new_feature(sample_dataset):
       """Test description."""
       # Arrange
       model = FiveDRegressor()
       
       # Act
       model.fit(sample_dataset.X, sample_dataset.y)
       result = model.predict(sample_dataset.X[:1])
       
       # Assert
       assert result.shape == (1,)