5D Neural Network Regressor Documentation
==========================================

This is the documentation for the **5D Neural Network Regressor**, a full-stack 
machine learning system for interpolating 5-dimensional datasets.

This project was developed as part of the MPhil in Data Intensive Science coursework 
in Research Computing C1 at the University of Cambridge.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   performance
   testing
   deployment

Overview
--------

The system consists of:

- **Python Package (fivedreg)**: Data loading, model training, and inference
- **FastAPI Backend**: REST API for training and prediction
- **Next.js Frontend**: Interactive web interface
- **Docker Support**: Containerized deployment

Features
--------

- Load and preprocess 5D datasets from ``.pkl`` files
- Handle missing values with multiple strategies
- Train configurable neural networks with early stopping
- REST API for model serving
- Interactive web UI for uploading, training, and predicting
- Comprehensive test suite
- Docker deployment support

Quick Links
-----------

- :doc:`installation` - Get started with installation
- :doc:`quickstart` - Quick start guide
- :doc:`api_reference` - Full API documentation
- :doc:`performance` - Performance benchmarks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`