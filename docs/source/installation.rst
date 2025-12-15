Installation
============

This guide covers the installation of the 5D Neural Network Regressor system.

Prerequisites
-------------

- Python 3.9 or higher
- Node.js 18 or higher (for frontend)
- Docker (optional, for containerized deployment)

Backend Installation
--------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/c1_coursework/sm3035.git
   cd interpolator

2. Create a virtual environment (recommended):

.. code-block:: bash

   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the package:

.. code-block:: bash

   pip install -e .

4. Verify installation:

.. code-block:: bash

   python -c "import fivedreg; print(fivedreg.__version__)"
   # Should output: 0.1.0

Frontend Installation
---------------------

1. Navigate to the frontend directory:

.. code-block:: bash

   cd frontend

2. Install dependencies:

.. code-block:: bash

   npm install

3. Verify installation:

.. code-block:: bash

   npm run build

Docker Installation
-------------------

If you prefer containerized deployment:

1. Ensure Docker and Docker Compose are installed

2. Build and run:

.. code-block:: bash

   docker-compose up --build

This will start both the backend (port 8000) and frontend (port 3000).

Dependencies
------------

Backend Dependencies
^^^^^^^^^^^^^^^^^^^^

The following Python packages are required:

- ``numpy>=1.21.0`` - Numerical computing
- ``torch>=2.0.0`` - Neural network framework
- ``scikit-learn>=1.0.0`` - Data preprocessing
- ``fastapi>=0.100.0`` - REST API framework
- ``uvicorn>=0.22.0`` - ASGI server
- ``python-multipart>=0.0.6`` - File upload support
- ``pydantic>=2.0.0`` - Data validation

Frontend Dependencies
^^^^^^^^^^^^^^^^^^^^^

- ``next@14.0.4`` - React framework
- ``react@18`` - UI library
- ``tailwindcss@3`` - CSS framework

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**ImportError: No module named 'fivedreg'**

Make sure you installed the package in development mode:

.. code-block:: bash

   cd backend
   pip install -e .

**Port already in use**

If port 8000 or 3000 is already in use:

.. code-block:: bash

   # Find the process using the port
   lsof -i :8000
   
   # Kill it
   kill -9 <PID>

**Docker build fails**

Try clearing the Docker cache:

.. code-block:: bash

   docker system prune -f
   docker-compose up --build