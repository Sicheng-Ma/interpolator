Deployment
==========

This section covers deployment options for the 5D Regressor system.

Local Development
-----------------

Running Backend
^^^^^^^^^^^^^^^

.. code-block:: bash

   cd backend
   uvicorn fivedreg.main:app --reload --host 127.0.0.1 --port 8000

The ``--reload`` flag enables auto-reload on code changes.

Running Frontend
^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd frontend
   npm run dev

Access at http://localhost:3000

Docker Deployment
-----------------

Docker Compose (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to deploy both services:

.. code-block:: bash

   # Build and start
   docker-compose up --build
   
   # Run in background
   docker-compose up -d --build
   
   # View logs
   docker-compose logs -f
   
   # Stop services
   docker-compose down

Services will be available at:

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

Individual Containers
^^^^^^^^^^^^^^^^^^^^^

Build and run containers separately:

**Backend:**

.. code-block:: bash

   cd backend
   docker build -t interpolator-backend .
   docker run -p 8000:8000 interpolator-backend

**Frontend:**

.. code-block:: bash

   cd frontend
   docker build -t interpolator-frontend .
   docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://localhost:8000 interpolator-frontend

Docker Configuration
^^^^^^^^^^^^^^^^^^^^

**docker-compose.yml:**

.. code-block:: yaml

   services:
     backend:
       build:
         context: ./backend
         dockerfile: Dockerfile
       ports:
         - "8000:8000"
       healthcheck:
         test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
         interval: 30s
         timeout: 10s
         retries: 3

     frontend:
       build:
         context: ./frontend
         dockerfile: Dockerfile
       ports:
         - "3000:3000"
       environment:
         - NEXT_PUBLIC_API_URL=http://localhost:8000
       depends_on:
         backend:
           condition: service_healthy

Environment Variables
---------------------

Backend
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Variable
     - Default
     - Description
   * - HOST
     - 0.0.0.0
     - Server host address
   * - PORT
     - 8000
     - Server port

Frontend
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Variable
     - Default
     - Description
   * - NEXT_PUBLIC_API_URL
     - http://localhost:8000
     - Backend API URL
   * - PORT
     - 3000
     - Frontend port

Shell Scripts
-------------

Build Documentation
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./scripts/build_docs.sh

This script:

1. Installs Sphinx dependencies
2. Builds HTML documentation
3. Opens documentation in browser

Launch Stack
^^^^^^^^^^^^

.. code-block:: bash

   ./scripts/launch.sh

This script:

1. Checks for required dependencies
2. Starts the backend server
3. Starts the frontend server
4. Opens the application in browser

Health Checks
-------------

Backend Health
^^^^^^^^^^^^^^

.. code-block:: bash

   curl http://localhost:8000/health

Expected response:

.. code-block:: json

   {
     "status": "healthy",
     "timestamp": "2025-12-15T10:00:00",
     "version": "0.1.0"
   }

Docker Health
^^^^^^^^^^^^^

.. code-block:: bash

   docker-compose ps

All services should show "healthy" status.

Troubleshooting
---------------

Port Already in Use
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Find process using port
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>

Docker Build Fails
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Clear Docker cache
   docker system prune -f
   
   # Rebuild
   docker-compose up --build

Frontend Can't Connect to Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Ensure backend is running and healthy
2. Check NEXT_PUBLIC_API_URL environment variable
3. Verify CORS settings in backend

Module Import Errors
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Reinstall package
   cd backend
   pip install -e .