# IBM AI Enterprise Workflow Capstone

Usage notes
===============

All commands are from this directory.


To test app.py
---------------------

.. code-block:: bash

    ~$ python app.py

or to start the flask app in debug mode

.. code-block:: bash

    ~$ python app.py -d

Go to http://0.0.0.0:8080/ and you will see a basic website that can be customtized for a project.

To test the model directly
----------------------------

see the code at the bottom of `model.py`

.. code-block:: bash

    ~$ python model.py

To build the docker container
--------------------------------

.. code-block:: bash

    ~$ docker build -t revenue-ml .

Check that the image is there.

.. code-block:: bash

    ~$ docker image ls

You may notice images that you no longer use. You may delete them with

.. code-block:: bash

    ~$ docker image rm IMAGE_ID_OR_NAME

And every once and a while if you want clean up you can

.. code-block:: bash

    ~$ docker system prune


Run the unittests
-------------------

Before running the unit tests launch the `app.py`.

To run only the api tests

.. code-block:: bash

    ~$ python unittests/ApiTests.py

To run only the model tests

.. code-block:: bash

    ~$ python unittests/ModelTests.py

To run only the model tests

.. code-block:: bash

    ~$ python unittests/LoggerTests.py


To run all of the tests

.. code-block:: bash

    ~$ python run-tests.py

Run the container to test that it is working
----------------------------------------------

.. code-block:: bash

    ~$ docker run -p 4000:8080 revenue-ml

Go to http://0.0.0.0:4000/ and you will see a basic website that can be customised for a project.


Performance and model comparison is in model_performance.py



# Evaluation Criteria Input
## Are there unit tests for the API?
unittests/ApiTests.py

## Are there unit tests for the model?
unittests/ModelTests.py

## Are there unit tests for the logging?
unittests/LoggerTests.py

## Can all of the unit tests be run with a single script and do all of the unit tests pass?

/run-tests.py

# Is there a mechanism to monitor performance?
logger.py

# Was there an attempt to isolate the read/write unit tests From production models and logs?
using sl as prefix for production models and logs.
For test no prefix is used


# Does the API work as expected?
 For example, can you get predictions for a specific country as well as for all countries combined?
 
 app.py
 
# Does the data ingestion exists as a function or script to facilitate automation?

cslib.py
# Where multiple models compared?
model_performance.py
# Did the EDA investigation use visualizations?
notebooks/Part1 EDA.ipynb

# Is everything containerized within a working Docker image?
Dockerfile

# Did they use a visualization to compare their model to the baseline model?

