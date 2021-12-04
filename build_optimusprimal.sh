# <This script needs to be ran from within optimus-primal root directory>

# Install core and extra requirements
pip install -r requirements/requirements-core.txt
pip install -r requirements/requirements-examples.txt
pip install -r requirements/requirements-tests.txt
pip install -r requirements/requirements-docs.txt

# Install specific converter for building tutorial documentation
conda install pandoc=1.19.2.1 -y

# Build Scatter
pip install -e .