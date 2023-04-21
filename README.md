# aucomet

Repository for various comet and space science related work

## Environment Setup for dev version of sbpy

### Create conda environment
    conda create -n sbpydev
    conda activate sbpydev

### install astropy as per the docs, which at the time of writing was
    conda install astropy
    conda install -c astropy -c defaults scipy matplotlib h5py beautifulsoup4 html5lib bleach pandas sortedcontainers pytz setuptools mpmath bottleneck jplephem asdf pyarrow

To run tests we also need
    pip3 install pytest-astropy sphinx-astropy

### install sbpy by cloning the git repo
    git clone https://github.com/sjoset/sbpy.git
    cd sbpy
    pip install -e .

### install other libraries used
## Vectorial model
    conda install dill plotly

### optional libraries for Abel transformations
    pip install PyAbel


### this allows us to edit sbpy source in place and run it without reinstalling every change
    python setup.py develop --user
