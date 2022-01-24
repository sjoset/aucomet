# aucomet

Repository for various comet and space science related work

## Environment Setup for dev version of sbpy

### Create conda environment
    conda create -n sbpy_dev
    conda activate sbpy_dev

### install astropy as per the docs, which at the time of writing was
    conda install astropy
    conda install -c astropy -c defaults scipy h5py beautifulsoup4 html5lib bleach pyyaml pandas sortedcontainers pytz matplotlib setuptools mpmath bottleneck jplephem asdf
    pip3 install pytest-astropy sphinx-astropy

### install astropy by cloning the git repo
    git clone https://github.com/sjoset/sbpy.git
    cd sbpy

### this allows us to edit sbpy source in place and run it without reinstalling every change
    python setup.py develop --user
