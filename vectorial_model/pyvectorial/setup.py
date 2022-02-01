#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='pyvectorial',
      version='0.1',
      description='Python Vectorial Model',
      author='Shawn Oset',
      author_email='szo0032@auburn.edu',
      url='https://www.github.com/sjoset/aucomet',
      # packages=find_packages(where='src'),
      # package_dir={'': 'src'}
      packages=['pyvectorial'],
      # py_modules=['pyvectorial.py']
     )
