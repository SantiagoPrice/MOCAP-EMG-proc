# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 11:49:56 2023

@author: UTAIL
"""
from setuptools import find_packages, setup

setup(
    name='EMGMOCAPproc',    packages=find_packages(include=['EMGMOCAPproc']), 
    version='0.1.0',
    description='My first Python library',
    author='Me',\
    license='MIT',
    install_requires = ['c3d','pickleshare','matplotlib','PIP'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='Tests',
)