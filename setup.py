#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='NNP-pl',
    version='0.0.0',
    description='Describe Your Cool Project',
    author='Ray Schireman',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/rschireman/ANI-NNP-pl',
    install_requires=['pytorch-lightning', 'torchani'],
    packages=find_packages(),
)

