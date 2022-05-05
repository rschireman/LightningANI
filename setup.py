#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='lightning_ani',
    version='0.0.1',
    description='Describe Your Cool Project',
    author='Ray Schireman',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/rschireman/LightningANI',
    install_requires=['pytorch-lightning', 'torchani'],
    packages=find_packages(),
)

