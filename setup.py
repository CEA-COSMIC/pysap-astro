#! /usr/bin/env python
##########################################################################
# pySAP - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
from setuptools import setup, find_packages


# Global parameters
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering"]
AUTHOR = """
Antoine Grigis <antoine.grigis@cea.fr>
Samuel Farrens <samuel.farrens@cea.fr>
Jean-Luc Starck <jl.stark@cea.fr>
Philippe Ciuciu <philippe.ciuciu@cea.fr>
"""

# Write setup
setup(
    name="pysap-astro",
    description="Python Sparse data Analysis Package external ASTRO plugin.",
    long_description=("Python Sparse data Analysis Package external ASTRO "
                      "plugin."),
    license="CeCILL-B",
    classifiers="CLASSIFIERS",
    author=AUTHOR,
    author_email="XXX",
    version="0.0.0",
    url="https://github.com/CEA-COSMIC/pysap-astro",
    packages=find_packages(),
    platforms="OS Independent",
    setup_requires=['pytest-runner', ],
    tests_require=['pytest>=5.0.1', 'pytest-cov>=2.7.1', 'pytest-pep8'],
)
