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
from pathlib import Path
from setuptools import setup, find_packages

# Set the package release version
major = 0
minor = 0
patch = 3
version = '.'.join(str(value) for value in (major, minor, patch))

# Global parameters
CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering']
AUTHOR = """
Antoine Grigis <antoine.grigis@cea.fr>
Samuel Farrens <samuel.farrens@cea.fr>
Jean-Luc Starck <jl.stark@cea.fr>
Philippe Ciuciu <philippe.ciuciu@cea.fr>
"""

# Source package description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.rst').read_text()

# Write setup
setup(
    name='pysap-astro',
    description='Python Sparse data Analysis Package external ASTRO plugin.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    license='CeCILL-B',
    classifiers='CLASSIFIERS',
    author=AUTHOR,
    author_email='samuel.farrens@cea.fr',
    version=version,
    url='https://github.com/CEA-COSMIC/pysap-astro',
    packages=find_packages(),
    platforms='OS Independent',
    install_requires=['sf_tools>=2.0.4'],
    setup_requires=['pytest-runner', ],
    tests_require=[
        'pytest>=6.2.2',
        'pytest-cov>=2.11.1',
        'pytest-pycodestyle>=2.2.0',
        'pytest-pydocstyle>=2.2.0',
    ],
)
