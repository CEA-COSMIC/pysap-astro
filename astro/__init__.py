# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Module defines all the Astro related plugins."""

__all__ = ['deconvolution', 'denoising']

from . import *
from .deconvolution import *
from .denoising import *

version_info = (0, 0, 0)
__version__ = '.'.join(str(c) for c in version_info)
