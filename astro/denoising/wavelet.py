# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Noise

The module contains functions for estimating the noise in images.

"""

import numpy as np
from pysap import load_transform


def decompose(data, n_scales=4):
    """ Decompose

    Obtain the wavelet decomposition of the input date using an isotropic
    undecimated wavelet transform.

    Parameters
    ----------
    data : np.ndarray
        Input 2D-array
    n_scales : int, optional
        Number of wavelet scales, default is 4

    Returns
    -------
    np.ndarray
        Wavelet decomposition 3D-array

    Raises
    ------
    TypeError
        For invalid input data type
    TypeError
        For invalid input n_scales type

    Examples
    --------
    >>> import numpy as np
    >>> from pysap.astro.denoising.wavelet import decompose
    >>> np.random.seed(0)
    >>> data = np.random.ranf((3, 3))
    >>> decompose(data)
    array([[[-0.06020004,  0.09427285, -0.03005594],
            [-0.06932276, -0.21794325, -0.02309608],
            [-0.22873539,  0.17666274,  0.19976479]],

           [[-0.04426706, -0.02943552, -0.01460403],
            [-0.0475564 , -0.01650959,  0.01453722],
            [-0.0240097 ,  0.02943558,  0.08288085]],

           [[-0.0094105 , -0.0110383 , -0.01266617],
            [-0.00393927, -0.00619102, -0.00844282],
            [ 0.01415205,  0.0110383 ,  0.00792474]],

           [[ 0.66269112,  0.6613903 ,  0.66008949],
            [ 0.66570163,  0.66429865,  0.6628958 ],
            [ 0.67618024,  0.67463636,  0.67309237]]])

    """

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError('Input data must be a 2D numpy array.')

    if not isinstance(n_scales, int) or n_scales < 1:
        raise TypeError('n_scales must be a positive integer.')

    trans_name = 'BsplineWaveletTransformATrousAlgorithm'
    trans = load_transform(trans_name)(nb_scale=n_scales,
                                       padding_mode="symmetric")
    trans.data = data
    trans.analysis()

    res = np.array(trans.analysis_data, dtype=np.float)

    return res


def recombine(data):
    """ Recombine

    Recombine wavelet decomposition.

    Parameters
    ----------
    data : np.ndarray
        Input 3D-array

    Returns
    -------
    np.ndarray
        Recombined 2D-array

    Raises
    ------
    TypeError
        For invalid input data type

    Examples
    --------
    >>> import numpy as np
    >>> from pysap.astro.denoising.wavelet import recombine
    >>> np.random.seed(0)
    >>> data = np.random.ranf((4, 3, 3))
    >>> recombine(data)
    array([[2.65508069, 2.89877487, 2.52493858],
           [2.17664192, 2.58496449, 1.95360968],
           [1.21142489, 1.57070222, 2.55727139]])

    """

    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise TypeError('Input data must be a 3D numpy array.')

    return np.sum(data, axis=0)
