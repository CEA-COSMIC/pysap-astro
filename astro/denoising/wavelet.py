# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""WAVELET.

The module contains functions for wavelet decomposition.

"""

import numpy as np
from pysap import load_transform


def decompose(data, n_scales=4):
    """Decompose.

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
    >>> data = np.arange(9).reshape((3, 3)) * 0.1
    >>> decompose(data)
    array([[[-1.50000006e-01, -1.12500034e-01, -7.50000030e-02],
            [-3.75000238e-02, -2.98023224e-08,  3.74999642e-02],
            [ 7.49999881e-02,  1.12499952e-01,  1.49999976e-01]],

          [[-1.56250030e-01, -1.17187500e-01, -7.81250298e-02],
            [-3.90625000e-02,  0.00000000e+00,  3.90625000e-02],
            [ 7.81250000e-02,  1.17187500e-01,  1.56250000e-01]],

          [[-5.85937500e-02, -4.39453125e-02, -2.92968750e-02],
            [-1.46484375e-02,  0.00000000e+00,  1.46484375e-02],
            [ 2.92968750e-02,  4.39453125e-02,  5.85937500e-02]],

          [[ 3.64843786e-01,  3.73632848e-01,  3.82421911e-01],
            [ 3.91210973e-01,  4.00000036e-01,  4.08789098e-01],
            [ 4.17578161e-01,  4.26367223e-01,  4.35156286e-01]]])

    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError('Input data must be a 2D numpy array.')

    if not isinstance(n_scales, int) or n_scales < 1:
        raise TypeError('n_scales must be a positive integer.')

    trans_name = 'BsplineWaveletTransformATrousAlgorithm'
    trans = load_transform(trans_name)(
        nb_scale=n_scales,
        padding_mode="symmetric",
    )
    trans.data = data
    trans.analysis()

    res = np.array(trans.analysis_data, dtype=np.float)

    return res


def recombine(data):
    """Recombine.

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
