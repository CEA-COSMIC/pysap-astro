# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Noise.

The module contains functions for estimating the noise in images.

"""

import numpy as np
from sf_tools.image.stamp import FetchStamps
from pysap.plugins.astro.denoising.wavelet import decompose


def sigma_clip(data, n_iter=3):
    """Sigma Clipping.

    Perform iterative sigma clipping on input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    n_iter : int, optional
        Number of iterations, default is 3

    Returns
    -------
    tuple
        mean and standard deviation of clipped sample

    Raises
    ------
    TypeError
        For invalid input data type
    TypeError
        For invalid input n_iter type

    Examples
    --------
    >>> import numpy as np
    >>> from pysap.astro.denoising.noise import sigma_clip
    >>> np.random.seed(0)
    >>> data = np.random.ranf((3, 3))
    >>> sigma_clip(data)
    (0.6415801460355164, 0.17648980804276407)

    """
    if not isinstance(data, np.ndarray):
        raise TypeError('Input data must be a numpy array.')

    if not isinstance(n_iter, int) or n_iter < 1:
        raise TypeError('n_iter must be a positive integer.')

    for _iter in range(n_iter):
        if _iter == 0:
            clipped_data = data
        else:
            clipped_data = data[np.abs(data - mean) < (3 * sigma)]
        mean = np.mean(clipped_data)
        sigma = np.std(clipped_data)

    return mean, sigma


def noise_est(data, n_iter=3):
    """Noise Estimate.

    Estimate noise standard deviation of input data using smoothed median.

    Parameters
    ----------
    data : np.ndarray
        Input 2D-array
    n_iter : int, optional
        Number of sigma clipping iterations, default is 3

    Returns
    -------
    float
        Noise standard deviation

    Raises
    ------
    TypeError
        For invalid input data type

    Examples
    --------
    >>> import numpy as np
    >>> from pysap.astro.denoising.noise import noise_est
    >>> np.random.seed(0)
    >>> data = np.random.ranf((3, 3))
    >>> noise_est(data)
    0.11018895815851695

    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError('Input data must be a 2D numpy array.')

    ft_obj = FetchStamps(data, pixel_rad=1, all=True, pad_mode='edge')
    median = ft_obj.scan(np.median).reshape(data.shape)
    mean, sigma = sigma_clip(data - median, n_iter)

    correction_factor = 0.972463

    return sigma / correction_factor


def sigma_scales(sigma, n_scales=4, kernel_shape=(51, 51)):
    """Sigma Scales.

    Get rescaled sigma values for wavelet decomposition.

    Parameters
    ----------
    sigma : float
        Noise standard deviation
    n_scales : int, optional
        Number of wavelet scales, default is 4
    kernel_shape : tuple, list or np.ndarray, optional
        Shape of dummy image kernel

    Returns
    -------
    np.ndarray
        Rescaled sigma values not including coarse scale

    Raises
    ------
    TypeError
        For invalid sigma type
    TypeError
        For invalid kernel_shape type

    Examples
    --------
    >>> from pysap.astro.denoising.noise import sigma_scales
    >>> sigma_scales(1)
    array([0.89079631, 0.20066385, 0.0855075 ])

    """
    if not isinstance(sigma, (int, float)):
        raise TypeError('Input sigma must be an int or a float.')

    if not isinstance(kernel_shape, (tuple, list, np.ndarray)):
        raise TypeError('kernel_shape must be a tuple, list or numpy array.')

    kernel_shape = np.array(kernel_shape)
    kernel_shape += kernel_shape % 2 - 1

    dirac = np.zeros(kernel_shape, dtype=float)
    dirac[tuple(zip(kernel_shape // 2))] = 1.

    return (
        float(sigma) * np.linalg.norm(
            decompose(dirac, n_scales=n_scales), axis=(1, 2)
        )[:-1]
    )
