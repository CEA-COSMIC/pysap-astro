# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Galaxy Image Denoising."""

import numpy as np
from modopt.signal.noise import thresh
from pysap.plugins.astro.denoising.noise import noise_est, sigma_scales
from pysap.plugins.astro.denoising.wavelet import decompose, recombine


def denoise(image, n_scales=4):
    """Denoise.

    This function provides a denoised version of the input image.

    Parameters
    ----------
    image : np.ndarray
        Input image
    n_scales : int
        Number of wavelet scales to use

    Returns
    -------
    np.ndarray
        Denoised image

    Examples
    --------
    >>> import numpy as np
    >>> from pysap.astro.denoising.denoise import denoise
    >>> data = np.arange(9).reshape((3, 3)) * 0.1
    >>> denoise(data)
    array([[0.15000001, 0.21250004, 0.27500001],
           [0.39121097, 0.40000004, 0.4087891 ],
           [0.52500004, 0.58750004, 0.65000004]])

    """
    sigma_est_scales = sigma_scales(noise_est(image), n_scales)
    weights = (
        np.array([4] + [3] * sigma_est_scales[:-1].size) * sigma_est_scales
    )
    data_decomp = decompose(image, n_scales)
    data_thresh = np.vstack([
        thresh(data_decomp[:-1].T, weights).T,
        data_decomp[-1, None]
    ])

    return recombine(data_thresh)
