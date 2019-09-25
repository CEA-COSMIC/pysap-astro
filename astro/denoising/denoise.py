# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Galaxy Image Denoising
"""

import numpy as np
from modopt.signal.noise import thresh
from .noise import noise_est, sigma_scales
from .wavelet import decompose, recombine


def denoise(image, n_scales=4):
    """Denoise

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
    >>> np.random.seed(1)
    >>> data = np.random.randn(3, 3)
    >>> denoise(data)
    array([[-0.08110572, -0.07125647, -0.06140723],
       [-0.08164661, -0.07204223, -0.06243785],
       [-0.06603668, -0.05656169, -0.04708671]])

    """

    sigma_est_scales = sigma_scales(noise_est(image), n_scales)
    weights = (np.array([4] + [3] * sigma_est_scales[:-1].size) *
               sigma_est_scales)
    data_decomp = decompose(image, n_scales)
    data_thresh = np.vstack([thresh(data_decomp[:-1].T, weights).T,
                             data_decomp[-1, None]])

    return recombine(data_thresh)
