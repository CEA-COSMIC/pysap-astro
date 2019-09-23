# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""UNIT TESTS FOR DENOISING

This module contains unit tests for the astro.denoising module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from ..denoising import *


class DenoiseTestCase(TestCase):

    def setUp(self):

        np.random.seed(1)
        self.data1 = np.random.randn(3, 3)
        self.res1 = np.array([[-0.08110572, -0.07125647, -0.06140723],
                              [-0.08164661, -0.07204223, -0.06243785],
                              [-0.06603668, -0.05656169, -0.04708671]])

    def tearDown(self):

        self.data1 = None
        self.res1 = None

    def test_denoise(self):

        npt.assert_almost_equal(denoise.denoise(self.data1), self.res1,
                                err_msg='Incorrect denoising')


class NoiseTestCase(TestCase):

    def setUp(self):

        np.random.seed(0)
        self.data1 = np.random.ranf((3, 3))

    def tearDown(self):

        self.data1 = None

    def test_sigma_clip(self):

        npt.assert_array_equal(noise.sigma_clip(self.data1),
                               (0.6415801460355164, 0.17648980804276407),
                               err_msg='Incorrect sigma clipping')

        npt.assert_raises(TypeError, noise.sigma_clip, 1)

        npt.assert_raises(TypeError, noise.sigma_clip, self.data1, 1.0)

        npt.assert_raises(TypeError, noise.sigma_clip, self.data1, -1)

    def test_noise_est(self):

        npt.assert_array_equal(noise.noise_est(self.data1),
                               0.11018895815851695,
                               err_msg='Incorrect noise estimate')

        npt.assert_raises(TypeError, noise.noise_est, 1)

        npt.assert_raises(TypeError, noise.noise_est, np.arange(5))

    def test_sigma_scales(self):

        npt.assert_almost_equal(noise.sigma_scales(1),
                                np.array([0.89079631, 0.20066385, 0.0855075]),
                                err_msg='Incorrect sigma scales')

        npt.assert_raises(TypeError, noise.sigma_scales, '1')

        npt.assert_raises(TypeError, noise.sigma_scales, 1, kernel_shape=1)


class WaveletTestCase(TestCase):

    def setUp(self):

        np.random.seed(0)
        self.data1 = np.random.ranf((3, 3))
        self.res1 = np.array([[[-0.06020004, 0.09427285, -0.03005594],
                               [-0.06932276, -0.21794325, -0.02309608],
                               [-0.22873539, 0.17666274, 0.19976479]],
                              [[-0.04426706, -0.02943552, -0.01460403],
                               [-0.0475564, -0.01650959, 0.01453722],
                               [-0.0240097, 0.02943558, 0.08288085]],
                              [[-0.0094105, -0.0110383, -0.01266617],
                               [-0.00393927, -0.00619102, -0.00844282],
                               [0.01415205, 0.0110383, 0.00792474]],
                              [[0.66269112, 0.6613903, 0.66008949],
                               [0.66570163, 0.66429865, 0.6628958],
                               [0.67618024, 0.67463636, 0.67309237]]])

    def tearDown(self):

        self.data1 = None
        self.res1 = None

    def test_decompose(self):

        npt.assert_almost_equal(wavelet.decompose(self.data1), self.res1,
                                err_msg='Incorrect decomposition')

        npt.assert_raises(TypeError, wavelet.decompose, 1)

        npt.assert_raises(TypeError, wavelet.decompose, np.arange(5))

        npt.assert_raises(TypeError, wavelet.decompose, self.data1, 1.0)

        npt.assert_raises(TypeError, wavelet.decompose, self.data1, -1)

    def test_recombine(self):

        npt.assert_almost_equal(wavelet.recombine(self.res1), self.data1,
                                err_msg='Incorrect recombination')

        npt.assert_raises(TypeError, wavelet.recombine, 1)

        npt.assert_raises(TypeError, wavelet.recombine, np.arange(5))
