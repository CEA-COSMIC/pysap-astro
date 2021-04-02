# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""UNIT TESTS FOR DENOISING.

This module contains unit tests for the astro.denoising module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from pysap.plugins.astro.denoising import denoise, noise, wavelet


class DenoiseTestCase(TestCase):

    def setUp(self):

        np.random.seed(1)
        self.data1 = np.arange(9).reshape((3, 3)) * 0.1
        self.res1 = np.array([
            [0.15000001, 0.21250004, 0.27500001],
            [0.39121097, 0.40000004, 0.4087891],
            [0.52500004, 0.58750004, 0.65000004]
        ])

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

        self.data1 = np.arange(9).reshape((3, 3)) * 0.1
        self.res1 = np.array([
            [
                [-1.50000006e-01, -1.12500034e-01, -7.50000030e-02],
                [-3.75000238e-02, -2.98023224e-08, 3.74999642e-02],
                [7.49999881e-02, 1.12499952e-01, 1.49999976e-01],
            ],
            [
                [-1.56250030e-01, -1.17187500e-01, -7.81250298e-02],
                [-3.90625000e-02, 0.00000000e+00, 3.90625000e-02],
                [7.81250000e-02, 1.17187500e-01, 1.56250000e-01],
            ],
            [
                [-5.85937500e-02, -4.39453125e-02, -2.92968750e-02],
                [-1.46484375e-02, 0.00000000e+00, 1.46484375e-02],
                [2.92968750e-02, 4.39453125e-02, 5.85937500e-02],
            ],
            [
                [3.64843786e-01, 3.73632848e-01, 3.82421911e-01],
                [3.91210973e-01, 4.00000036e-01, 4.08789098e-01],
                [4.17578161e-01, 4.26367223e-01, 4.35156286e-01],
            ],
        ])

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
