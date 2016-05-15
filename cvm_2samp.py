#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__  = "William Hanlon"
__email__   = "whanlon@cosmic.utah.edu"
__version__ = "1.0.0"

import numpy as np
from scipy.interpolate import interp1d

class cvm_2samp:
    """Performs two sample Cramér-von Mises test on two distributions: x and y.

    See "On the Distribution of the Two-Sample Cramér-von Mises Criterion",
    T.W. Anderson
    (http://projecteuclid.org/download/pdf_1/euclid.aoms/1177704477) for a
    description of the test."""

    # table of test statistics and lim_{n->\infinity} P(n\omega^2 <= z)
    # table extracted from "Asymptotic Theory of Certain 'Goodness of Fit'
    # Criteria Based on Stochastic Processes", Anderson and Darling, p. 203
    # (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177729437)
    _a1_z = np.linspace(0., 0.99, 100)
    _a1_z = np.append(_a1_z, 0.999)
    _z = [0.00000, 0.02480, 0.02878, 0.03177, 0.03430, 0.03656, 0.03865,
         0.04061, 0.04247, 0.04427, 0.04601, 0.04772, 0.04939, 0.05103,
         0.05265, 0.05426, 0.05586, 0.05746, 0.05904, 0.06063, 0.06222,
         0.06381, 0.06541, 0.06702, 0.06863, 0.07025, 0.07189, 0.07354,
         0.07521, 0.07690, 0.07860, 0.08032, 0.08206, 0.08383, 0.08562,
         0.08744, 0.08928, 0.09115, 0.09306, 0.09499, 0.09696, 0.09896,
         0.10100, 0.10308, 0.10520, 0.10736, 0.10956, 0.11182, 0.11412,
         0.11647, 0.11888, 0.12134, 0.12387, 0.12646, 0.12911, 0.13183,
         0.13463, 0.13751, 0.14046, 0.14350, 0.14663, 0.14986, 0.15319,
         0.15663, 0.16018, 0.16385, 0.16765, 0.17159, 0.17568, 0.17992,
         0.18433, 0.18892, 0.19371, 0.19870, 0.20392, 0.20939, 0.21512,
         0.22114, 0.22748, 0.23417, 0.24124, 0.24874, 0.25670, 0.26520,
         0.27429, 0.28406, 0.29460, 0.30603, 0.31849, 0.33217, 0.34730,
         0.36421, 0.38331, 0.40520, 0.43077, 0.46136, 0.49929, 0.54885,
         0.61981, 0.74346, 1.16786]
    # create the interpolation function
    _interp_f = interp1d(_z, _a1_z, bounds_error = False)

    def __init__(self, x, y):
        self.x = np.sort(x)   # distribution 1  (vector-like)
        self.y = np.sort(y)   # distribution 2  (vector-like)

        # generate the ecdfs of the distributions
        self.ecdf_x = self.gen_ecdf(self.x)
        self.ecdf_y = self.gen_ecdf(self.y)

        # to plot the ecdf, one can do:
        # plt.step(x, ecdf_x)


    def gen_ecdf(self, x):
        fx = []
        if len(x) == 0:
            return np.array(fx)
        n  = 0
        for s in x:
            n += 1
            fx.append(n)

        return np.array(fx)/float(n)


    def eval_ecdf(self, vx, ecdf, x):
        """Given a distribution of x values and the ecdf of the distribution,
        return the ECDF evaluated at x."""
        if len(vx) != len(ecdf):
            raise ValueError('eval_ecdf: distribution and ecdf lengths don\'t match') 

        i = np.searchsorted(vx, x, side = 'right') - 1
        if i < 0:
            return 0.
        else:
            return ecdf[i]

    def eval(self):
        """The CVM test statistic and p-value are computed.
        
        Function returns the CVM test statistic, the test statistic adjusted to
        the limiting value (where n, m -> infinity), and p-value."""

        T = 0.  # the test statistic
        N = float(len(self.x))
        M = float(len(self.y))

        if N == 0 or M == 0:
            raise ValueError('cvm: empty vector')

        s1 = 0.
        for ex in self.x:
            s1 += (self.eval_ecdf(self.x, self.ecdf_x, ex) -
                    self.eval_ecdf(self.y, self.ecdf_y, ex))**2
        
        s2 = 0.
        for ey in self.y:
            s2 += (self.eval_ecdf(self.x, self.ecdf_x, ey) -
                    self.eval_ecdf(self.y, self.ecdf_y, ey))**2

        # the CVM test statistic
        T = N*M/(N + M)**2*(s1 + s2)

        # the expected value of T (under the null hypothesis)
        expT = 1./6. + 1./(6.*(M + N))

        # the variance of T
        varT = 1./45.*(M + N + 1.)/(M + N)**2*\
                (4.*M*N*(M + N) - 3.*(M**2 + N**2) - 2.*M*N)/(4.*M*N)

        # adjust T so that its significance can be computed using the limiting
        # distribution
        limitT = (T - expT)/np.sqrt(45.*varT) + 1./6.


        # p-value for this test statistic
        if limitT > self._z[-1]:
            p = 0.
        else:
            p = 1. - self._interp_f(limitT)

        return T, limitT, p
