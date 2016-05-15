__author__ = "William Hanlon"
__email__ = "whanlon@cosmic.utah.edu"
__version__ = "1.0.0"

import numpy as np
import numpy.ma as ma
from scipy import optimize
from scipy.special import erfc

class xmaxFit:
    """Class to fit an Xmax distribution histogram for a given energy bin to an
    analytical function described in "Comparison of the moments of the Xmax
    distribution predicted by different cosmic ray shower simulation models",
    Peixoto, C.J.T, et al., (2013), http://arxiv.org/abs/1301.5555.
   
    User provides a numpy histogram of Xmax data for an energy bin,
    mean energy of the bin (required to seed the fitter), and range over
    which to fit.
    """

    def __init__(self, data, energyMean, bins = 80, dataRange = [500., 1300.],
            weights = None, fitMin = 500., fitMax = 1300.):
        # energy in log10(E/ev)
        self.energyMean = energyMean

        # range in g/cm^2 over which to fit
        self.fitMin = fitMin
        self.fitMax = fitMax

        # create a histogram of the data
        hist, binEdges = np.histogram(data, bins = bins, range = dataRange,
                weights = weights)

        # create of list of bin centers, these are the x values of the
        # fit function
        binCent = (np.array(binEdges[:-1]) + np.array(binEdges[1:]))/2.

        # histogram error for now is sqrt(n).
        histErr = np.sqrt(hist)

        # parameter names and order
        self.pname = ('N', 't0', 'lambda', 'sigma')

        # starting values for parameters are specified in the paper. note
        # this is for protons, so one term is missing
        p0 = [np.max(hist),
                53.32*energyMean - 283.93,
                -1.73*energyMean + 82.69,
                0.06*energyMean + 35.99]
        
        # we can't fit with empty bins, so fill lists with that only
        # correspond to bins with frequency > 0
        x = []
        y = []
        yerr = []
        for i in range(len(hist)):
            if hist[i] > 0:
                x.append(binCent[i])
                y.append(hist[i])
                yerr.append(histErr[i])

        self.pfit, self.pcov = optimize.curve_fit(self.func, x,
                y, p0 = p0, sigma = yerr)
        perr_ = []
        chi2 = 0.
        for i in range(len(self.pfit)):
            perr_.append(np.sqrt(self.pcov[i][i]))
            print self.pname[i], '= ', self.pfit[i], '+/-', perr_[i]
        self.perr = np.array(perr_)

    # break up the analytical function into constiuent parts for
    # clarity.
    def y(self, t, t0, lambdat, sigma):
        return (t0 - t)/lambdat - np.power(sigma/lambdat, 2.)/2.
    def z(self, t, t0, lambdat, sigma):
        return (t0 - t + np.power(sigma, 2.)/lambdat)/np.sqrt(2.)/sigma 
        
    def func(self, t, n, t0, lambdat, sigma):
        """Analytical function to describe an Xmax distribution at depth t.
        Parameters are n, t0, lambdat, and sigma."""
        return n*np.exp(self.y(t, t0, lambdat, sigma))* \
                erfc(self.z(t, t0, lambdat, sigma))
