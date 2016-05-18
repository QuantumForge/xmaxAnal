#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate
import matplotlib as mpl

import xmaxFit
import cvm_2samp

class xmaxCompat:
    def __init__(self,
            fnameHanlon = 'dump_hanlon_xmaxAcceptance_reconOnly.txt.bz2',
            fnameIkeda  = 'dump_ikeda_xmaxAcceptance.txt.bz2'):
        self.dfHanlon = pd.read_csv(fnameHanlon,
                  delim_whitespace=True, header=None, names=['date', 'time',
                      'weight', 'eThrown', 'xmaxThrown', 'eRecon', 'xmaxRecon'],
                  na_values=['-1'], parse_dates=[[0, 1]])
        self.dfIkeda = pd.read_csv(fnameIkeda,
                delim_whitespace=True, header=None, names=['date', 'time',
                    'weight', 'eThrown', 'xmaxThrown', 'eRecon', 'xmaxRecon'],
                na_values=['-1'], parse_dates=[[0, 1]])
        self.hxmf = None
        self.ixmf = None

    def std_weighted(self, data, weights = None):
        """compute the weighted standard deviation of a given set of data."""
        if weights is None:
            return np.std(data)
        mu = np.average(data, weights=weights)
        n = np.sum(np.power(data - mu, 2.)*weights)
        d = np.sum(weights)
        if d > 0.:
            return np.sqrt(n/d)
        return 0.
        

    def sample(self, func, funcArgs, nsamp=1000, dataRange=[500., 1300.]):
        """Sample an xmax function by acceptance-rejection method and return
        nsamp xmax values randomly distributed by func."""

        # scan over range for the function maximum
        x = np.linspace(dataRange[0], dataRange[1], 1000)
        fmax = np.max(func(x, *funcArgs))
        # list of sampled xmax
        val = []
        # acceptance-rejection method sampling. I need to learn proper MCMC
        # techniques...
        n = 0
        while (True):
            r = np.random.rand(2)
            rx = dataRange[0] + (dataRange[1] - dataRange[0])*r[0]
            ry = fmax*r[1]    # lower bound is 0
            if (ry < func(rx, *funcArgs)):
                val.append(rx)
                n += 1
            if (n >= nsamp):
                return np.array(val)

    def main(self, energyLow = 18.2, energyHigh = 18.4):
        mpl.rcParams['figure.figsize'] = (20., 10.)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = \
                plt.subplots(3, 2, sharex=True)
        
        print 'Energy range: [', energyLow, energyHigh, '] log10(E/eV)'
        # read in hanlon xmax data. my data is unweighted
        hdata = \
            self.dfHanlon.xmaxRecon[(self.dfHanlon['eThrown']>=energyLow) & \
            (self.dfHanlon['eThrown']<energyHigh)]
        print 'Hanlon Xmax'
        # use the mean of energy bin to set the fit starting parameters
        self.hxmf = xmaxFit.xmaxFit(hdata, (energyLow + energyHigh)/2.)
        #print hxmf.pname
        #print hxmf.pfit
        #print hxmf.perr

        # plot weighed distribution and fit
        ax1.hist(hdata, bins=80, range=[500., 1300.], histtype='stepfilled')
        funcX = np.linspace(500., 1300., 100)
        funcY = self.hxmf.func(funcX, *self.hxmf.pfit)

        ax1.plot(funcX, funcY, linewidth=2., linestyle='dashed',
                color='black')
        ax1.set_xlabel('Hanlon $X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax1.set_ylabel('$N$')
        ax1.grid()

        # read in ikeda xmax data. his data is weighted, so we must weight
        # the distribution, fit it, then sample it so we can generate an
        # undistorted ECDF.
        idata = \
          self.dfIkeda.xmaxRecon[(self.dfIkeda['eThrown']>=energyLow) & \
                (self.dfIkeda['eThrown']<energyHigh) & \
                (self.dfIkeda['eRecon'] > 0)]
        iweight = \
          self.dfIkeda.weight[(self.dfIkeda['eThrown']>=energyLow) & \
                (self.dfIkeda['eThrown']<energyHigh) & \
                (self.dfIkeda['eRecon'] > 0)]

        print '\n'
        print 'Weighted distribution moments:'
        print '<Hanlon Xmax> =', np.average(hdata), 'rms(Xmax) =', \
                self.std_weighted(hdata)
        print '<Ikeda Xmax> =', np.average(idata, weights=iweight), \
                'rms(Xmax) =', self.std_weighted(idata, iweight)

        print '\n'
        print 'Ikeda Xmax'
        # use the mean of energy bin to set the fit starting parameters
        # recall Ikeda-san's data is weighted and weights must be applied here
        # for the fit to make sense
        self.ixmf = xmaxFit.xmaxFit(idata, (energyHigh + energyLow)/2.,
                weights=iweight)
        #print ixmf.pname
        #print ixmf.pfit
        #print ixmf.perr

        # plot the weighted distribution and fit
        ax2.hist(idata, bins=80, range=[500., 1300.], weights=iweight,
                histtype='stepfilled', color='red')
        funcX = np.linspace(500., 1300., 100)
        funcY = self.ixmf.func(funcX, *self.ixmf.pfit)

        ax2.plot(funcX, funcY, linewidth=2., linestyle='dashed',
                color='black')
        ax2.set_xlabel('Ikeda $X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax2.set_ylabel('$N$')
        ax2.grid()

        # now generate a sample of randomly drawn xmax from the fitted
        # distributions. this removes weighting bias
        hXmaxSamples, iXmaxSamples = self.drawSamples(1000)

        ax3.hist(hXmaxSamples, bins=80, range=[500., 1300.],
                histtype='stepfilled')
        ax3.set_xlabel('Hanlon sampled $X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax3.set_ylabel('$N$')
        ax3.grid()
        
        ax4.hist(iXmaxSamples, bins=80, range=[500., 1300.],
                histtype='stepfilled', color='red')
        ax4.set_xlabel('Ikeda sampled $X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax4.set_ylabel('$N$')
        ax4.grid()

        # means of the functions, not the samples. just to make sure the 
        # continuous function means and rms are close to what we expect from
        # the weighted distributions.
        #
        # integral for first moment
        mf, _ = integrate.quad(self.hxmf.func, 500., 1300.,
                args=(self.hxmf.pfit[0], self.hxmf.pfit[1], self.hxmf.pfit[2],
                    self.hxmf.pfit[3], 1))
        # normalization required for first moment
        mf_norm, _ = integrate.quad(self.hxmf.func, 500., 1300.,
                args=(self.hxmf.pfit[0], self.hxmf.pfit[1], self.hxmf.pfit[2],
                    self.hxmf.pfit[3]))
    
        hMeanFunc = mf/mf_norm
        print '\n'
        print '<Hanlon Xmax> of fit function = ', hMeanFunc

        # integral for first moment
        mf, _ = integrate.quad(self.ixmf.func, 500., 1300.,
                args=(self.ixmf.pfit[0], self.ixmf.pfit[1], self.ixmf.pfit[2],
                    self.ixmf.pfit[3], 1))
        # normalization required for first moment
        mf_norm, _ = integrate.quad(self.ixmf.func, 500., 1300.,
                args=(self.ixmf.pfit[0], self.ixmf.pfit[1], self.ixmf.pfit[2],
                    self.ixmf.pfit[3]))

        iMeanFunc = mf/mf_norm
        print '<Ikeda Xmax> of the fit function = ', iMeanFunc

        # for thrown Xmax distributions there is a small systematic shift
        # between these distributions. we want to test the hypothesis that
        # the samples are drawn from the sample parent distribution after
        # shifting for any small difference in the means (we believe measuring
        # only the means in the presence of non-Gaussian tails is not sufficient
        # information to state the compatibility of the data).
        #
        # measure the shift of the means from the fitted functions and shift by
        # that amount. we should try this without shifting as well...

        # save the xmaxShift so that we can repeatedly draw samples and create
        # a distribution of CvM test statistics using tdist
        self.xmaxShift = hMeanFunc - iMeanFunc
        print 'shift =', self.xmaxShift

        # means of the two sampled distributions
        mean_hXmaxSamples= np.mean(hXmaxSamples)
        mean_iXmaxSamples= np.mean(iXmaxSamples)
        #shift = mean_hXmaxSamples - mean_iXmaxSamples
        print '\n'
        print ' <Hanlon Xmax> of sampled distributions =', mean_hXmaxSamples
        print ' <Ikeda Xmax> of sampled distributions =',  mean_iXmaxSamples
        
        # use a Cramer-von Mises 2 sample test to test the compatibility of the
        # data. there is most likely a systematic bias bewteen the distributions
        # so we may have to look at shifting them as well...
        cvm = cvm_2samp.cvm_2samp(hXmaxSamples, iXmaxSamples + self.xmaxShift)

        # evaluate the p-value.
        # H0: hXmaxSamples and iXmaxSamples are both drawn from the same parent
        # population
        # Ha: hXmaxSamples and iXmaxSamples are not both drawn from the
        # same parent population
        # cvm_2samp evaluates the test statistic for the limiting distribution
        # as the number of samples approaches infinity. report the true test
        # statistic measured and the p-value of the limiting distribution.
        # (we have many. many samples here)
        (T, _, p) = cvm.eval()
        print '\n'
        print 'CvM test statistic:', T
        print 'p-value:', p

        # this test is much more sensitive in the tails than I expected...

        # plot ECDFs of the sampled distributions. input xmax distributions
        # aren't sorted, so provide sorted lists for plotting.
        ax5.plot(np.sort(hXmaxSamples), cvm.ecdf_x, linewidth=2.,
                label='hanlon')
        ax5.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax5.set_ylabel('Cumulative probability')
        ax5.legend()
        ax5.grid()
        ax6.plot(np.sort(iXmaxSamples), cvm.ecdf_y, linewidth=2., color='red')
        ax6.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax6.set_ylabel('Cumulative probability')
        ax6.grid()

        plt.tight_layout()
        plt.show()

    def drawSamples(self, nsamp = 1000):
        """Draw nsamp samples from the analytically fitted xmax distributions
        (this removes weighting bias from the distributions). Return the
        randomly sampled distributions from the Hanlon and Ikeda analytical
        functions."""
        hXmaxSamples = self.sample(self.hxmf.func, self.hxmf.pfit, nsamp=nsamp,
                dataRange=[500., 1300.])
        iXmaxSamples = self.sample(self.ixmf.func, self.ixmf.pfit, nsamp=nsamp,
                dataRange=[500., 1300.])
        return hXmaxSamples, iXmaxSamples

    def tdist(self):
        """the true value of the CvM test statistic is not known because we fit
        the xmax distributions and then sample them. repeatedly draw samples and
        create a distribution of the CvM test statistic. we can then generate a
        confidence interval and state with a certain percentage what is the the
        probability the two distributions are statistically compatible."""
        self.t = []
        self.pval = []
        for i in range(1000):
            hXmaxSamples, iXmaxSamples = self.drawSamples(1000)
            cvm = cvm_2samp.cvm_2samp(hXmaxSamples,
                    iXmaxSamples + self.xmaxShift)
            (T, _, p) = cvm.eval()
            self.t.append(T)
            self.pval.append(p)

if __name__ == '__main__':
    xmaxCompat().main()
